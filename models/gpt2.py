import torch
from torch import nn
from attention.attention import LayerNorm


#功能与一个线性层完全相同，实现的操作是output=input⋅weight+bias
#gpt2中的Conv1D主要是用来对QKV向量的计算和最终的输出做线性投影

ACT2FC = {'relu':nn.ReLU, 'gelu': nn.GELU}
class Conv1D(nn.Module):
    def __init__(self, out_dim, in_dim):
        super.__init__()
        weight = torch.zeros(in_dim, out_dim)
        nn.init.normal_(weight, std = 0.02)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    
    def forward(self, x):
        # 权重矩阵在这里被转置是因为torch.nn的线性层里的实际计算公式是output=input⋅weight^T+bias。由于我们自己写了权重向量，为了适配linear函数，需要先转置一次
        return nn.functional.linear(x, self.weight.transpose(0,1), self.bias) 
    
class AttentionLayer(nn.Module):
    def __init__(self, config, scale = False):
        self.configm, self.scale = config, scale
        self.hidden_size, self.n_head, self.n_ctx = config.n_embd, config.n_head, config.n_ctx  #n_ctx表示训练时最大序列窗口
        assert self.hidden_size % self.n_head == 0
        self.c_attn = Conv1D(3 * self.hidden_size, self.hidden_size)   #qkv计算线性投影
        self.c_proj = Conv1D(self.hidden_size, self.hidden_size)  #  输出计算线性投影
        self.attn_dropout = torch.nn.Dropout(config.pattn_dropout)   #注意力层dropout
        self.resid_dropout = torch.nn.Dropout(config.presid_dropout)  #v投影层dropout
        #torch.tril创建下三角矩阵，register_buffer方法创建需要保存的缓存参数（但是不被反向传播更新）
        self.register_buffer = ('bias',torch.tril(torch.ones(self.n_ctx,self.n_ctx)).view(1, 1 , self.n_ctx, self.n_ctx))
    
    def _split_head(self, x):
        b, n, d = x.shape
        x.view(b, n, self.n_head, -1)
        return x.permute(0 , 2, 1, 3)

    def _softmax(self, x):
        max = torch.max(x ,dim=-1, keepdim=True).values  
        ex = torch.exp(x - max)
        return ex / torch.sum(ex, dim=-1, keepdim=True) 

    def forward(self, hidden_states, kv_past = None, attn_mask = None, head_mask = None):
        hidden_states = self.c_attn(hidden_states)
        q, k, v = hidden_states.split(self.hidden_size, dim=-1)
        q, k, v = self._split_head(q), self._split_head(k), self._split_head(v)

        #kv cache
        if kv_past is not None:
            k_past, v_past = kv_past
            k = torch.concat((k_past,k),dim=-2)
            v = torch.concat((v_past,v),dim=-2)
        kv_past = (k , v)


        weight = torch.matmul(q, k.transpose(-2, -1))
        b = self.bias[:, :, k.size(-2) - q.size(-2), k.size(-2)]  #从整体掩码矩阵中从提取当前token对应的掩码矩阵。大小为[q_len,k_len]，与qk初步权重计算后的尺寸相同
        weight = weight * b + -1e5 * (1 - b)
        if self.scale is not None:
            weight = weight / torch.sqrt(self.hidden_size/self.n_head)
        
        if attn_mask is not None:
            weight += attn_mask
        weight = self._softmax(weight)
        weight = self.attn_dropout(weight)

        if head_mask is not None:
            weight *= head_mask

        v = torch.matmul(weight, v)
        
        b, h, n, d = v.shape
        v.permute(0, 2, 1, 3)
        v = v.reshape(b, n, h * d)

        #线性投影层的目的是让模型学习如何为不同头学习到的特征分配权重，从而更好地提取各自头中需要的特征，并这些信息进行转换和融合，映射回嵌入空间
        v = self.c_proj(v)
        v = self.resid_dropout(v)

        return v, kv_past

class TransformerBlock(nn.Module):
    def __init__(self, config, version = 'gpt'):
        super.__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.version = version
        self.norm1 = LayerNorm(config,eps = config.layer_norm_epsilon)
        self.attnlayer = AttentionLayer(config, scale=True)
        self.mlp = nn.Sequential(
            Conv1D(4 * self.hidden_size, self.hidden_size),
            ACT2FC['gelu'],
            Conv1D(self.hidden_size, 4 * self.hidden_size),
            nn.Dropout()
        )
        self.norm2 = LayerNorm(config,eps = config.layer_norm_epsilon)
    
    def forward(self, x ,attn_output=None, kv_past = None, attn_mask = None, head_mask = None):
        if attn_output is None:
            attn_output, kv_past = self.attnlayer(self.norm1(x), kv_past , attn_mask , head_mask ) #先对输入进行层归一化再计算注意力
        attn_output = x + attn_output
        
        ffw_output = self.mlp(self.norm2(attn_output))
        output = attn_output + ffw_output

        return output, kv_past


class GPTModel(nn.Module):
    def __init__(self, config, version = 'gpt'):
        super.__init__()
        self.token_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.posion_embd = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdropout)
        nn.ModuleList([TransformerBlock(config, version) for _ in range (config.n_layer)])
    
    def forward(self, input_ids, attention_mask = None, head_mask = None, position_ids = None, segment_ids = None, kv_pasts = None):
        input_embds= self.token_embd(input_ids)

        if position_ids is None:
            # attention_mask是对输入序列的填充标注。如果输入序列长度达不到预设的训练长度L，模型会对序列进行填充，并在attention_mask里把填充位置标注为0，原始位置标注为1
            position_ids = attention_mask.long().cumsum(-1) - 1  #long 转为长整形，cumsum对最后一个维度累积求和（指定维度 dim 上的第 i 个元素，得到在该维度上从第0个到第i个元素的和）
            position_ids.masked_fill(attention_mask == 0, 1)  #masked_fill_ 是PyTorch中张量的原地操作，用于将张量中满足条件的元素替换为指定的值。
            position_ids = position_ids[:, -input_embds[1]:]


        


