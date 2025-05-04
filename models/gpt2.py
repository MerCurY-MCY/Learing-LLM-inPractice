import torch
from torch import nn
from attention.attention import LayerNorm


#功能与一个线性层完全相同，实现的操作是output=input⋅weight+bias
#gpt2中的Conv1D主要是用来对QKV向量的计算和最终的输出做线性投影
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
        self.resid_dropout = torch.nn.Dropout(config.presid_dropout)  #ffw层dropout
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

    def forward(self, hidden_states,kv_past = None, attn_musk = None, head_musk = None):
        hidden_states = self.c_attn(hidden_states)
        q, k, v = hidden_states.split(self.hidden_size, dim=-1)
        q, k, v = self._split_head(q), self._split_head(k), self._split_head(v)

        #kv cach
        if kv_past is not None:
            k_past, v_past = kv_past
            k = torch.concat((k_past,k),dim=-2)
            v = torch.concat((v_past,v),dim=-2)
        kv_past = (k , v)
        weight = torch.matmul(q, k.transpose(-2, -1))
        if self.scale is not None:
            weight = weight / torch.sqrt(self.hidden_size/self.n_head)
        weight = self._softmax(weight)
        if attn_musk is not None:
            weight += attn_musk
        
        


