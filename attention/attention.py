import math
import torch.nn as nn
import torch


def softmax(x):
    max = torch.max(x, dim=-1, keepdim=True).values
    e_x = torch.exp(x - max)
    softmax_x = e_x/torch.sum(x, dim=-1, keepdim=True)

    return softmax_x


class AttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_multihead == 0
        self.hidden_size = config.hidden_size
        self.dim, self.num_head = config.hidden_size, config.num_multihead

        self.q = nn.Linear(self.hidden_size, self.hidden_size)
        self.k = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def _divideHead(self, x):  # 多头注意力实现
        batch, num_token, dim_feature = x.shape
        # 使用矩阵reshape操作，view函数把矩阵展平，并使用头维度把每个token的hiden向量拆分到不同空间中
        x = x.view(batch, num_token, self.num_head, -1)
        # 交换第二和第三维度的索引，把同一个头指向的所有token的特征聚合到一起形成子矩阵，而原本头数量告诉我们有多少个这样的子矩阵
        return x.permute(0, 2, 1, 3)

    # 如要屏蔽，attention_mask一般为极大负数，head_mask一般为0
    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        q, k, v = self.q(hidden_states), self.k(
            hidden_states), self.v(hidden_states)

        q, k, v = self._divideHead(q), self._divideHead(
            k), self._divideHead(v)  # 把q，k，v切割到不同头的子空间

        weight = torch.matmul(q, k.transpose(2, 3))
        weight = weight/math.sqrt(self.dim/self.num_head)
        if attention_mask:
            weight += attention_mask
        weight = softmax(weight)  # 对最后一个维度的每一行分别归一化
        weight = self.dropout(weight)

        if head_mask:
            weight *= head_mask

        attention = torch.matmul(weight, v)

        '''
        这几步比较难以理解，是合并多头注意力得到最终注意力的过程，依靠的是矩阵的形状变换。

        注意前面我们把矩阵的维度变成了（batch_size,头的数量，token数，特征数）的形状

        现在首先需要把矩阵第二维度和第三维度交换，转换为（batch_size,token数，头的数量，特征数）的形状，这样就把原本属于同一个token的特征重新聚合在一起

        后两个维度共同代表同一个token的特征，现在只需要用view函数把矩阵reshape为（batch_size,token数，头的数量*特征数）就成功多头注意力聚合

        '''
        batch, num_head, num_token, dim_feature = attention.shape

        attention = attention.transpose(1, 2).contiguous().view(
            batch, num_token, num_head*dim_feature)

        return attention


class LayerNorm(nn.Module):
    def __init__(self, norm_size, eps=1e-5):
        super().__init__()
        self.norm_size = norm_size
        if isinstance(self.norm_size, int):
            self.norm_size = (self.norm_size, )

        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(self.norm_size))
        self.weight = nn.Parameter(torch.ones(self.norm_size))

    def _mean(self, feature):
        '''
        feature为多维度张量，输入的norm_size为需要归一化的特征，可为整数可为张量

        一二行个人理解来看，一般可能用于特殊的归一化（比如对于多头注意力合并前QKV矩阵的归一化，此时后两个维度是共同代表一个token的特征的）

        所以第一行通过切片把除了需要归一化的特征之外的维度提取出来，并reshape矩阵，把所有特征统一展平到最后一维，方便进行均值计算

        '''
        temp_shape = list(feature.shape[:-len(self.norm_size)])+[-1]

        _feature = feature.view(*temp_shape)

        # 用最后一个维度的值综合除以最后一个维度的形状（也就是元素数）
        mean = torch.sum(_feature, dim=-1)/_feature.shape[-1]

        # 这里是为了对其均值和输入矩阵维度（因为之前有展平操作），以便后续计算不报错
        for i in range(len(feature.shape)-len(mean.shape)):
            mean = mean.unsqueeze(-1)
        return mean

    def forward(self, x):
        mean = self._mean(x)

        std = (self._mean((x - mean).pow(2) +
               self.eps)).pow(0.5)  # 由于要使用归一化公式，std为分母，+eps防止除以0

        x = (x - mean) / std

        return x * self.weight + self.bias


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.attnlayer = AttentionLayer(config)

        self.layrNorm1 = LayerNorm(self.hidden_size, config.layernorm_eps)

        self.layrNorm2 = LayerNorm(self.hidden_size, config.layernorm_eps)

        self.ffw = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size*4),
            nn.ReLU(),
            nn.Linear(self.hidden_size*4, self.hidden_size)
        )

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.ffw_dropout = nn.Dropout(config.ffw_dropout)

    def forward(self, x):
        attn_out = self.attnlayer(x)
        attn_out = self.attn_dropout(attn_out)
        attn_norm = self.layrNorm1(x + attn_out)  # 残差连接

        ffw_out = self.ffw(attn_norm)
        ffw_out = self.ffw_dropout(ffw_out)
        ffw_norm = self.layrNorm2(attn_norm + ffw_out)

        return ffw_norm


class ExampleConfig():
    def __init__(self):
        self.num_multihead = 3
        self.layernorm_eps = 1e-5
        self.resid_pdrop = 0.1
        self.attention_dropout = 0.1
        self.hidden_size = 12
        self.ffw_dropout = 0.1


def layernorm_sample():
    torch.manual_seed(999)
    x = torch.rand((3, 4, 6))
    normalized_shape = [4, 6]
    norm1 = LayerNorm(normalized_shape)
    norm2 = torch.nn.LayerNorm(normalized_shape)
    print(norm1(x))
    print(norm2(x))


def t_TransformerBlock():
    torch.manual_seed(999)
    config = ExampleConfig()
    trans = TransformerBlock(config)
    q = torch.rand((3, 4, config.hidden_size))
    r = trans(q)
    print(q)
    print(r)


if __name__ == "__main__":
    layernorm_sample()
    t_TransformerBlock()
