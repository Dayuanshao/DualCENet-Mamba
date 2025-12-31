from net.SS2D import SS2D
from net.transformer_utils import *
from moudle.LG import LGFT
from moudle.CSLA import *

import torch
import torch.nn as nn
from einops import rearrange

class DAF(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(DAF, self).__init__()
        self.num_heads = num_heads

        # 可学习温度参数（初始化为 log(1) = 0）
        self.temperature = nn.Parameter(torch.log(torch.ones(num_heads, 1, 1)))

        # 增强查询路径
        self.q = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=bias),
            nn.GroupNorm(4, dim)
        )

        # 增强键值路径
        self.kv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2, bias=bias),
            nn.GroupNorm(4, dim * 2)
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

        # 空间注意力增强
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.project_out = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=bias),
            nn.GELU(),
            nn.Dropout(0.1),  # 加入 Dropout
            nn.Conv2d(dim, dim, 1, bias=bias)
        )

        self.alpha = nn.Parameter(torch.tensor(0.8))
        self.beta = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, y):
        b, c, h, w = x.shape
        q = self.q(x)
        kv = self.kv(y)
        k, v = kv.chunk(2, dim=1)
        v = v * self.channel_att(v)
        spatial_avg = torch.mean(v, dim=1, keepdim=True)
        spatial_max, _ = torch.max(v, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([spatial_avg, spatial_max], dim=1))
        v = v * spatial_att

        # 多头注意力准备
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        temperature = torch.clamp(self.temperature.exp(), min=0.1, max=10)
        attn = (q @ k.transpose(-2, -1)) * temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        identity = x
        dual_identity = y[:, :c, :, :] if y.shape[1] > c else y

        out = self.project_out(torch.cat([out, identity], dim=1))
        out = out * self.alpha + dual_identity * self.beta

        return out



class LCF(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(LCF, self).__init__()  # 继承Module基类
        self.LGFT = LGFT(dim)  # 特征增强模块（与CDL结构相同）
        self.norm = LayerNorm(dim)  # 层归一化（保持特征稳定性）
        self.DAF = DAF(dim, num_heads, bias=bias)  # 交叉注意力模块
        # self.ss2d = SS2D(d_model=dim, dropout=0, d_state=16)  # 交叉注意力

    def forward(self, x, y):

        x = x + self.DAF(  # 残差连接保留原始信息
            self.norm(x),  # 输入x归一化（query来源）
            self.norm(y)  # 输入y归一化（key/value来源）
        )

        # 阶段2：特征增强处理
        x = self.LGFT(
            self.norm(x)
        )

        return x  # 最终输出（增强后的跨模态特征）


##############################
### I_LCA 模块（交互式交叉注意力）
##############################
class I_CSEB(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_CSEB, self).__init__()

        # 组件初始化（与HV_LCA顺序不同）------------------------------
        self.norm = LayerNorm(dim)  # 层归一化
        self.LGFT = LGFT(dim)  # 特征增强模块
        # self.ss2d = SS2D(d_model=dim*2)
        self.DAF = DAF(dim, num_heads, bias=bias)  # 交叉注意力
        # print(self.ffn.share_memory())
        self.ss2d = SS2D(d_model=dim, dropout=0, d_state=16)  # 交叉注意力

    def forward(self, x, y):
        # 双残差交互流程 ---------------------------------------------
        # 第一阶段：跨模态交互
        x1 = self.ss2d(self.norm(x))
        y1 = self.ss2d(self.norm(y))
        x = x + self.DAF(x1,y1)
        # 第二阶段：独立特征增强（新增残差路径）
        x = x + self.LGFT(self.norm(x))
        return x


