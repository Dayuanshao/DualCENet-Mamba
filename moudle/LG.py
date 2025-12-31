
from net.transformer_utils import *

class LightGrant(nn.Module):
    def __init__(self, dim, ffn_expansion=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion)

        self.in_proj = nn.Conv2d(dim, hidden*2, 1, bias=bias)
        self.norm_in  = nn.InstanceNorm2d(hidden*2, affine=True)
        self.dw = nn.Conv2d(hidden*2, hidden*2, 3, padding=1,
                            groups=hidden*2, bias=bias)
        self.norm_dw = nn.InstanceNorm2d(hidden*2, affine=True)

        # 分支
        self.dw1 = nn.Conv2d(hidden, hidden, 3, padding=1,
                             groups=hidden, bias=bias)
        self.dw2 = nn.Conv2d(hidden, hidden, 3, padding=1,
                             groups=hidden, bias=bias)
        self.act = nn.SiLU(inplace=True)

        # 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden//4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden//4, hidden, 1, bias=False),
            nn.Sigmoid()
        )

        # 门控融合
        self.gate = nn.Conv2d(hidden, hidden, 1)
        self.out_proj = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        residual = x
        x = self.in_proj(x)
        x = self.norm_in(x)
        x = self.dw(x)
        x = self.norm_dw(x)
        x1, x2 = x.chunk(2, 1)

        x1 = x1 + self.act(self.dw1(x1))
        x2 = x2 + self.act(self.dw2(x2))

        # 加通道注意力
        x1 = x1 * self.se(x1)
        x2 = x2 * self.se(x2)

        # 门控融合
        alpha = torch.sigmoid(self.gate(x1))
        x = alpha * x1 + (1-alpha) * x2

        x = self.out_proj(x)
        return x + residual
