import torch
import torch.nn as nn

pi = 3.141592653589793


#  实现 RGB ↔ HVI 颜色空间的双向转换，核心目标是：
class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2))  # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0

    #  正向转换 该模块实现 RGB -> HVI 颜色空间的转换，核心目标是：
    #  将RGB图像分解为H（色相相关）、V（明度相关）、I（亮度）三个通道。
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        # 计算亮度 取RGB三通道的最大值 I 直接取自 value，表示亮度分量。
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        # 计算色相 根据RGB中最大值的位置，分三种情况计算色相：
        # B通道最大
        hue[img[:, 2] == value] = 4.0 + ((img[:, 0] - img[:, 1]) / (value - img_min + eps))[img[:, 2] == value]
        # G通道最大
        hue[img[:, 1] == value] = 2.0 + ((img[:, 2] - img[:, 0]) / (value - img_min + eps))[img[:, 1] == value]
        # R通道最大
        hue[img[:, 0] == value] = (0.0 + ((img[:, 1] - img[:, 2]) / (value - img_min + eps))[img[:, 0] == value]) % 6

        hue[img.min(1)[0] == value] = 0.0
        # 归一化到 [0,1]：hue = hue/6.0
        hue = hue / 6.0

        # 计算饱和度 (Saturation)： 与HSV饱和度类似。
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()
        # 引入可学习参数 density_k，通过 color_sensitive = sin(v * 0.5π).pow(k) 调整颜色敏感度。
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        # 颜色敏感度 * 饱和度 * 余弦
        H = color_sensitive * saturation * ch
        # 颜色敏感度 * 饱和度 * 正弦
        V = color_sensitive * saturation * cv
        # 直接取自Value表示亮度分量
        I = value
        xyz = torch.cat([H, V, I], dim=1)


        return xyz

    def PHVIT(self, img):
        eps = 1e-8
        H, V, I = img[:, 0, :, :], img[:, 1, :, :], img[:, 2, :, :]

        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        # 解调颜色敏感度 恢复原始色度：
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        # 计算色相 (h)：
        h = torch.atan2(V + eps, H + eps) / (2 * pi)
        # 并取模1确保范围。
        h = h % 1
        # 计算饱和度(s)
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)  # # 饱和度限制在[0,1]
        v = torch.clamp(v, 0, 1)  # # 亮度限制在[0,1]

        r = torch.zeros_like(h)  # 初始化R通道全0
        g = torch.zeros_like(h)  # 初始化G通道全0
        b = torch.zeros_like(h)  # 初始化B通道全0
        # 将色相h划分为6个区间（类似彩虹的6个色段），每个区间对应不同的RGB组合规则。
        hi = torch.floor(h * 6.0)  # 将色相h ∈ [0,1) 映射到整数0~5
        # f：表示在色相区间内的相对位置（如 h=0.25 → hi=1, f=0.5）。
        f = h * 6.0 - hi  # 小数部分 f ∈ [0,1)，用于插值计算

        # 当饱和度为1时，p=0（纯色）；当饱和度为0时，p=v（灰度）。
        # q 和 t：根据f在色相区间内的位置，动态调整RGB分量。
        p = v * (1. - s)  # 最小RGB分量
        q = v * (1. - (f * s))  # 中间值1
        t = v * (1. - ((1. - f) * s))  # 中间值2
        # 根据 hi 的取值（0~5），将色相分为6个区间，每个区间对应不同的RGB计算规则：
        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5
        # 1) hi=0 (红色 → 黄色)
        r[hi0] = v[hi0]  # R = 最大亮度
        g[hi0] = t[hi0]  # G 随f增加从p→v
        b[hi0] = p[hi0]  # B = 最小值

        # (2) hi=1 (黄色 → 绿色)
        r[hi1] = q[hi1]  # R 随f增加从v→p
        g[hi1] = v[hi1]  # G = 最大亮度
        b[hi1] = p[hi1]  # B = 最小值

        #  hi=2 (绿色 → 青色)
        r[hi2] = p[hi2]  # R = 最小值
        g[hi2] = v[hi2]  # G = 最大亮度
        b[hi2] = t[hi2]  # B 随f增加从p→v

        # (4) hi=3 (青色 → 蓝色)
        r[hi3] = p[hi3]  # R = 最小值
        g[hi3] = q[hi3]  # G 随f增加从v→p
        b[hi3] = v[hi3]  # B = 最大亮度

        # hi=4 (蓝色 → 品红)
        r[hi4] = t[hi4]  # R 随f增加从p→v
        g[hi4] = p[hi4]  # G = 最小值
        b[hi4] = v[hi4]  # B = 最大亮度

        # hi=5 (品红 → 红色)
        r[hi5] = v[hi5]  # R = 最大亮度
        g[hi5] = p[hi5]  # G = 最小值
        b[hi5] = q[hi5]  # B 随f增加从v→p

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb
