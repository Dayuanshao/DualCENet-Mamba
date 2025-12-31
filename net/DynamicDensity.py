import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDensity(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=8, out_channels=1):
        super(DynamicDensity, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        # 此处不进行 conv_down 操作，直接使用 x 的下采样特征
        self.conv_out = nn.Conv2d(hidden_channels + in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.min_k = 0.1
        self.max_k = 1.0

    def forward(self, x):
        feat1 = self.relu(self.conv1(x))
        feat1 = self.relu(self.conv2(feat1))
        feat_down = F.avg_pool2d(x, kernel_size=2)
        feat_down = F.interpolate(feat_down, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 此时 feat_down 的通道数仍为 in_channels (1)
        fused = torch.cat([feat1, x.new_tensor(feat_down)], dim=1)  # 拼接后的通道数为 8+1=9
        density_map = self.conv_out(fused)
        density_map = self.sigmoid(density_map)
        density_map = density_map * (self.max_k - self.min_k) + self.min_k
        return density_map
