import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.CSB import *

class DCENet(nn.Module):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(DCENet, self).__init__()


        [ch1, ch2, ch3, ch4] = channels # 四级特征通道数
        [head1, head2, head3, head4] = heads # 各层注意力头数
        # HV分支（色相-饱和度路径）#########################################
        # 编码器部分（下采样）
        # HV_ways
        self.HVE_block0 = nn.Sequential(  # 输入预处理
            nn.ReplicationPad2d(1), # 边缘复制填充
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            ) # 3通道转ch1
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm) # 标准化下采样
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        # 解码器部分（上采样）
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)  # 标准化上采样
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential( # 输出层
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        ) # 输出2通道

        # I分支（亮度路径）###############################################
        # 编码器
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )  # 单通道输入
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        # 解码器
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )# 单通道输出

        # 跨模态交互模块 ################################################
        # HV路径的交叉注意力（处理不同层级特征）
        self.HV_CSEB1 = HV_CSEB(ch2, head2) # 层级1交互
        self.HV_CSEB2 = HV_CSEB(ch3, head3) # 层级2
        self.HV_CSEB3 = HV_CSEB(ch4, head4)  # 层级3
        self.HV_CSEB4 = HV_CSEB(ch4, head4)  # 最深层的二次交互
        self.HV_CSEB5 = HV_CSEB(ch3, head3)  # 解码层级2
        self.HV_CSEB6 = HV_CSEB(ch2, head2)  # 解码层级1

        # I路径的交叉注意力
        self.I_CSEB1 = I_CSEB(ch2, head2)
        self.I_CSEB2 = I_CSEB(ch3, head3)
        self.I_CSEB3 = I_CSEB(ch4, head4)
        self.I_CSEB4 = I_CSEB(ch4, head4)
        self.I_CSEB5 = I_CSEB(ch3, head3)
        self.I_CSEB6 = I_CSEB(ch2, head2)
        # 颜色空间转换模块 ##############################################
        self.trans = RGB_HVI().cuda() # RGB与HVI空间互转（假设HVI=色相/饱和度/亮度空间）

    def forward(self, x):
        dtypes = x.dtype
        # 阶段1：RGB转HVI空间 -----------------------------------------
        hvi = self.trans.HVIT(x) # RGB->HVI转换 [B,3,H,W]
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes) # 提取亮度分量I [B,1,H,W]
        # 阶段2：双路径编码 -------------------------------------------
        # I路径编码
        i_enc0 = self.IE_block0(i) # 初始编码 [B,ch1,H,W]
        i_enc1 = self.IE_block1(i_enc0)  # 下采样1 [B,ch2,H/2,W/2]
        # HV路径编码
        hv_0 = self.HVE_block0(hvi) # 初始编码 [B,ch1,H,W]
        hv_1 = self.HVE_block1(hv_0)  # 下采样1 [B,ch2,H/2,W/2]
        # 跳跃连接保留
        i_jump0 = i_enc0 # 用于最终解码
        hv_jump0 = hv_0
        # 阶段3：多层级跨模态交互 -------------------------------------
        # 层级1交互
        i_enc2 = self.I_CSEB1(i_enc1, hv_1) # I路径融合HV信息
        hv_2 = self.HV_CSEB1(hv_1, i_enc1) # HV路径融合I信息
        v_jump1 = i_enc2 # 跳跃连接存储
        hv_jump1 = hv_2
        # 层级2处理
        i_enc2 = self.IE_block2(i_enc2)  # I下采样2 [B,ch3,H/4,W/4]
        hv_2 = self.HVE_block2(hv_2)  # HV下采样2

        i_enc3 = self.I_CSEB2(i_enc2, hv_2)  # 层级2交互
        hv_3 = self.HV_CSEB2(hv_2, i_enc2)
        v_jump2 = i_enc3  # 跳跃连接
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2) # I下采样3 [B,ch4,H/8,W/8]
        hv_3 = self.HVE_block3(hv_2)  # HV下采样3

        # 层级3交互（最深特征）
        i_enc4 = self.I_CSEB3(i_enc3, hv_3)
        hv_4 = self.HV_CSEB3(hv_3, i_enc3)

        # 阶段4：解码与重构 ------------------------------------------
        # 深层交互
        i_dec4 = self.I_CSEB4(i_enc4,hv_4)
        hv_4 = self.HV_CSEB4(hv_4, i_enc4)

        # 上采样过程（类U-Net结构）
        hv_3 = self.HVD_block3(hv_4, hv_jump2) # 跳跃连接融合
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_CSEB5(i_dec3, hv_3) # 解码层交互
        hv_2 = self.HV_CSEB5(hv_3, i_dec3)

        # 中层级重构
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        # 浅层重构
        i_dec1 = self.I_CSEB6(i_dec2, hv_2)
        hv_1 = self.HV_CSEB6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0) # 融合最浅层特征
        i_dec0 = self.ID_block0(i_dec1)  # I路径最终输出 [B,1,H,W]
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1) # HV路径最终输出 [B,2,H,W]

        # 阶段5：结果融合与逆变换 ------------------------------------
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi # 残差连接
        output_rgb = self.trans.PHVIT(output_hvi)  # HVI转回RGB

        return output_rgb # 最终输出增强后的RGB图像

    def HVIT(self,x):  # 辅助方法：直接获取HVI表示
        hvi = self.trans.HVIT(x)
        return hvi




