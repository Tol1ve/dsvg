import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def downsample_3d(img, mask=None, scale=2):
    """
    三维图像降采样函数：将3D图像和掩码沿depth/height/width维度降采样指定倍数。
    
    参数:
    - img: 输入3D图像张量，支持两种形状：
        - 5维：[batch_size, channels, depth, height, width]（带batch）
        - 4维：[channels, depth, height, width]（无batch）
    - mask: 可选的3D掩码张量，形状与img一致（通道数通常为1）
    - scale: 降采样倍数，可选值为2或4，默认值为2
    
    返回:
    - img_down: 降采样后的3D图像张量（维度与输入一致）
    - mask_down: 降采样后的3D掩码张量（如果提供了掩码，维度与输入一致）
    
    异常:
    - ValueError: 当scale不是2或4时抛出
    """
    if scale==1:
        return img,mask
    # 校验采样倍数合法性
    if scale not in {2, 4,6,8,16}:
        raise ValueError(f"降采样倍数scale必须为2或4，当前输入：{scale}")
    
    # 记录原始维度，处理4维（无batch）→ 5维（加batch）
    is_4d = img.dim() == 4
    if is_4d:
        img = img.unsqueeze(0)  # [C,D,H,W] → [1,C,D,H,W]
        mask = mask.unsqueeze(0) if mask is not None else None
    
    # 计算降采样缩放因子：2倍→0.5，4倍→0.25
    scale_factor = 1.0 / scale
    # 3D降采样：depth/height/width按指定倍数缩放
    img_down = F.interpolate(
        img, 
        scale_factor=scale_factor, 
        mode='trilinear',  # 3D线性插值
        align_corners=False
    )
    
    if mask is not None:
        mask_down = F.interpolate(
            mask.float(), 
            scale_factor=scale_factor, 
            mode='nearest'  # 掩码用最近邻插值，避免非0/1值
        )
        mask_down = torch.round(mask_down)  # 四舍五入消除微小偏差
        mask_down = torch.clamp(mask_down, -1.0, 1.0)  # 钳位到[-1,1]范围
    else:
        mask_down = None
    
    # 恢复原始维度（去掉batch维度）
    if is_4d:
        img_down = img_down.squeeze(0)
        mask_down = mask_down.squeeze(0) if mask_down is not None else None

    return img_down, mask_down


def upsample_3d(img, mask=None, scale=2):
    """
    三维图像上采样函数：将3D图像和掩码沿depth/height/width维度上采样指定倍数。
    
    参数:
    - img: 输入3D图像张量，支持两种形状：
        - 5维：[batch_size, channels, depth, height, width]（带batch）
        - 4维：[channels, depth, height, width]（无batch）
    - mask: 可选的3D掩码张量，形状与img一致（通道数通常为1）
    - scale: 上采样倍数，可选值为2或4，默认值为2
    
    返回:
    - img_up: 上采样后的3D图像张量（维度与输入一致）
    - mask_up: 上采样后的3D掩码张量（如果提供了掩码，维度与输入一致）
    
    异常:
    - ValueError: 当scale不是2或4时抛出
    """
    # 校验采样倍数合法性
    if scale==1:
        return img,mask
    if scale not in {2, 4,6,8,16}:
        raise ValueError(f"上采样倍数scale必须为2或4，当前输入：{scale}")
    
    # 记录原始维度，处理4维（无batch）→ 5维（加batch）
    is_4d = img.dim() == 4
    if is_4d:
        img = img.unsqueeze(0)  # [C,D,H,W] → [1,C,D,H,W]
        mask = mask.unsqueeze(0) if mask is not None else None
    
    # 计算上采样缩放因子：2倍→2.0，4倍→4.0
    scale_factor = float(scale)
    # 3D上采样：depth/height/width按指定倍数缩放
    img_up = F.interpolate(
        img, 
        scale_factor=scale_factor, 
        mode='trilinear',  # 3D线性插值
        align_corners=False
    )
    
    if mask is not None:
        mask_up = F.interpolate(
            mask, 
            scale_factor=scale_factor, 
            mode='nearest'  # 掩码用最近邻插值
        )
        mask_up = torch.round(mask_up)  # 四舍五入消除微小偏差
        mask_up = torch.clamp(mask_up, -1.0, 1.0)  # 钳位到[-1,1]范围
    else:
        mask_up = None
    
    # 恢复原始维度（去掉batch维度）
    if is_4d:
        img_up = img_up.squeeze(0)
        mask_up = mask_up.squeeze(0) if mask_up is not None else None

    return img_up, mask_up


# ------------------- 示例测试 -------------------
if __name__ == "__main__":
    # 测试1：带batch的5维3D张量 [batch, channels, depth, height, width]
    print("=== 测试5维张量（带batch）===")
    img_5d = torch.randn(1, 3, 64, 128, 128)  # batch=1, c=3, d=64, h=128, w=128
    # 构造float类型的mask，取值仅为-1/1
    mask_5d = torch.randint(0, 2, (1, 1, 64, 128, 128)).float() * 2 - 1  # 0→-1，1→1

    # 降采样
    img_5d_down, mask_5d_down = downsample_3d(img_5d, mask_5d)
    print("降采样后图像形状:", img_5d_down.shape)  # 预期: [1,3,32,64,64]
    print("降采样后掩码形状:", mask_5d_down.shape)  # 预期: [1,1,32,64,64]
    print("降采样后掩码取值范围:", f"{mask_5d_down.min().item():.2f} ~ {mask_5d_down.max().item():.2f}")  # 预期: -1.0 ~ 1.0

    # 上采样
    img_5d_up, mask_5d_up = upsample_3d(img_5d_down, mask_5d_down)
    print("上采样后图像形状:", img_5d_up.shape)    # 预期: [1,3,64,128,128]
    print("上采样后掩码形状:", mask_5d_up.shape)    # 预期: [1,1,64,128,128]
    print("上采样后掩码取值范围:", f"{mask_5d_up.min().item():.2f} ~ {mask_5d_up.max().item():.2f}")  # 预期: -1.0 ~ 1.0

    # 测试2：无batch的4维3D张量 [channels, depth, height, width]
    print("\n=== 测试4维张量（无batch）===")
    img_4d = torch.randn(3, 64, 128, 128)  # c=3, d=64, h=128, w=128
    # 构造float类型的mask，取值仅为-1/1
    mask_4d = torch.randint(0, 2, (1, 64, 128, 128)).float() * 2 - 1  # 0→-1，1→1

    # 降采样
    img_4d_down, mask_4d_down = downsample_3d(img_4d, mask_4d)
    print("降采样后图像形状:", img_4d_down.shape)  # 预期: [3,32,64,64]
    print("降采样后掩码形状:", mask_4d_down.shape)  # 预期: [1,32,64,64]
    print("降采样后掩码取值范围:", f"{mask_4d_down.min().item():.2f} ~ {mask_4d_down.max().item():.2f}")  # 预期: -1.0 ~ 1.0

    # 上采样
    img_4d_up, mask_4d_up = upsample_3d(img_4d_down, mask_4d_down)
    print("上采样后图像形状:", img_4d_up.shape)    # 预期: [3,64,128,128]
    print("上采样后掩码形状:", mask_4d_up.shape)    # 预期: [1,64,128,128]
    print("上采样后掩码取值范围:", f"{mask_4d_up.min().item():.2f} ~ {mask_4d_up.max().item():.2f}")  # 预期: -1.0 ~ 1.0