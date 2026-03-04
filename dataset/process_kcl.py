

# ====================== 请根据实际情况修改以下导入 ======================
# 替换为包含 Volume 类、load_volume 等函数的模块路径
from nesvor.image import (
    Volume, load_volume
)
from nesvor.transform import RigidTransform
from nesvor.image.image_utils import (
    affine2transformation,
    compare_resolution_affine,
    transformation2affine,
    load_nii_volume,
    save_nii_volume,
)
import os
import pathlib
import torch
import nibabel as nib
import numpy as np
from typing import Tuple, Optional, Union



def resize_volume_tensor(
    tensor: torch.Tensor, 
    target_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """
    将3D张量（D, H, W）裁剪/填充到目标形状，优先居中裁剪，不足补0
    
    参数:
        tensor: 输入3D张量 (D, H, W)
        target_shape: 目标形状 (D, H, W)
    
    返回:
        调整后的3D张量，形状严格等于target_shape
    """
    assert len(tensor.shape) == 3, "输入必须是3D张量 (D, H, W)"
    curr_d, curr_h, curr_w = tensor.shape
    tgt_d, tgt_h, tgt_w = target_shape

    # -------------------------- 计算裁剪偏移（居中） --------------------------
    # 深度(D)方向
    crop_d_start = max(0, (curr_d - tgt_d) // 2)
    crop_d_end = min(curr_d, crop_d_start + tgt_d)
    # 高度(H)方向
    crop_h_start = max(0, (curr_h - tgt_h) // 2)
    crop_h_end = min(curr_h, crop_h_start + tgt_h)
    # 宽度(W)方向
    crop_w_start = max(0, (curr_w - tgt_w) // 2)
    crop_w_end = min(curr_w, crop_w_start + tgt_w)

    # -------------------------- 执行裁剪 --------------------------
    tensor_cropped = tensor[
        crop_d_start:crop_d_end,
        crop_h_start:crop_h_end,
        crop_w_start:crop_w_end
    ]

    # -------------------------- 计算填充偏移（居中） --------------------------
    pad_d_before = max(0, (tgt_d - tensor_cropped.shape[0]) // 2)
    pad_d_after = tgt_d - tensor_cropped.shape[0] - pad_d_before
    pad_h_before = max(0, (tgt_h - tensor_cropped.shape[1]) // 2)
    pad_h_after = tgt_h - tensor_cropped.shape[1] - pad_h_before
    pad_w_before = max(0, (tgt_w - tensor_cropped.shape[2]) // 2)
    pad_w_after = tgt_w - tensor_cropped.shape[2] - pad_w_before

    # -------------------------- 执行填充 --------------------------
    tensor_resized = torch.nn.functional.pad(
        tensor_cropped,
        pad=(pad_w_before, pad_w_after, pad_h_before, pad_h_after, pad_d_before, pad_d_after),
        mode="constant",
        value=0  # 填充值：图像/mask均补0
    )

    # 校验最终形状
    assert tensor_resized.shape == target_shape, \
        f"调整后形状{tensor_resized.shape}与目标{target_shape}不符"
    return tensor_resized

def save_processed_volume(
    volume: Volume,
    img_save_path: pathlib.Path,
    mask_save_path: pathlib.Path,
    target_shape: Tuple[int, int, int] = (135,155,189)
):
    """
    保存处理后的Volume（图像+掩码），掩码单独保存且前缀为tissue_
    
    参数:
        volume: 处理后的Volume对象
        img_save_path: 图像保存路径（.nii.gz）
        mask_save_path: 掩码保存路径（.nii.gz，前缀tissue_）
        target_shape: 目标形状 (D, H, W)
    """
    # 1. 调整图像和掩码到目标形状
    img=volume.image.permute(2,1,0)
    msk=volume.mask.float().permute(2,1,0)
    img_resized = resize_volume_tensor(img, target_shape)
    mask_resized = resize_volume_tensor(msk, target_shape).bool()

    # 2. 构建适配0.8分辨率的affine矩阵（RAS坐标系）
    res = 0.8  # 目标分辨率
    affine = np.eye(4, dtype=np.float32)
    # 设置分辨率
    affine[0, 0] = res   # X轴（宽度）
    affine[1, 1] = res   # Y轴（高度）
    affine[2, 2] = res   # Z轴（深度）
    # 调整原点到图像中心
    affine[0, 3] = - (target_shape[2] - 1) * res / 2  # W方向中心
    affine[1, 3] = - (target_shape[1] - 1) * res / 2  # H方向中心
    affine[2, 3] = - (target_shape[0] - 1) * res / 2  # D方向中心

    # 3. 保存图像
    img_np = img_resized.cpu().numpy()
    img_nii = nib.Nifti1Image(img_np, affine)
    nib.save(img_nii, str(img_save_path))

    # 4. 保存掩码（转uint8格式，0/1）
    mask_np = mask_resized.cpu().numpy().astype(np.uint8)
    mask_nii = nib.Nifti1Image(mask_np, affine)
    nib.save(mask_nii, str(mask_save_path))

    print(f"✅ 保存完成：")
    print(f"  - 图像: {img_save_path}")
    print(f"  - 掩码: {mask_save_path}")

def main():
    # ====================== 配置参数（请根据实际情况修改） ======================
    # 原始图像文件夹（存放待处理的nii.gz）
    IMG_ROOT = pathlib.Path("/home/lvyao/git/dataset/kcl/image")
    # 原始掩码文件夹（存放tissue_前缀的nii.gz）
    MASK_ROOT = pathlib.Path("/home/lvyao/git/dataset/kcl/label")
    # 处理后图像输出文件夹
    OUTPUT_IMG_ROOT = pathlib.Path("/home/lvyao/git/dataset/kcl/image_resample")
    # 处理后掩码输出文件夹（单独存放）
    OUTPUT_MASK_ROOT = pathlib.Path("/home/lvyao/git/dataset/kcl/label_resample")
    # 目标分辨率（mm）
    TARGET_RES = 0.8
    # 目标形状 (D, H, W)
    TARGET_SHAPE = (155, 189,135)
    # 计算设备（优先GPU）
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================== 初始化文件夹 ======================
    OUTPUT_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"📌 配置信息：")
    print(f"  - 图像源目录: {IMG_ROOT}")
    print(f"  - 掩码源目录: {MASK_ROOT}")
    print(f"  - 图像输出目录: {OUTPUT_IMG_ROOT}")
    print(f"  - 掩码输出目录: {OUTPUT_MASK_ROOT}")
    print(f"  - 目标分辨率: {TARGET_RES}mm")
    print(f"  - 目标形状: {TARGET_SHAPE}")
    print(f"  - 计算设备: {DEVICE}\n")

    # ====================== 遍历处理所有nii.gz文件 ======================
    img_files = list(IMG_ROOT.glob("*.nii.gz"))
    if not img_files:
        print(f"❌ 错误：在{IMG_ROOT}中未找到任何nii.gz文件")
        return

    total_files = len(img_files)
    print(f"📄 共发现 {total_files} 个待处理文件\n")

    for idx, img_file in enumerate(img_files, 1):
        # 1. 解析文件名（去除.nii.gz后缀）
        file_stem = img_file.name.replace(".nii.gz", "")
        print(f"[{idx}/{total_files}] 处理文件: {file_stem}")

        # 2. 构建原始掩码文件路径（tissue_前缀）
        mask_file = MASK_ROOT / f"{file_stem.replace('t2-','tissue-')}_dhcp-19.nii.gz"
        if not mask_file.exists():
            print(f"⚠️  警告：掩码文件 {mask_file} 不存在，跳过该文件\n")
            continue

        # 3. 加载Volume（含mask）
        try:
            volume = load_volume(
                path_vol=img_file,
                path_mask=mask_file,
                device=DEVICE
            )
        except Exception as e:
            print(f"❌ 错误：加载文件失败 - {str(e)}\n")
            continue

        # 4. 重采样到目标分辨率
        try:
            volume_resampled = volume.resample(
                resolution_new=TARGET_RES,
                transformation_new=None  # 使用原始空间变换
            )
        except Exception as e:
            print(f"❌ 错误：重采样失败 - {str(e)}\n")
            continue

        # 5. 构建保存路径
        # 处理后图像名：原文件名_resampled.nii.gz
        img_save_name = f"{file_stem}_resampled.nii.gz"
        img_save_path = OUTPUT_IMG_ROOT / img_save_name
        # 处理后掩码名：tissue_原文件名_resampled.nii.gz（保持tissue_前缀）
        mask_save_name = f"tissue-{file_stem}_resampled.nii.gz"
        mask_save_path = OUTPUT_MASK_ROOT / mask_save_name

        # 6. 保存处理后的图像和掩码
        try:
            save_processed_volume(volume_resampled, img_save_path, mask_save_path, TARGET_SHAPE)
        except Exception as e:
            print(f"❌ 错误：保存文件失败 - {str(e)}\n")
            continue

        print("-" * 60 + "\n")

    print(f"\n🎉 所有文件处理完成！")
    print(f"📁 处理后图像路径：{OUTPUT_IMG_ROOT}")
    print(f"📁 处理后掩码路径：{OUTPUT_MASK_ROOT}")

if __name__ == "__main__":
    # 安装依赖提示（如需）
    # pip install nibabel torch numpy opencv-python antspyx
    main()