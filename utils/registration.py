import logging
import time
import math
from typing import List, Tuple, Optional, cast
import numpy as np
import torch
import torch.nn.functional as F
from nesvor.svort import SVoRT, SVoRTv2,SVoRTv3,SVoRTv4_pos
from nesvor.transform import RigidTransform
from nesvor.utils import get_PSF, ncc_loss, resample
from nesvor.image import Stack, Slice, Volume
from nesvor.svr.reconstruction import simulate_slices,simulated_error,slices_scale
from nesvor.svr.outlier import EM, global_ncc_exclusion, local_ssim_exclusion
from nesvor.svr.reconstruction import (
    psf_reconstruction,
    srr_update,
    simulate_slices,
    slices_scale,
    simulated_error,
    SRR_CG,
)
from nesvor.svr.pipeline import slice_to_volume_reconstruction,SliceToVolumeRegistration
from nesvor.utils import DeviceType, PathType, get_PSF
from nesvor.image import Volume, Slice, load_volume, load_mask, Stack
from nesvor.inr.data import PointDataset
def DEBUG_test_save(img,name,save_dir='/home/lvyao/git/med-ddpm/results/DM_results/test_dpcp_svr_cor'):
    import os
    import numpy as np
    import nibabel as nib
    os.makedirs(save_dir,exist_ok=True)
    out=img.cpu().detach().numpy()
    nifti_img = nib.Nifti1Image(out.squeeze(), affine=np.eye(4))
    nib.save(nifti_img,os.path.join(save_dir,name))
def parse_data(
    dataset: List[Stack], svort: bool
) -> Tuple[
    List[Stack],
    List[Stack],
    List[Stack],
    List[Stack],
    List[torch.Tensor],
    Volume,
    torch.Tensor,
]:
    stacks = []  # resampled, cropped, normalized
    stacks_ori = []  # resampled
    transforms = []  # cropped, reset (SVoRT input)
    transforms_full = []  # reset, but with original size
    transforms_ori = []  # original
    crop_idx = []  # z
    dataset_out = []

    res_s = 1.0
    res_r = 0.8

    for data in dataset:
        logging.debug("Preprocessing stack %s for registration.", data.name)
        # resample
        slices = resample(
            data.slices * data.mask,
            (data.resolution_x, data.resolution_y),
            (res_s, res_s),
        )
        slices_ori = slices.clone()
        # crop x,y
        s = slices[torch.argmax((slices > 0).sum((1, 2, 3))), 0]
        i1, i2, j1, j2 = 0, s.shape[0] - 1, 0, s.shape[1] - 1
        while i1 < s.shape[0] and s[i1, :].sum() == 0:
            i1 += 1
        while i2 and s[i2, :].sum() == 0:
            i2 -= 1
        while j1 < s.shape[1] and s[:, j1].sum() == 0:
            j1 += 1
        while j2 and s[:, j2].sum() == 0:
            j2 -= 1
        if ((i2 - i1) > 128 or (j2 - j1) > 128) and svort:
            logging.warning('ROI in input stack "%s" is too large for SVoRT', data.name)
        if (i2 - i1) <= 0:
            logging.warning(
                'Input stack "%s" is all zero after maksing and will be skipped. Please check your data!',
                data.name,
            )
            continue
        pad_margin = 64
        slices = F.pad(
            slices, (pad_margin, pad_margin, pad_margin, pad_margin), "constant", 0
        )
        i = pad_margin + (i1 + i2) // 2
        j = pad_margin + (j1 + j2) // 2
        slices = slices[:, :, i - 64 : i + 64, j - 64 : j + 64]
        # crop z
        idx = (slices > 0).float().sum((1, 2, 3)) > 0
        nz = torch.nonzero(idx)
        nnz = torch.numel(nz)
        if nnz < 7:
            logging.warning(
                'Input stack "%s" only has %d nonzero slices after masking. Consider remove this stack.',
                data.name,
                nnz,
            )
        else:
            logging.debug(
                'Input stack "%s" has %d nonzero slices after masking.', data.name, nnz
            )
        idx[int(nz[0, 0]) : int(nz[-1, 0] + 1)] = True
        crop_idx.append(idx)
        slices = slices[idx]
        # normalize
        stacks.append(slices / torch.quantile(slices[slices > 0], 0.99))
        stacks_ori.append(slices_ori)
        # transformation
        transform = data.transformation
        transforms_ori.append(transform)
        transform_full_ax = transform.axisangle().clone()
        transform_ax = transform_full_ax[idx].clone()

        transform_full_ax[:, :-1] = 0
        transform_full_ax[:, 3] = -((j1 + j2) // 2 - slices_ori.shape[-1] / 2) * res_s
        transform_full_ax[:, 4] = -((i1 + i2) // 2 - slices_ori.shape[-2] / 2) * res_s
        transform_full_ax[:, -1] -= transform_ax[:, -1].mean()

        transform_ax[:, :-1] = 0
        transform_ax[:, -1] -= transform_ax[:, -1].mean()

        transforms.append(RigidTransform(transform_ax))
        transforms_full.append(RigidTransform(transform_full_ax))

        dataset_out.append(data)

    assert len(dataset_out) > 0, "Input data is empty!"

    s_thick = np.mean([data.thickness for data in dataset_out])
    gaps = [data.gap for data in dataset_out]

    stacks_svort_in = [
        Stack(
            stacks[j],
            stacks[j] > 0,
            transforms[j],
            res_s,
            res_s,
            s_thick,
            gaps[j],
        )
        for j in range(len(dataset_out))
    ]

    stacks_resampled = [
        Stack(
            stacks_ori[j],
            stacks_ori[j] > 0,
            transforms_ori[j],
            res_s,
            res_s,
            s_thick,
            gaps[j],
        )
        for j in range(len(dataset_out))
    ]

    stacks_resampled_reset = [s.clone(zero=False, deep=False) for s in stacks_resampled]
    for j in range(len(dataset_out)):
        stacks_resampled_reset[j].transformation = transforms_full[j]

    volume = Volume.zeros((200, 200, 200), res_r, device=dataset_out[0].device)

    psf = get_PSF(
        res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
        device=volume.device,
    )

    return (
        dataset_out,
        stacks_svort_in,
        stacks_resampled,
        stacks_resampled_reset,
        crop_idx,
        volume,
        psf,
    )
def _check_resolution_and_shape(slices: List[Slice]) -> List[Slice]:
    res_inplane = []
    thicknesses = []
    for s in slices:
        res_inplane.append(float(s.resolution_x))
        res_inplane.append(float(s.resolution_y))
        thicknesses.append(float(s.resolution_z))

    res_s = min(res_inplane)
    s_thick = np.mean(thicknesses).item()
    slices = [s.resample((res_s, res_s, s_thick)) for s in slices]
    slices = Stack.pad_stacks(slices)

    if max(thicknesses) - min(thicknesses) > 0.001:
        logging.warning("The input data have different thicknesses!")

    return slices

def _initial_mask(
    slices: List[Slice],
    output_resolution: float,
    sample_mask: Optional[PathType]=None,
    sample_orientation: Optional[PathType]=None,
    device: DeviceType='cpu',
    mask_atlas=None,
) -> Tuple[Volume, bool]:
    dataset = PointDataset(slices)
    #dataset.dilate=True
    if mask_atlas is not None:
        dataset.use_mask_atlas=mask_atlas
    mask = dataset.mask
    if sample_mask is not None:
        mask = load_mask(sample_mask, device)
    transformation = None
    if sample_orientation is not None:
        transformation = load_volume(
            sample_orientation,
            device=device,
        ).transformation
    mask = mask.resample(output_resolution, transformation)
    mask.mask = mask.image > 0
    return mask, sample_mask is None

def _normalize(
    stack: Stack, output_intensity_mean: float
) -> Tuple[Stack, float, float]:
    masked_v = stack.slices[stack.mask]
    mean_intensity = masked_v.mean().item()
    max_intensity = masked_v.max().item()
    min_intensity = masked_v.min().item()
    stack.slices = stack.slices * (output_intensity_mean / mean_intensity)
    max_intensity = max_intensity * (output_intensity_mean / mean_intensity)
    min_intensity = min_intensity * (output_intensity_mean / mean_intensity)
    return stack, max_intensity, min_intensity

def svr_reconstruction(
    slices: List[Slice],
    *,
    with_background: bool = False,
    output_resolution: float = 0.8,
    output_intensity_mean: float = 700,
    delta: float = 150 / 700,
    n_iter: int = 3,
    n_iter_rec: List[int] = [7, 7, 21],
    global_ncc_threshold: float = 0.5,
    local_ssim_threshold: float = 0.4,
    no_slice_robust_statistics: bool = False,
    no_pixel_robust_statistics: bool = False,
    no_global_exclusion: bool = False,
    no_local_exclusion: bool = False,
    sample_mask: Optional[PathType] = None,
    sample_orientation: Optional[PathType] = None,
    psf: str = "gaussian",
    device: DeviceType = torch.device("cpu"),
    **unused
):
    slices = _check_resolution_and_shape(slices)
    stack = Stack.cat(slices)
    slices_mask_backup = stack.mask.clone()

    # init volume
    volume, is_refine_mask = _initial_mask(
        slices,
        output_resolution,
        sample_mask,
        sample_orientation,
        device,
    )

    # data normalization
    stack, max_intensity, min_intensity = _normalize(stack, output_intensity_mean)

    # define psf
    psf_tensor = get_PSF(
        res_ratio=(
            stack.resolution_x / output_resolution,
            stack.resolution_y / output_resolution,
            stack.thickness / output_resolution,
        ),
        device=volume.device,
        psf_type=psf,
    )

    # outer loop
    for i in range(n_iter):
        logging.info("outer %d", i)
        # slice-to-volume registration
        if i > 0:  # skip slice-to-volume registration for the first iteration
            svr = SliceToVolumeRegistration(
                num_levels=3,
                num_steps=5,
                step_size=2,
                max_iter=30,
            )
            slices_transform, _ = svr(
                stack,
                volume,
                use_mask=True,
            )
            stack.transformation = slices_transform

        # global structual exclusion
        if i > 0 and not no_global_exclusion:
            stack.mask = slices_mask_backup.clone()
            excluded = global_ncc_exclusion(stack, volume, global_ncc_threshold)
            stack.mask[excluded] = False
        # PSF reconstruction & volume mask
        volume = psf_reconstruction(
            stack,
            volume,
            update_mask=is_refine_mask,
            use_mask=not with_background,
            psf=psf_tensor,
        )

        # init EM
        em = EM(max_intensity, min_intensity)
        p_voxel = torch.ones_like(stack.slices)
        # super-resolution reconstruction (inner loop)
        for j in range(n_iter_rec[i]):
            logging.info("inner %d", j)
            # simulate slices
            slices_sim, slices_weight = cast(
                Tuple[Stack, Stack],
                simulate_slices(
                    stack,
                    volume,
                    return_weight=True,
                    use_mask=not with_background,
                    psf=psf_tensor,
                ),
            )
            # scale
            scale = slices_scale(stack, slices_sim, slices_weight, p_voxel, True)
            # err
            err = simulated_error(stack, slices_sim, scale)
            # EM robust statistics
            if (not no_pixel_robust_statistics) or (not no_slice_robust_statistics):
                p_voxel, p_slice = em(err, slices_weight, scale, 1)
                if no_pixel_robust_statistics:  # reset p_voxel
                    p_voxel = torch.ones_like(stack.slices)
            p = p_voxel
            if not no_slice_robust_statistics:
                p = p_voxel * p_slice.view(-1, 1, 1, 1)
            # local structural exclusion
            if not no_local_exclusion:
                p = p * local_ssim_exclusion(stack, slices_sim, local_ssim_threshold)
            # super-resolution update
            beta = max(0.01, 0.08 / (2**i))
            alpha = min(1, 0.05 / beta)
            volume = srr_update(
                err,
                volume,
                p,
                alpha,
                beta,
                delta * output_intensity_mean,
                use_mask=not with_background,
                psf=psf_tensor,
            )

    # reconstruction finished
    # prepare outputs
    slices_sim = cast(
        Stack,
        simulate_slices(
            stack, volume, return_weight=False, use_mask=True, psf=psf_tensor
        ),
    )
    simulated_slices = stack[:]
    output_slices = slices_sim[:]
    return volume, output_slices, simulated_slices
def cg_reconstruction(slices,volume):
    srr=SRR_CG( 
        n_iter=3,
        average_init = False,
        output_relu = True,
        use_mask = True)
    stack = Stack.cat(slices)
    v=srr(stack,volume)
    return v

def forward_reconstruction(
    slices: List[Slice],
    init_volume: Volume,
    *,
    with_background: bool = False,
    output_resolution: float = 0.8,
    output_intensity_mean: float = 700,
    delta: float = 150 / 700,
    local_ssim_threshold: float = 0.4,
    no_slice_robust_statistics: bool = False,
    no_pixel_robust_statistics: bool = False,
    no_local_exclusion: bool = False,
    sample_mask: Optional[PathType] = None,
    sample_orientation: Optional[PathType] = None,
    psf: str = "gaussian",
    no_regular: bool =None,
    device: DeviceType = torch.device("cpu"),
    k_schedule:Optional[int]=None,
    **unused
) -> Tuple[Volume, List[Slice], List[Slice]]:
    # check data
    slices = _check_resolution_and_shape(slices)
    stack = Stack.cat(slices)
    # init volume
    def test_stack_save(stack,name,save_dir="/home/lvyao/git/med-ddpm/results/DM_results/figures"):
        import os
        v=stack.get_volume()
        v.save(os.path.join(save_dir,name))
    if init_volume is None:
        volume=_initial_mask(
        slices,
        output_resolution,
        sample_mask,
        sample_orientation,
        device,
    )
    else:
        init_volume.rescale(output_intensity_mean,masked=True)
        volume=init_volume
    # data normalization
    stack, max_intensity, min_intensity = _normalize(stack, output_intensity_mean)
    if psf:
    # define psf
        psf_tensor = get_PSF(
            res_ratio=(
                stack.resolution_x / output_resolution,
                stack.resolution_y / output_resolution,
                stack.thickness / output_resolution,
            ),
            device=volume.device,
            psf_type=psf,
        )
    else:
        psf_tensor =None
    # outer loop
    # for i in range(n_iter):
    #     logging.info("outer %d", i)
    #     # slice-to-volume registration
    #     if i > 0:  # skip slice-to-volume registration for the first iteration
    #         svr = SliceToVolumeRegistration(
    #             num_levels=3,
    #             num_steps=5,
    #             step_size=2,
    #             max_iter=30,
    #         )
    #         slices_transform, _ = svr(
    #             stack,
    #             volume,
    #             use_mask=True,
    #         )
    #         stack.transformation = slices_transform

    #     # global structual exclusion
    #     if i > 0 and not no_global_exclusion:
    #         stack.mask = slices_mask_backup.clone()
    #         excluded = global_ncc_exclusion(stack, volume, global_ncc_threshold)
    #         stack.mask[excluded] = False
    #     # PSF reconstruction & volume mask
    #     volume = psf_reconstruction(
    #         stack,
    #         volume,
    #         update_mask=is_refine_mask,
    #         use_mask=not with_background,
    #         psf=psf_tensor,
    #     )

    # init EM
    em = EM(max_intensity, min_intensity)
    p_voxel = torch.ones_like(stack.slices)
    # super-resolution reconstruction (inner loop)
    #n_iter=30
    #n_iter=120-k_schedule*25 if k_schedule is not None else 120
    n_iter=85-k_schedule*40 if k_schedule is not None else 85
    for j in range(n_iter):
        logging.info("inner %d", j)
        # simulate slices
        slices_sim, slices_weight = cast(
            Tuple[Stack, Stack],
            simulate_slices(
                stack,
                volume,
                return_weight=True,
                use_mask=not with_background,
                psf=psf_tensor,
            ),
        )
        # scale
        scale = slices_scale(stack, slices_sim, slices_weight, p_voxel, True)
        # err
        err = simulated_error(stack, slices_sim, scale)
        # EM robust statistics
        if (not no_pixel_robust_statistics) or (not no_slice_robust_statistics):
            p_voxel, p_slice = em(err, slices_weight, scale, 1)
            if no_pixel_robust_statistics:  # reset p_voxel
                p_voxel = torch.ones_like(stack.slices)
        p = p_voxel
        if not no_slice_robust_statistics:
            p = p_voxel * p_slice.view(-1, 1, 1, 1)
        # local structural exclusion
        if not no_local_exclusion:
            p = p * local_ssim_exclusion(stack, slices_sim, local_ssim_threshold)
        # super-resolution update
        beta = 0.04
        alpha = 0.04
        volume = srr_update(
            err,
            volume,
            p,
            alpha,
            beta,
            delta * output_intensity_mean,
            use_mask=not with_background,
            psf=psf_tensor,
            no_regular=no_regular
        )

    # reconstruction finished
    # prepare outputs
    slices_sim = cast(
        Stack,
        simulate_slices(
            stack, volume, return_weight=False, use_mask=True, psf=psf_tensor
        ),
    )
    simulated_slices = stack[:]
    output_slices = slices_sim[:]
    return volume, output_slices, simulated_slices

def _register(args,dataset,update_volume=None):

    svort=args.use_svort
    res_r = args.resolution_volume
    device_index = args.device
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")



    if args.svort_version==2:
        model=SVoRTv2(n_iter=4)
    else:
        model=SVoRTv3(n_iter=4)
        
    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device  # 自动将权重映射到指定设备（CPU/GPU）
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    (dataset, stacks_in, ss_ori, stacks_full, crop_idx, volume, psf) = parse_data(
        dataset, svort
    )
    origin_device=stacks_in[0].device

    positions = torch.cat(
        [
            torch.stack(
                (
                    torch.arange(len(s), dtype=torch.float32, device=device)
                    - len(s) // 2,
                    torch.full((len(s),), i, dtype=torch.float32, device=device),
                ),
                dim=-1,
            )
            for i, s in enumerate(stacks_in)
        ],
        dim=0,
    )

    res_s = stacks_in[0].resolution_x
    data = {
        "psf_rec": psf.to(device=device),
        "slice_shape": stacks_in[0].shape[-2:],  # (128, 128)
        "resolution_slice": res_s,
        "resolution_recon": res_r,
        "volume_shape": volume.shape,  # 
        "transforms": RigidTransform.cat(
            [s.transformation for s in stacks_in]
        ).matrix().to(device),
        "stacks": torch.cat([s.slices for s in stacks_in], dim=0).to(device),
        "positions": positions,
        "slice_thickness":stacks_in[0].thickness,
    }
    with torch.no_grad():
        if update_volume is not None:

            v_out= model.update_volume(data,update_volume.to(device))
            return v_out.to(origin_device)
        else:
            t_out, v_out, _ = model(data)
    transforms_out = [t_out[-1][positions[:, -1] == i] for i in range(len(stacks_in))]
    stacks_out = []
    for i in range(len(stacks_in)):
        stack_out = stacks_in[i].clone(zero=False, deep=False)
        stack_out.transformation = transforms_out[i].to(origin_device)
        stack_out.slices=stack_out.slices.to(origin_device)
        stack_out.mask=stack_out.mask.to(origin_device)
        stacks_out.append(stack_out)
    slices = []
    for stack in stacks_out:
        idx_nonempty = stack.mask.flatten(1).any(1)
        stack.slices /= torch.quantile(stack.slices[stack.mask], 0.99)  # normalize
        slices.extend(stack[idx_nonempty])
    if args.use_svr_volume:
        volume,_,_=slice_to_volume_reconstruction(slices)
        pad_odd_to_even = lambda shape: tuple(
            [(0, 1 if dim % 2 != 0 else 0) for dim in reversed(shape)]
        )
        pad_params = pad_odd_to_even(volume.image)
        volume_padded = torch.nn.functional.pad(volume_padded, pad_params)
        v_out=[volume_padded.image.view((1,1)+volume.image.shape)]
    return slices,v_out[-1].to(origin_device)   
def forward_process(volumes,slices,args):
    volumes.rescale(1)
    stacks=Stack.cat(slices)
    output_resolution=args.resolution_volume
    psf_tensor = get_PSF(
        res_ratio=(
            stacks.resolution_x / output_resolution,
            stacks.resolution_y / output_resolution,
            stacks.thickness / output_resolution,
        ),
        device=args.device,
        psf_type=args.psf,
    )

    slices_sim, slices_weight = cast(
        Tuple[Stack, Stack],
        simulate_slices(
            stacks,
            volumes,
            return_weight=True,
            use_mask=True,
            psf=psf_tensor,
        ),
    )
    err = stacks.slices  - slices_sim.slices
    return Stack.like(stacks, slices=err, deep=False)
    p_voxel = torch.ones_like(stacks.slices)
    scale = slices_scale(stacks, slices_sim, slices_weight, p_voxel, True)
    err = simulated_error(stacks, slices_sim, scale)

    return err



def detect_view_code(transform_matrix):
    """
    根据变换矩阵自动判断切片方向。
    输入: transform_matrix [3, 4] or [4, 4]
    输出: float code (0=Axial, 1=Coronal, 2=Sagittal, 3=Unknown)
    """
    # 1. 获取旋转矩阵的第三列 (切片法向量在世界坐标系的投影)
    #    假设切片坐标系中 Z轴 (0,0,1) 是法线
    slice_normal = transform_matrix[:3, 2].abs() # 取绝对值
    
    # 2. 找到最大分量所在的轴
    axis = torch.argmax(slice_normal).item()
    
    # 3. 映射为 View Code
    # World Z-axis (index 2) dominant -> Axial -> 0
    # World Y-axis (index 1) dominant -> Coronal -> 1
    # World X-axis (index 0) dominant -> Sagittal -> 2
    if axis == 2:
        return 0.0
    elif axis == 1:
        return 1.0
    elif axis == 0:
        return 2.0
    else:
        return 3.0
    

def _register_pe(args, dataset):
    print("Using position encoding for registration")
    svort = args.use_svort
    res_r = args.resolution_volume
    device_index = args.device
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    # 1. 自动判断模型版本
    use_view_code = False
    if args.svort_version == 2:
        model = SVoRTv2(n_iter=4)
        use_view_code = False
    else:
        model = SVoRTv4_pos(n_iter=4)
        use_view_code = True

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # 2. 解析数据
    # dataset: 原始数据
    # stacks_in: 用于输入的 Stack (旋转已归零)
    # ss_ori: 原始 Stack (保留旋转信息，用于检测 View Code)
    (dataset, stacks_in, ss_ori, stacks_full, crop_idx, volume, psf) = parse_data(
        dataset, svort
    )
    origin_device = stacks_in[0].device

    # 3. 构建 Positions
    positions_list = []
    
    for i, s in enumerate(stacks_in):
        n_slices = len(s)
        
        # A. 基础信息
        z_coords = torch.arange(n_slices, dtype=torch.float32, device=device) - n_slices // 2
        stack_ids = torch.full((n_slices,), i, dtype=torch.float32, device=device)
        items_to_stack = [z_coords, stack_ids]
        
        # B. 视图信息 (使用 ss_ori 检测)
        if use_view_code:
            original_stack = ss_ori[i]
            mat = original_stack.transformation.matrix()
            if mat.dim() == 3:
                mat_sample = mat[0] 
            else:
                mat_sample = mat
            
            code = detect_view_code(mat_sample)
            view_ids = torch.full((n_slices,), code, dtype=torch.float32, device=device)
            items_to_stack.append(view_ids)
        
        positions_list.append(torch.stack(items_to_stack, dim=-1))

    positions = torch.cat(positions_list, dim=0)

    # 4. 构造输入
    res_s = stacks_in[0].resolution_x
    data = {
        "psf_rec": psf.to(device=device),
        "slice_shape": stacks_in[0].shape[-2:],
        "resolution_slice": res_s,
        "resolution_recon": res_r,
        "volume_shape": volume.shape,
        "transforms": RigidTransform.cat(
            [s.transformation for s in stacks_in]
        ).matrix().to(device),
        "stacks": torch.cat([s.slices for s in stacks_in], dim=0).to(device),
        "positions": positions,
        "slice_thickness": stacks_in[0].thickness,
    }
    
    with torch.no_grad():
        t_out, v_out, _ = model(data)
        
    # =========================================================================
    #  使用 torch.split 按长度分割，替代布尔索引筛选
    # =========================================================================
    
    # 1. 提取总输出的矩阵 (t_out[-1] 是 RigidTransform 对象，先转为 Tensor)
    all_transforms_matrix = t_out[-1].matrix()
    
    # 2. 获取每个 Stack 的长度
    stack_lengths = [len(s) for s in stacks_in]
    
    # 3. 按长度切分 Tensor -> 得到一个 tuple，每个元素对应一个 stack 的变换矩阵
    transforms_split = torch.split(all_transforms_matrix, stack_lengths, dim=0)
    
    stacks_out = []
    for i in range(len(stacks_in)):
        stack_out = stacks_in[i].clone(zero=False, deep=False)
        
        # 取出对应的切片部分
        t_stack_matrix = transforms_split[i]
        
        # 赋值给 stack.transformation (Stack 会自动将其封装为 RigidTransform)
        stack_out.transformation = t_stack_matrix.to(origin_device)
        stack_out.slices = stack_out.slices.to(origin_device)
        stack_out.mask = stack_out.mask.to(origin_device)
        stacks_out.append(stack_out)
    # =========================================================================

    slices = []
    for stack in stacks_out:
        idx_nonempty = stack.mask.flatten(1).any(1)
        stack.slices /= torch.quantile(stack.slices[stack.mask], 0.99)
        slices.extend(stack[idx_nonempty])

    return slices, v_out[-1].to(origin_device)

