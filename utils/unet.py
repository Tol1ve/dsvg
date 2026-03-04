
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from torch.cuda.amp import autocast, GradScaler
from nesvor.image import Stack,Volume
from nesvor.svr.reconstruction import simulate_slices,simulated_error,slices_scale,get_PSF
def DEBUG_test_save(img,name,save_dir='/home/lvyao/git/med-ddpm/results/DM_results/test_dpcp_generator'):
    import os 
    import nibabel as nib 
    import numpy as np
    os.makedirs(save_dir,exist_ok=True)
    out=img.cpu().detach().numpy()
    nifti_img = nib.Nifti1Image(out.squeeze(), affine=np.eye(4))
    nib.save(nifti_img,os.path.join(save_dir,name))
# -------- 3D U-Net building blocks ----------
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, act=nn.LeakyReLU(0.1, inplace=True)):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, padding=p, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            act,
            nn.Conv3d(out_ch, out_ch, k, padding=p, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            act,
        )
    def forward(self, x): return self.block(x)

class Down3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.down = nn.Conv3d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.down(x)

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
    def forward(self, x): return self.up(x)

class UNet3D(nn.Module):
    """
    输入: [B, 1, D, H, W] 的模板体积（或多通道也可）
    输出: [B, 1, D, H, W] 的修正体积
    """
    def __init__(self, in_ch=1, base_ch=32, out_ch=1):
        super().__init__()
        C = base_ch
        self.enc1 = ConvBlock3D(in_ch, C)       # D,H,W
        self.down1 = Down3D(C)                  # D/2
        self.enc2 = ConvBlock3D(C, 2*C)         # 2C
        self.down2 = Down3D(2*C)                # D/4
        self.enc3 = ConvBlock3D(2*C, 4*C)       # 4C (bottleneck)

        self.up2  = Up3D(4*C, 2*C)
        self.dec2 = ConvBlock3D(4*C, 2*C)
        self.up1  = Up3D(2*C, C)
        self.dec1 = ConvBlock3D(2*C, C)

        self.out  = nn.Conv3d(C, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        y  = self.out(d1)
        return y
def train_reconstruction(
    model: nn.Module,
    template_vol: torch.Tensor,   # 初始模板 [B,1,D,H,W]，可固定为某个 atlas
    train_loader,                 # 迭代提供 stack (含 .slices / .mask)
    psf_tensor: Optional[torch.Tensor],
    epochs: int = 200,
    lr: float = 2e-4,
    tv_w: float = 1e-4,
    lap_w: float = 1e-5,
    device: str = "cuda",
    max_grad_norm: float = 1.0
):
    model = model.to(device)
    template_vol = template_vol.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    for ep in range(1, epochs + 1):
        model.train()
        log = {"loss": 0.0, "data": 0.0, "tv": 0.0, "lap": 0.0}
        for batch in train_loader:
            # 假设 batch 提供 stack；若有 batch 内多样本，把 template_vol repeat
            stack = batch["stack"].to(device)  # 需你在 Dataset 中把 .to(device) 做好或这里迁移
            B = stack.slices.shape[0]
            tpl = template_vol if template_vol.shape[0] == B else template_vol[:1].repeat(B,1,1,1,1)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.startswith("cuda"))):
                pred_vol = model(tpl)
                loss, aux = compute_loss(
                    stack=stack,
                    pred_vol=pred_vol,
                    psf_tensor=psf_tensor,
                    tv_w=tv_w,
                    lap_w=lap_w
                )

            scaler.scale(loss).backward()
            # 稳定训练
            if max_grad_norm is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(opt)
            scaler.update()

            # 累积日志
            log["loss"] += float(loss.detach())
            log["data"] += float(aux["data"])
            log["tv"]   += float(aux["tv"])
            log["lap"]  += float(aux["lap"])

        n = len(train_loader)
        print(f"[Ep {ep:03d}] loss={log['loss']/n:.4f} data={log['data']/n:.4f} "
              f"tv={log['tv']/n:.4f} lap={log['lap']/n:.4f}")

    # 训练完成后，用全批次/或模板本身前向导出最终体积
    model.eval()
    with torch.no_grad():
        final_vol = model(template_vol)  # [B,1,D,H,W]
    return final_vol

def solve_slice_scale_ls(
    real_slices: torch.Tensor,    # [B, N, H, W]
    sim_slices: torch.Tensor,     # [B, N, H, W]
    weight: Optional[torch.Tensor] = None,  # [B, N, H, W]
    mask:   Optional[torch.Tensor] = None,  # [B, N, H, W] (0/1)
    eps: float = 1e-8
) -> torch.Tensor:
    """
    按切片闭式估计 scale: argmin_s || s*real - sim ||_W^2
    s = <W*real, W*sim> / <W*real, W*real>
    可微（autograd能穿过所有张量运算）
    返回: [B, N]
    """
    x = real_slices
    y = sim_slices
    w = 1.0
    if weight is not None: w = w * weight
    if mask   is not None: w = w * mask

    num = (w * x * y).sum(dim=(-1, -2))           # [B,N]
    den = (w * x * x).sum(dim=(-1, -2)) + eps     # [B,N]
    s   = num / den
    return s
def charbonnier(x, eps=1e-6):  # 更稳健的 L1
    return torch.sqrt(x * x + eps)

def total_variation_3d(vol, eps=1e-6):
    dx = vol[:, :, 1:, :, :] - vol[:, :, :-1, :, :]
    dy = vol[:, :, :, 1:, :] - vol[:, :, :, :-1, :]
    dz = vol[:, :, :, :, 1:] - vol[:, :, :, :, :-1]
    return (charbonnier(dx, eps).mean() +
            charbonnier(dy, eps).mean() +
            charbonnier(dz, eps).mean())

def laplacian_smooth_3d(vol):
    # 6邻域离散拉普拉斯
    pad = (1,1,1,1,1,1)
    vpad = F.pad(vol, pad, mode='replicate')
    cx = vpad[:, :, 2:, 1:-1, 1:-1] + vpad[:, :, :-2, 1:-1, 1:-1]
    cy = vpad[:, :, 1:-1, 2:, 1:-1] + vpad[:, :, 1:-1, :-2, 1:-1]
    cz = vpad[:, :, 1:-1, 1:-1, 2:] + vpad[:, :, 1:-1, 1:-1, :-2]
    center = 6 * vol
    lap = cx + cy + cz - center
    return (lap * lap).mean()
class ReconSystem(nn.Module):
    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet  # 输入模板 -> 修正体积

    def forward(self, template_vol: torch.Tensor):
        """
        template_vol: [B, 1, D, H, W]
        return: pred_vol 同尺寸
        """
        return self.unet(template_vol)

def compute_loss(
    stack,                              # 真实切片容器, 至少有 .slices [B,N,H,W]，可选 .mask
    pred_vol: torch.Tensor,             # [B,1,D,H,W]
    psf_tensor: Optional[torch.Tensor], # 传给 simulate_slices
    tv_w: float = 1e-4,
    lap_w: float = 1e-5,
    robust_eps: float = 1e-6,
    use_scale_ls: bool = True,
):
    # 物理前向
    slices_sim, slices_weight = simulate_slices(
        stack=stack,
        volume=pred_vol,    # 直接给张量即可，若需 Volume 类可包装
        return_weight=True,
        use_mask=True,
        psf=psf_tensor,
    )
    y_sim = slices_sim.slices        # [B,N,H,W]
    w     = slices_weight.slices if hasattr(slices_weight, 'slices') else slices_weight
    x_real = stack.slices            # [B,N,H,W]
    mask   = getattr(stack, 'mask', None)  # 若存在

    # 逐切片强度尺度
    if use_scale_ls:
        scale = solve_slice_scale_ls(x_real, y_sim, weight=w, mask=mask)  # [B,N]
    else:
        # 或者用可学习参数: scale_param = softplus(param)；此处略
        scale = torch.ones_like(x_real.mean(dim=(-1,-2)))  # [B,N]

    err = x_real * scale[..., None, None] - y_sim          # [B,N,H,W]

    # 加权 + 鲁棒主损
    if w is not None: err = err * w
    if mask is not None: err = err * mask
    data_term = charbonnier(err, eps=robust_eps).mean()

    # 体积正则
    tv  = total_variation_3d(pred_vol)
    lap = laplacian_smooth_3d(pred_vol)

    loss = data_term + tv_w * tv + lap_w * lap
    aux  = {
        "data": data_term.detach(),
        "tv": tv.detach(),
        "lap": lap.detach(),
        "scale_mean": scale.mean().detach()
    }
    return loss, aux


class GeneratorUNet(nn.Module):
    """输入 z: [B, C, D, H, W] 噪声，输出体积 [B,1,D,H,W]"""
    def __init__(self, in_ch=4, base_ch=32, out_ch=1):
        super().__init__()
        self.unet = UNet3D(in_ch=in_ch, base_ch=base_ch, out_ch=out_ch)

    def forward(self, z):
        return self.unet(z)
class LearnableSliceScale(nn.Module):
    """
    """
    def __init__(
        self,
        num_slices: int,
        init: float = 1.0,
        scale_min: float = 0.05,
        scale_max: float = 10.0,
        use_log_space_reg: bool = True,
    ):
        super().__init__()
        self.num_slices = num_slices
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.use_log_space_reg = use_log_space_reg

        # 让 softplus(logit) ≈ init
        # softplus^{-1}(init - scale_min) 近似: log(exp(x)-1)
        init_eff = max(init - scale_min, 1e-6)
        logit_init = torch.log(torch.exp(torch.tensor(init_eff)) - 1.0)
        self.logit_scale = nn.Parameter(logit_init.repeat(num_slices))  # [N]

    def forward(self, B: int, device: torch.device) -> torch.Tensor:
        scale = F.softplus(self.logit_scale) + self.scale_min  # [N]
        scale = torch.clamp(scale, self.scale_min, self.scale_max)
        return scale[None, :].expand(B, -1).to(device)         # [B,N]

    def reg_loss(self, scale: torch.Tensor) -> torch.Tensor:

        if self.use_log_space_reg:
            # 对乘性缩放更合理：log(scale) 靠近 0
            return (torch.log(scale + 1e-8) ** 2).mean()
        else:
            return ((scale - 1.0) ** 2).mean()
# ------- 与 simulate_slices 串起来的损失 -------
def recon_loss_from_slices(
    stack,              # 真实切片容器, 至少有 stack.slices [B,N,H,W] 和可选 stack.mask
    vol_pred,           # [B,1,D,H,W] = G(z)
    psf_tensor=None,
    tv_w: float = 1e-4,
    lap_w: float = 1e-5,
    robust_eps: float = 1e-6,
    scale_module: Optional[nn.Module] = None,
    scale_reg_w: float = 1e-3,     # 你可以从 1e-3 起试
    detach_scale_from_err: bool = False,
):
    slices_sim, slices_weight = simulate_slices(
        slices=stack,
        volume=vol_pred,
        return_weight=True,
        use_mask=True,
        psf=psf_tensor,
    )
    y_sim = slices_sim.slices             # [B,N,H,W]
    w     = slices_weight.slices if hasattr(slices_weight, 'slices') else slices_weight
    x_real = stack.slices                 # [B,N,H,W]
    mask   = getattr(stack, 'mask', None)
    
    N,B, _, _ = x_real.shape
    device = x_real.device
    # 逐切片亮度尺度
    if scale_module is not None:
        scale = scale_module(B=B, device=device)  # [B,N]
        if detach_scale_from_err:
            scale_used = scale.detach()
        else:
            scale_used = scale
        scale_reg = scale_module.reg_loss(scale) * scale_reg_w
    else:
        scale_used = torch.ones((B, N), device=device)
        scale_reg = torch.tensor(0.0, device=device)

    err = x_real * scale_used[..., None, None] - y_sim
    if w is not None:   err = err * w
    if mask is not None: err = err * mask

    data_term = charbonnier(err, eps=robust_eps).mean()
    vol_pred_tensor=vol_pred.image.view((1,1)+vol_pred.image.shape)
    tv  = total_variation_3d(vol_pred_tensor)
    lap = laplacian_smooth_3d(vol_pred_tensor)

    loss = data_term + tv_w * tv + lap_w * lap
    aux = {
        "data": data_term.detach(),
        "tv": tv.detach(),
        "lap": lap.detach(),
        "scale_mean": (scale_used.mean().detach() if scale_module is not None else torch.tensor(1.0, device=device)),
        "scale_reg": scale_reg.detach(),
    }
    return loss, aux

# ================= 阶段A：预训练 Gθ 使 Gθ(z0) ≈ template_vol =================
@torch.no_grad()
def make_fixed_noise_like(template_vol, in_ch=4, device="cuda", seed=0):
    torch.manual_seed(seed)
    B, _, D, H, W = template_vol.shape
    z0 = torch.randn(B, in_ch, D, H, W, device=device)
    return z0

def pretrain_generator_to_template(
    G: nn.Module,
    template_vol: torch.Tensor,       # [B,1,D,H,W]
    steps: int = 2000,
    lr: float = 2e-4,
    in_ch: int = 4,
    device: str = "cuda",
    l1_w: float = 1.0,
    tv_w: float = 1e-5,
    lap_w: float = 1e-6,
    seed: int = 0,
):
    G = G.to(device).train()
    template_vol = template_vol.to(device)
    z0 = make_fixed_noise_like(template_vol, in_ch=in_ch, device=device, seed=seed)  # 固定噪声

    opt = torch.optim.Adam(G.parameters(), lr=lr)

    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        v = G(z0)  # 期望逼近 template_vol
        l_recon = F.l1_loss(v, template_vol) * l1_w
        l_tv    = total_variation_3d(v) * tv_w
        l_lap   = laplacian_smooth_3d(v) * lap_w
        loss = l_recon + l_tv + l_lap
        loss.backward()
        opt.step()

        if t % 100 == 0:
            print(f"[Pretrain {t:04d}] L1={float(l_recon):.4g} TV={float(l_tv):.4g} Lap={float(l_lap):.4g}")
    return z0  # 返回预训练使用的固定噪声（后续可作初始化参考）

# ================= 阶段B：冻结 θ，优化 z（可选微调 θ） =================
def slice_driven_latent_optim(
    G: nn.Module,
    stack,                          # DataLoader 里的一个 batch（或迭代），至少含 stack.slices
    psf_tensor=None,
    B: int = 1,
    in_ch: int = 4,
    steps: int = 1000,
    lr_z: float = 1e-2,
    lr_theta: float = 0.0,          # >0 时允许微调 θ（很小）
    z_init: torch.Tensor = None,    # 若提供，用其初始化；否则随机
    z_prior_w: float = 1e-4,        # ||z||^2 正则，防止 z 跑飞
    tv_w: float = 1e-4,
    lap_w: float = 1e-5,
    device: str = "cuda",
):
    G = G.to(device)
    stack = stack.to(device)  # 需你的 Stack 类实现 .to()

    # 构造 z（可学习）
    if z_init is None:
        # 用 stack 的体积尺寸来决定 z 的 D,H,W；这里假设你能从 stack/metadata 里拿到目标体积网格
        # 如果没有，可在预训练阶段记下 template_vol 的体素大小并沿用
        # 这里直接从 G 的第一层需要的空间大小约定：以 stack.meta 里 D,H,W 为例
        D, H, W = getattr(stack, "volume_shape")  # 你可以在 Dataset 里提供
        z = torch.randn(B, in_ch, D, H, W, device=device, requires_grad=True)
    else:
        z = z_init.detach().clone().to(device).requires_grad_(True)

    # 优化器
    params = [z]
    if lr_theta > 0:
        for p in G.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam([{"params": [z], "lr": lr_z},
                                {"params": G.parameters(), "lr": lr_theta}])
    else:
        for p in G.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam([{"params": [z], "lr": lr_z}])

    for t in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        vol_pred = G(z)                             # [B,1,D,H,W]
        loss_data, aux = recon_loss_from_slices(    # simulate_slices + 重投影损失
            stack=stack, vol_pred=vol_pred, psf_tensor=psf_tensor,
            tv_w=tv_w, lap_w=lap_w
        )
        loss_prior = (z * z).mean() * z_prior_w
        loss = loss_data + loss_prior
        loss.backward()
        opt.step()

        if t % 50 == 0:
            print(f"[Latent {t:04d}] loss={float(loss):.4g} data={float(aux['data']):.4g} "
                  f"tv={float(aux['tv']):.3g} lap={float(aux['lap']):.3g} |z|^2={float(loss_prior/z_prior_w):.3g}")

    with torch.no_grad():
        final_vol = G(z)  # 优化后的体积
    return final_vol, z


# -------- 小积木 --------
class ResBlock3D(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv3d(ch, ch, k, padding=p, bias=False)
        self.in1   = nn.InstanceNorm3d(ch, affine=True)
        self.act1  = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(ch, ch, k, padding=p, bias=False)
        self.in2   = nn.InstanceNorm3d(ch, affine=True)
        self.act2  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.act1(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        return self.act2(x + y)

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.rb   = ResBlock3D(out_ch)

    def forward(self, x):
        x = self.up(x)
        return self.rb(x)

# -------- 卷积式 3D 解码器 --------
# -------- 小积木 --------
class ResBlock3D(nn.Module):
    def __init__(self, ch, k=3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv3d(ch, ch, k, padding=p, bias=False)
        self.in1   = nn.InstanceNorm3d(ch, affine=True)
        self.act1  = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(ch, ch, k, padding=p, bias=False)
        self.in2   = nn.InstanceNorm3d(ch, affine=True)
        self.act2  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.act1(self.in1(self.conv1(x)))
        y = self.in2(self.conv2(y))
        return self.act2(x + y)

class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.rb   = ResBlock3D(out_ch)

    def forward(self, x):
        x = self.up(x)
        return self.rb(x)

# -------- 卷积式 3D 解码器 --------
class ConvDecoder3D(nn.Module):
    """
    从低分辨率潜变量 z: [B, Cz, Dz, Hz, Wz] 逐级上采样为体积 [B, 1, D, H, W]
    up_factors: 例如 (2,2,2,2) 表示总上采样 16x
    channels:   每层通道数，从底到顶，例如 [256,128,64,32]
    """
    def __init__(self, z_ch=32, channels=(256,128,64,32), up_factors=(2,2,2,2), out_ch=1):
        super().__init__()
        assert len(channels) == len(up_factors), "channels 与 up_factors 长度需一致"
        self.in_proj = nn.Sequential(
            nn.Conv3d(z_ch, channels[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels[0], affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock3D(channels[0])
        )
        self.ups = nn.ModuleList()
        for i in range(1, len(channels)):
            self.ups.append(UpBlock3D(channels[i-1], channels[i]))

        self.head = nn.Conv3d(channels[-1], out_ch, kernel_size=1)

    def forward(self, z):
        x = self.in_proj(z)
        for up in self.ups:
            x = up(x)
        y = self.head(x)
        return y
def l2_sp_reg(current_params, ref_params):
    """
    L2-SP: sum(||theta - theta0||^2)
    current_params/ref_params 都是可迭代的参数张量列表, 一一对应
    """
    reg = 0.0
    for p, p0 in zip(current_params, ref_params):
        reg = reg + torch.sum((p - p0)**2)
    return reg

def slice_driven_theta_optim_conv_decoder(
    G: nn.Module,
    stack,                      # 含 .slices / (可选) .mask / .to(device)
    z_fixed: torch.Tensor,      # 预训练阶段的低分辨率潜变量 z0，形状 [B,Cz,Dz,Hz,Wz]
    psf_tensor=None,
    steps: int = 800,
    lr_theta: float = 1e-5,     # 小学习率！(1e-5~5e-5 常用)
    tv_w: float = 1e-4,
    lap_w: float = 1e-5,
    sp_w: float = 1e-4,         # L2-SP 权重
    max_grad_norm: float = 1.0,
    device: str = "cuda",
):
    G = G.to(device)
    stack = stack.to(device)
    z_fixed = z_fixed.to(device).detach()        # 固定输入
    for p in G.parameters():
        p.requires_grad_(True)

    # 备份预训练权重，用于 L2-SP
    theta0 = [p.detach().clone() for p in G.parameters()]

    opt = torch.optim.Adam(G.parameters(), lr=lr_theta)

    for t in range(1, steps+1):
        opt.zero_grad(set_to_none=True)

        vol_pred = G(z_fixed)   # 仅 θ 在变
        loss_data, aux = recon_loss_from_slices(
            stack=stack, vol_pred=vol_pred, psf_tensor=psf_tensor,
            tv_w=tv_w, lap_w=lap_w
        )
        # L2-SP: 约束 θ 贴近预训练 θ0，防止先验“散掉”
        sp = l2_sp_reg(list(G.parameters()), theta0) * sp_w

        loss = loss_data + sp
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_grad_norm)
        opt.step()

        if t % 50 == 0:
            print(f"[Theta {t:04d}] loss={float(loss):.4g} data={float(aux['data']):.4g} "
                  f"tv={float(aux['tv']):.3g} lap={float(aux['lap']):.3g} L2SP={float(sp/sp_w):.3g}")

    with torch.no_grad():
        final_vol = G(z_fixed)
    return final_vol
@torch.no_grad()
def make_fixed_latent(template_vol: torch.Tensor, z_ch: int, down_scale: int, device: str, seed: int = 0):
    torch.manual_seed(seed)
    D, H, W = template_vol.shape
    Dz, Hz, Wz = max(1, D // down_scale), max(1, H // down_scale), max(1, W // down_scale)
    return torch.randn(1, z_ch, Dz, Hz, Wz, device=device)
def optimize_volume_from_slices_pipeline(
    volume: torch.Tensor,          # [B,1,D,H,W] (template volume)
    slices,                         # Stack, has .slices [B,N,H,W], and .to(device)
    psf_tensor=None,
    device: str = "cuda",

    # --- stage A ---
    stageA_steps: int = 800,
    stageA_lr: float = 2e-4,
    stageA_l1_w: float = 1.0,
    stageA_tv_w: float = 1e-5,
    stageA_lap_w: float = 1e-6,

    # --- stage B (theta finetune, z fixed) ---
    stageB_steps: int = 1500,
    stageB_lr_theta: float = 1e-5,
    stageB_tv_w: float = 1e-5,
    stageB_lap_w: float = 1e-6,
    stageB_sp_w: float = 1e-6,     # L2-SP weight
    max_grad_norm: float = 1.0,

    seed: int = 0,
    verbose_every: int = 100,
) -> torch.Tensor:
    """
    返回: updated_volume [B,1,D,H,W]
    """
    #
    z_ch = 32
    down_scale = 8                 # latent = 12×12×12
    channels = (256, 128, 64, 32)
    # --- move to device ---
    template_vol = volume.image.to(device)
    output_resolution=volume.resolution_x
    stack = Stack.cat(slices)
    psf_tensor = get_PSF(
            res_ratio=(
                stack.resolution_x / output_resolution,
                stack.resolution_y / output_resolution,
                stack.thickness / output_resolution,
            ),
            device=volume.device,
    )
    # --- build generator ---
    G = ConvDecoder3D(z_ch=z_ch, channels=channels, out_ch=1).to(device)

    N = len(slices)
    scale_module = LearnableSliceScale(num_slices=N, init=1.0, scale_min=0.05, scale_max=5.0).to(device)
    # make fixed latent z0 (small)
    z0 = make_fixed_latent(template_vol, z_ch=z_ch, down_scale=down_scale, device=device, seed=seed)

    # =========================
    # Stage A: fit template
    # =========================
    G.train()
    optA = torch.optim.Adam(G.parameters(), lr=stageA_lr)

    for t in range(1, stageA_steps + 1):
        optA.zero_grad(set_to_none=True)
        vol_hat = G(z0)
        vol_hat = F.interpolate(vol_hat, size=(100,100,100), mode="trilinear", align_corners=False)
        l1 = F.l1_loss(vol_hat, template_vol) * stageA_l1_w
        tv = total_variation_3d(vol_hat) * stageA_tv_w
        lp = laplacian_smooth_3d(vol_hat) * stageA_lap_w
        lossA = l1 + tv + lp

        lossA.backward()
        optA.step()

        if (verbose_every is not None) and (t % verbose_every == 0):
            print(f"[StageA {t:04d}] loss={float(lossA):.4g} l1={float(l1):.4g} tv={float(tv):.3g} lap={float(lp):.3g}")

    # save theta0 for L2-SP
    theta0 = [p.detach().clone() for p in G.parameters()]

    # =========================
    # Stage B: slice-driven finetune theta, z fixed
    # =========================
    G.train()
    for p in G.parameters():
        p.requires_grad_(True)
    optB = torch.optim.Adam(
        list(G.parameters()) + list(scale_module.parameters()),
        lr=stageB_lr_theta
    )

    z_fixed = z0.detach()  # keep z fixed

    for t in range(1, stageB_steps + 1):
        optB.zero_grad(set_to_none=True)
        
        vol_pred = G(z_fixed)
        volume_pred=Volume(vol_pred.squeeze().contiguous(),resolution_x=output_resolution,transformation=volume.transformation)
        loss_data, aux = recon_loss_from_slices(
            stack=stack, vol_pred=volume_pred, psf_tensor=psf_tensor,
            tv_w=stageB_tv_w, lap_w=stageB_lap_w,
            scale_module=scale_module,
            scale_reg_w=1e-3,              # 先从 1e-3 试
            detach_scale_from_err=False
        )
        sp = l2_sp_reg(list(G.parameters()), theta0) * stageB_sp_w
        lossB = loss_data + sp

        lossB.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_grad_norm)
        optB.step()

        if (verbose_every is not None) and (t % verbose_every == 0):
            print(f"[StageB {t:04d}] loss={float(lossB):.4g} data={float(aux['data']):.4g} "
                  f"tv={float(aux['tv']):.3g} lap={float(aux['lap']):.3g} sp={float(sp/stageB_sp_w):.3g}")

    # =========================
    # Output updated volume
    # =========================
    G.eval()
    with torch.no_grad():
        updated_vol = G(z_fixed)  # [B,1,D,H,W]
    updated_vol  = F.interpolate(updated_vol , size=(100,100,100), mode="trilinear", align_corners=False)
    mask=vol_hat>0.06
    updated_vol[mask==False]=0
    return updated_vol