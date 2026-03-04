#-*- coding:utf-8 -*-
#
# *Main part of the code is adopted from the following repository: https://github.com/lucidrains/denoising-diffusion-pytorch


import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
import nibabel as nib
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
try:
    from apex import amp
    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")


DEBUG=False
def DEBUG_test_save(img,name,save_dir='/home/lvyao/git/med-ddpm/results/results_0_analyse/evaluate_output'):
    import nibabel as nib
    import numpy as np
    os.makedirs(save_dir,exist_ok=True)
    out=img.cpu().numpy()
    nifti_img = nib.Nifti1Image(out.squeeze(), affine=np.eye(4))
    nib.save(nifti_img,os.path.join(save_dir,name))

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# gaussian diffusion trainer class 

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels = 1,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None,
        with_condition = False,
        with_pairwised = False,
        apply_bce = False,
        lambda_bce = 0.0
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.with_pairwised = with_pairwised
        self.apply_bce = apply_bce
        self.lambda_bce = lambda_bce

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef3', to_torch(
            1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t, c=None):
        x_hat = 0
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        x_hat = 0.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat = 0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, c = None):
        if self.with_condition:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([x, c ], 1), t))
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t, c=c)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, condition_tensors=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    @torch.no_grad()
    def p_sample_cfg(self, x, t, condition_tensors, null_condition_tensors, guidance_scale):
        # 1) 有条件噪声预测
        eps_cond = self.model(x, t, condition_tensors=condition_tensors)

        # 2) 无条件噪声预测（传全 0 条件，但通道一致）
        eps_uncond = self.model(x, t, condition_tensors=null_condition_tensors)

        # 3) CFG 组合
        w = guidance_scale
        eps = (1.0 + w) * eps_cond - w * eps_uncond

        # 4) 用组合后的 eps 做一次 DDPM 更新
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t)

        if t[0] == 0:
            return model_mean

        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensors = None,debug_save_dir=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if self.with_condition:
                t = torch.full((b,), i, device=device, dtype=torch.long)
                img = self.p_sample(img, t, condition_tensors=condition_tensors)
                if DEBUG and i%10==0:
                    DEBUG_test_save(img,f'diff-process-{i}.nii.gz',save_dir=debug_save_dir)
            else:
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img
    @torch.no_grad()
    def p_sample_single(self,shape,input_img,timestep,condition_tensors=None):
        device = self.betas.device
        b = shape[0]
        if self.with_condition:
            t = torch.full((b,), timestep, device=device, dtype=torch.long)
            img = self.p_sample(input_img, t, condition_tensors=condition_tensors)

        else:
            img = self.p_sample(input_img, torch.full((b,), timestep, device=device, dtype=torch.long))
        return img
    # @torch.no_grad()
    def sample(self, batch_size = 2, condition_tensors = None):
        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, depth_size, image_size, image_size), condition_tensors = condition_tensors)
    @torch.no_grad()
    def sample_same_shape(self, batch_size = 2, condition_tensors = None,debug_save_dir=None):
        batch_size, _, depth_size, height_size, width_size=condition_tensors.shape
        channels=self.channels

    def sample_from_z(
        self,
        img,
        batch_size=1,
        condition_tensors=None,
        device="cpu",
        mode="ddpm",
        ddim_steps=None,
        eta=0.0,
    ):
        """
        """

        img = img.to(device)

        if mode.lower() == "ddpm":
            # 原版 DDPM：逐步 i = T-1 ... 0
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step (DDPM)",
                total=self.num_timesteps,
            ):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                if self.with_condition:
                    img = self.p_sample(img, t, condition_tensors=condition_tensors)
                else:
                    img = self.p_sample(img, t)
            return img

        elif mode.lower() == "ddim":
            # -------- DDIM 时间步序列（可跳步）--------
            T = self.num_timesteps
            if ddim_steps is None:
                ddim_steps = T  # 不跳步，就等价于全步 DDIM

            # 例如：T=1000, steps=50 -> [999, ..., 0] 的均匀子序列
            times = torch.linspace(0, T - 1, steps=ddim_steps, device=device).long()
            times = torch.flip(times, dims=[0])  # 变成从大到小
            # 为了计算 t_prev，末尾补一个 -1（表示到 x_{-1}，可视为 x0）
            times_prev = torch.cat([times[1:], torch.tensor([-1], device=device)])

            for idx in tqdm(range(ddim_steps), desc="sampling loop time step (DDIM)", total=ddim_steps):
                t = times[idx].repeat(batch_size)
                t_prev = times_prev[idx].repeat(batch_size)

                # 取 alpha_bar_t, alpha_bar_{t_prev}
                alpha_bar_t = extract(self.alphas_cumprod, t, img.shape)  # \bar{alpha}_t
                if (t_prev[0].item() >= 0):
                    alpha_bar_prev = extract(self.alphas_cumprod, t_prev, img.shape)
                else:
                    # t_prev = -1 表示直接到 x0
                    alpha_bar_prev = torch.ones_like(alpha_bar_t)

                # -------- 1) 预测 eps（噪声）--------
                # 你需要把这里替换成你自己的网络输出接口
                if self.with_condition:
                    eps = self.denoise_fn(torch.cat([img, condition_tensors], 1), t)
                else:
                    eps = self.denoise_fn(img, t)

                # -------- 2) 由 eps 得到 x0_pred --------
                # x0 = (x_t - sqrt(1-alpha_bar_t)*eps) / sqrt(alpha_bar_t)
                x0_pred = (img - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

                # 可选：clip（很多实现会 clip 到 [-1,1] 或 [0,1]）
                if hasattr(self, "clip_denoised") and self.clip_denoised:
                    x0_pred = x0_pred.clamp(-1.0, 1.0)

                # -------- 3) DDIM 更新公式 --------
                # sigma_t = eta * sqrt((1-a_prev)/(1-a_t)) * sqrt(1 - a_t/a_prev)
                # 注意：这里用 alpha_bar 表示 cumulative product
                # 推导版本不同实现略有差异，但这是常见写法
                sigma = eta * torch.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_prev)
                )

                # direction: sqrt(1 - alpha_prev - sigma^2) * eps
                dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps

                noise = torch.randn_like(img) if eta > 0 else torch.zeros_like(img)

                img = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise

            return img

        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from ['ddpm', 'ddim'].")
    @torch.no_grad()
    def sample_from_z_t(
        self,
        z: torch.Tensor,
        t_start: int,
        condition_tensors: torch.Tensor = None,
        device: str = None,
        debug_save_dir: str = None,
        debug_interval: int = 10,
    ):
        """
        从给定时间步 t_start 和初始张量 z (= x_{t_start}) 开始，
        逐步调用 p_sample 去噪到 t=0，得到 img_t0。

        Args:
            z: Tensor, 形状 [B, C, D, H, W]，视为 x_{t_start}
            t_start: int, 起始时间步（0 <= t_start < self.num_timesteps）
            condition_tensors: 条件输入（当 self.with_condition=True 时使用）
            device: 可选，强制放到某个 device；默认沿用 z.device
            debug_save_dir: 可选，若开启 DEBUG 则保存中间结果
            debug_interval: 保存间隔（默认每 10 步保存一次）

        Returns:
            img_t0: Tensor, 去噪到 t=0 的结果
        """
        assert isinstance(t_start, int), "t_start 必须是 int"
        assert 0 <= t_start < self.num_timesteps, f"t_start={t_start} 超出范围 [0, {self.num_timesteps-1}]"
        assert z.ndim == 5, f"z 应为 5D [B,C,D,H,W]，但得到 {z.shape}"

        if device is None:
            device = z.device
        else:
            device = torch.device(device)

        img = z.to(device)
        b = img.shape[0]

        # 从 t_start -> 0
        for i in tqdm(reversed(range(0, t_start + 1)),
                    desc=f"sample_from_z_t (t_start={t_start})",
                    total=t_start + 1):
            t = torch.full((b,), i, device=device, dtype=torch.long)

            if self.with_condition:
                img = self.p_sample(img, t, condition_tensors=condition_tensors)
            else:
                img = self.p_sample(img, t)
        return img
    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    def q_sample(self, x_start, t, noise=None, c=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_hat = 0.
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise + x_hat
        )

    def p_losses(self, x_start, t, condition_tensors = None, noise = None):
        b, c, h, w, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.with_condition:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(torch.cat([x_noisy, condition_tensors], 1), t)
        else:
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        b, c, d, h, w, device, img_size, depth_size = *x.shape, x.device, self.image_size, self.depth_size
        #assert h == img_size and w == img_size and d == depth_size, f'Expected dimensions: height={img_size}, width={img_size}, depth={depth_size}. Actual: height={h}, width={w}, depth={d}.'
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if condition_tensors is not None and condition_tensors.shape[1]==1:
            condition_tensors=condition_tensors.repeat(1, 2, 1, 1, 1)
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)

class CFGDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels=1,
        timesteps=1000,
        loss_type='l1',
        betas=None,
        with_condition=False,
        with_pairwised=False,
        apply_bce=False,
        lambda_bce=0.0
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.with_pairwised = with_pairwised
        self.apply_bce = apply_bce
        self.lambda_bce = lambda_bce

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # q(x_t | x_0) 相关
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # q(x_{t-1} | x_t, x_0) 相关
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        )
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        )
        self.register_buffer(
            'posterior_mean_coef3',
            to_torch(1. - (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        )

    # --------- 基本 q / p 相关函数（没动） ---------
    def q_mean_variance(self, x_start, t, c=None):
        x_hat = 0
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + x_hat
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise, c=None):
        x_hat = 0.
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_hat
        )

    def q_posterior(self, x_start, x_t, t, c=None):
        x_hat = 0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_hat
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ======== 小工具：给定 x_t / t / cond 预测 epsilon（噪声）========
    def _predict_eps(self, x, t, condition_tensors=None):
        """
        统一封装 denoise_fn 的调用，返回噪声预测 eps。
        注意：with_condition=True 时内部是 cat([x, c], 1)。
        """
        if self.with_condition and condition_tensors is not None:
            eps = self.denoise_fn(torch.cat([x, condition_tensors], 1), t)
        else:
            eps = self.denoise_fn(x, t)
        return eps

    # ======== 原版 DDPM 的 mean/var，用 eps 直接推 x0 再算 posterior ========
    def p_mean_variance(self, x, t, clip_denoised: bool, c=None):
        eps = self._predict_eps(x, t, condition_tensors=c)
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t, c=c
        )
        return model_mean, posterior_variance, posterior_log_variance

    # ======== DDPM：普通单步采样（无 CFG）========
    @torch.no_grad()
    def p_sample(self, x, t, condition_tensors=None, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, c=condition_tensors, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # ======== DDPM：加入 CFG 的单步采样 ========
    @torch.no_grad()
    def p_sample_cfg(self, x, t, condition_tensors, null_condition_tensors, guidance_scale):
        """
        基于“噪声预测”的 CFG：
        eps_cfg = (1+w)*eps_cond - w*eps_uncond
        然后用 eps_cfg → x0 → q_posterior → x_{t-1}
        """
        # 有条件 / 无条件 eps
        eps_cond = self._predict_eps(x, t, condition_tensors=condition_tensors)
        eps_uncond = self._predict_eps(x, t, condition_tensors=null_condition_tensors)

        w = guidance_scale
        eps = (1.0 + w) * eps_cond - w * eps_uncond

        # 用 eps 重建 x0
        x_recon = self.predict_start_from_noise(x, t=t, noise=eps)
        x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        b, *_, device = *x.shape, x.device
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * (0.5 * posterior_log_variance).exp() * noise

    # ======== DDIM：单步更新（给定 eps）========
    def ddim_step(self, x_t, t, t_prev, eps, eta=0.0):
        """
        标准 DDIM 更新，基于 eps（噪声预测）。
        t, t_prev: [B] long
        eta=0 时为完全确定性 DDIM。
        """
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t.shape)          # \bar{α}_t
        alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)  # \bar{α}_{t-1}

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)

        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # 1) 先预测 x0
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
        x0_pred = x0_pred.clamp(-1., 1.)

        # 如果已经是 t=0，就直接返回 x0
        if (t_prev[0] == 0) and (t[0] == 0):
            return x0_pred

        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

        # DDIM 中 sigma_t
        sigma_t = eta * torch.sqrt(
            (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) *
            (1.0 - alpha_bar_t / alpha_bar_prev)
        )

        noise = torch.randn_like(x_t)

        # 方向项（从 x0 指向 x_t）
        dir_xt = torch.sqrt(
            torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0)
        ) * eps

        x_prev = sqrt_alpha_bar_prev * x0_pred + dir_xt + sigma_t * noise
        return x_prev

    # ======== DDIM：有 CFG 的单步更新 ========
    @torch.no_grad()
    def p_sample_ddim_cfg(self, x, t, t_prev,
                          condition_tensors, null_condition_tensors,
                          guidance_scale, eta=0.0):
        eps_cond = self._predict_eps(x, t, condition_tensors=condition_tensors)
        eps_uncond = self._predict_eps(x, t, condition_tensors=null_condition_tensors)

        w = guidance_scale
        eps_cfg = (1.0 + w) * eps_cond - w * eps_uncond

        x_prev = self.ddim_step(x, t, t_prev, eps_cfg, eta=eta)
        return x_prev

    # ======== DDIM：无 CFG 的单步更新 ========
    def p_sample_ddim(self, x, t, t_prev, condition_tensors=None, eta=0.0):
        eps = self._predict_eps(x, t, condition_tensors=condition_tensors)
        x_prev = self.ddim_step(x, t, t_prev, eps, eta=eta)
        return x_prev

    # ======== 统一的采样循环：支持 DDPM / DDIM + CFG ========
    def p_sample_loop(self, shape, condition_tensors=None,
                      guidance_scale=0.0, mode='ddpm', eta=0.0,ddim_steps=None):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)

        # 构造“空条件”用于 CFG（你是 [-1,1] 归一化，所以用 -1 作为 null condition）
        null_condition = None
        if self.with_condition and condition_tensors is not None and guidance_scale > 0.0:
            null_condition = -1 * torch.ones_like(condition_tensors)

        # ========= 构造时间步序列 =========
        if mode == 'ddim':
            # DDIM 可以选择子时间步
            if (ddim_steps is None) or (ddim_steps >= self.num_timesteps):
                # 用满全程步数
                time_seq = torch.arange(self.num_timesteps - 1, -1, -1, device=device, dtype=torch.long)
            else:
                # 只选 ddim_steps 个时间点（从 0~T-1 均匀采样），再反转
                step_indices = np.linspace(0, self.num_timesteps - 1, ddim_steps, dtype=int)
                time_seq = torch.from_numpy(step_indices).to(device=device, dtype=torch.long)
                time_seq = time_seq.flip(0)  # 从大到小
        else:
            # DDPM 按传统走满所有步
            time_seq = torch.arange(self.num_timesteps - 1, -1, -1, device=device, dtype=torch.long)

        # ========= 主采样循环 =========
        for idx, i in enumerate(time_seq):
            t = torch.full((b,), int(i.item()), device=device, dtype=torch.long)

            if mode == 'ddim':
                # DDIM 需要 t_prev（下一个时间步）
                if idx == len(time_seq) - 1:
                    # 最后一跳直接认为下一个是 0
                    t_prev = torch.zeros_like(t)
                else:
                    t_prev = torch.full(
                        (b,),
                        int(time_seq[idx + 1].item()),
                        device=device,
                        dtype=torch.long
                    )

                # ========== DDIM 分支 ==========
                if self.with_condition and condition_tensors is not None and guidance_scale > 0.0:
                    img = self.p_sample_ddim_cfg(
                        img, t, t_prev,
                        condition_tensors=condition_tensors,
                        null_condition_tensors=null_condition,
                        guidance_scale=guidance_scale,
                        eta=eta
                    )
                elif self.with_condition and condition_tensors is not None:
                    img = self.p_sample_ddim(
                        img, t, t_prev,
                        condition_tensors=condition_tensors,
                        eta=eta
                    )
                else:
                    img = self.p_sample_ddim(
                        img, t, t_prev,
                        condition_tensors=None,
                        eta=eta
                    )

            else:
                # ========== DDPM 分支 ==========
                if self.with_condition and condition_tensors is not None and guidance_scale > 0.0:
                    img = self.p_sample_cfg(
                        img, t,
                        condition_tensors=condition_tensors,
                        null_condition_tensors=null_condition,
                        guidance_scale=guidance_scale
                    )
                elif self.with_condition and condition_tensors is not None:
                    img = self.p_sample(
                        img, t,
                        condition_tensors=condition_tensors
                    )
                else:
                    img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def p_sample_single(self, shape, input_img, timestep, condition_tensors=None,
                        guidance_scale=0.0, mode='ddpm', eta=0.0):
        """
        单步可视化 / debug 用，如果要支持 CFG & DDIM，也可以类似改造。
        这里先保持简单：仍然按 DDPM 无 CFG 逻辑。
        """
        device = self.betas.device
        b = shape[0]
        if self.with_condition:
            t = torch.full((b,), timestep, device=device, dtype=torch.long)
            img = self.p_sample(input_img, t, condition_tensors=condition_tensors)
        else:
            img = self.p_sample(input_img, torch.full((b,), timestep, device=device, dtype=torch.long))
        return img

    # ======== 对外接口：sample / sample_same_shape / sample_from_z ========
    @torch.no_grad()
    def sample(self, batch_size=2, condition_tensors=None,
               guidance_scale=0.0, mode='ddpm', eta=0.0):
        image_size = self.image_size
        depth_size = self.depth_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, depth_size, image_size, image_size),
            condition_tensors=condition_tensors,
            guidance_scale=guidance_scale,
            mode=mode,
            eta=eta
        )

    @torch.no_grad()
    def sample_same_shape(self, batch_size=2, condition_tensors=None,
                          guidance_scale=0.0, mode='ddpm', eta=0.0):
        batch_size, _, depth_size, height_size, width_size = condition_tensors.shape
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, depth_size, height_size, width_size),
            condition_tensors=condition_tensors,
            guidance_scale=guidance_scale,
            mode=mode,
            eta=eta
        )


    def sample_from_z(self, img, batch_size=1, condition_tensors=None,
                      device='cpu', mode='ddpm', guidance_scale=0.0, eta=0.0,ddim_steps=5):
        """
        从给定初始 z 开始采样（而不是从 N(0, I)）。
        """
        img = img.to(device)
        if condition_tensors is not None:
            condition_tensors = condition_tensors.to(device)

        null_condition = None
        if self.with_condition and condition_tensors is not None and guidance_scale > 0.0:
            null_condition =-1*torch.ones_like(condition_tensors).to(device)
        if mode == 'ddim':
            # 如果未指定子步数，则恢复到全步数
            if (ddim_steps is None) or (ddim_steps >= self.num_timesteps):
                time_steps = torch.arange(self.num_timesteps - 1, -1, -1,
                                        device=device, dtype=torch.long)
            else:
                # 均匀采样 ddim_steps 个 index（0~T-1），再反转
                step_indices = np.linspace(0, self.num_timesteps - 1, ddim_steps, dtype=int)
                time_steps = torch.from_numpy(step_indices).to(device=device, dtype=torch.long)
                time_steps = time_steps.flip(0)
        else:
            # DDPM：走完整 T 步
            time_steps = torch.arange(self.num_timesteps - 1, -1, -1,
                                    device=device, dtype=torch.long)
        for idx, i in enumerate(time_steps):
            t = torch.full((batch_size,), int(i.item()), device=device, dtype=torch.long)

            if mode == 'ddim':
                if idx == len(time_steps) - 1:
                    t_prev = torch.zeros_like(t)
                else:
                    t_prev = torch.full(
                        (batch_size,),
                        int(time_steps[idx + 1].item()),
                        device=device,
                        dtype=torch.long
                    )

                if self.with_condition and condition_tensors is not None and guidance_scale > 0.0:
                    img = self.p_sample_ddim_cfg(
                        img, t, t_prev,
                        condition_tensors=condition_tensors,
                        null_condition_tensors=null_condition,
                        guidance_scale=guidance_scale,
                        eta=eta
                    )
                elif self.with_condition and condition_tensors is not None:
                    img = self.p_sample_ddim(
                        img, t, t_prev,
                        condition_tensors=condition_tensors,
                        eta=eta
                    )
                else:
                    img = self.p_sample_ddim(
                        img, t, t_prev,
                        condition_tensors=None,
                        eta=eta
                    )
            else:
                if self.with_condition and condition_tensors is not None and guidance_scale > 0.0:
                    img = self.p_sample_cfg(
                        img, t,
                        condition_tensors=condition_tensors,
                        null_condition_tensors=null_condition,
                        guidance_scale=guidance_scale
                    )
                elif self.with_condition and condition_tensors is not None:
                    img = self.p_sample(img, t, condition_tensors=condition_tensors)
                else:
                    img = self.p_sample(img, t)

        return img


    # ======== q_sample & loss 部分（基本没动）========
    def q_sample(self, x_start, t, noise=None, c=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_hat = 0.
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise + x_hat
        )

    def p_losses(self, x_start, t, condition_tensors=None, noise=None):
        b, c, h, w, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.with_condition and condition_tensors is not None:
            x_recon = self.denoise_fn(torch.cat([x_noisy, condition_tensors], 1), t)
        else:
            x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_recon, noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, condition_tensors=None, *args, **kwargs):
        b, c, d, h, w, device, img_size, depth_size = *x.shape, x.device, self.image_size, self.depth_size

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if condition_tensors is not None and condition_tensors.shape[1] == 1:
            condition_tensors = condition_tensors.repeat(1, 2, 1, 1, 1)
        return self.p_losses(x, t, condition_tensors=condition_tensors, *args, **kwargs)
class DPCPDiffusion(GaussianDiffusion):
    """
    在 GaussianDiffusion 基础上加入 diffusion purification 过程：
    对于给定的 T1、k、K：
      1) 计算 tk：在 [T1, 0] 上线性递减采样的第 k 个时间步；
      2) 将输入 img 视为 x_0，加噪到 tk 得到 x_{tk}；
      3) 从 tk 开始，逐步去噪到 0（tk, tk-1, ..., 0）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_tk(self,T1, k: int, K: int) -> int:
        """
        根据最大时间步 T1、索引 k 以及总步数 K 计算 tk。

        - 若 T1 是 int：tk 在线性区间 [T1, 0] 上递减
        k=0 -> tk=T1, k=K-1 -> tk=0

        - 若 T1 是 list/tuple：tk 在线性区间 [T1[0], T1[1]] 上递减
        k=0 -> tk=T1[0], k=K-1 -> tk=T1[1]
        """
        assert K >= 1, "K 必须 >= 1"
        assert 0 <= k < K, "k 必须在 [0, K-1] 范围内"

        # 解析区间端点：t_start -> t_end
        if isinstance(T1, (list, tuple)):
            assert len(T1) >= 2, "当 T1 为列表/元组时，至少需要两个元素：T1[0], T1[1]"
            t_start = int(T1[0])
            t_end   = int(T1[1])
            t_min, t_max = (t_start, t_end) if t_start <= t_end else (t_end, t_start)
        else:
            t_start = int(T1)
            t_end   = 0
            t_min, t_max = (0, t_start) if 0 <= t_start else (t_start, 0)

        if K == 1:
            return t_start

        # 线性插值：tk = t_start + (t_end - t_start) * (k/(K-1))
        ratio = k / (K - 1)
        tk = t_start + (t_end - t_start) * ratio

        tk = int(round(tk))
        tk = max(t_min, min(t_max, tk))
        return tk
    @torch.no_grad()
    def tweedie_denoise_one_step(self, x_t, t, condition_tensors=None, clip_denoised=True):
        model_mean, posterior_variance, posterior_log_variance = self.p_mean_variance(
            x=x_t,
            t=t,
            c=condition_tensors,
            clip_denoised=clip_denoised,
        )
        return model_mean
    @torch.no_grad()
    def diffusion_purification(
        self,
        img: torch.Tensor,          # [B, C, D, H, W]，视为当前 x_0 的估计
        T1: int,                    # 最大扩散步（噪声强度上限）
        k: int,                     # 当前使用的索引 k
        K: int,                     # 总共有 K 个线性采样的 tk
        condition_tensors: torch.Tensor = None,
        clip_denoised: bool = True,
        add_noise_weight=0,
        use_tweedie_one_step: bool = False, 
    ):
        """
        单次 diffusion purification 过程：

        1) 根据 (T1, k, K) 计算 tk；
        2) 把 img 视为 x_0，加噪到 tk（得到 x_{tk}）；
        3) 从 tk 开始，按 DDPM 的方式一步步去噪到 t=0。

        返回：
            purified_img: 经过这一轮 purification 后的结果（接近 t=0）
        """
        device = img.device
        B = img.shape[0]

        # 1) 计算 tk
        
        tk = self.compute_tk(T1, k, K) 
        assert 0 <= tk < self.num_timesteps, f"tk={tk} 不在合法时间步范围内"

        # 2) 将 img 作为 x_0，加噪到 tk：获得 x_{tk}
        t_batch = torch.full((B,), tk, device=device, dtype=torch.long)
        x_t = self.q_sample(x_start=img, t=t_batch)
        
        #from DMpipe import DEBUG_test_save
        if add_noise_weight!=0:
            x_t_from_noise= torch.randn_like(img)
        # 使用 p_sample 从 T_max 到 t_k+1 逐步去噪
            for t in range(self.num_timesteps-1, tk, -1):  # 从 T_max 到 t_k+1
                t_cur = torch.full((B,), t, device=device, dtype=torch.long)
                x_t_from_noise = self.p_sample(
                    x_t_from_noise, 
                    t_cur, 
                    condition_tensors=condition_tensors,
                    clip_denoised=clip_denoised
                )
            x_t_from_noise[condition_tensors[:,:1,...]<0]==0
            x = add_noise_weight * x_t_from_noise + (1 - add_noise_weight) * x_t
        # 3) 从 tk 开始逐步去噪到 0
        else:
            x = x_t

        if use_tweedie_one_step:
            x0_hat = self.tweedie_denoise_one_step(
                x_t=img,
                t=t_batch,
                condition_tensors=condition_tensors,
                clip_denoised=clip_denoised,
            )
            return x0_hat
        progress = tqdm(
            reversed(range(0, tk + 1)),
            desc=f"Purification downsampling (tk={tk})",
            total=tk + 1
        )
        for t in progress:  # t = tk, tk-1, ..., 0
            t_cur = torch.full((B,), t, device=device, dtype=torch.long)
            x = self.p_sample(
                x,
                t_cur,
                condition_tensors=condition_tensors,
                clip_denoised=clip_denoised,
            )

        return x

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        eval_dataset=None,
        ema_decay = 0.995,
        image_size = 128,
        depth_size = 128,
        train_batch_size = 2,
        train_lr = 2e-6,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        with_condition = False,
        cond_drop_prob =0.2,
        guidance_scale =7.5,
        use_cfg=False
        ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.eds= eval_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, num_workers=4, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size
        self.with_condition = with_condition
        self.cond_drop_prob = cond_drop_prob   # classifier-free guidance 中的“丢条件概率”，一般 0.1~0.2
        self.guidance_scale = guidance_scale 

        self.step = 0
        self.use_cfg=use_cfg
        # assert not fp16 or fp16 and  PEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'
        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.log_dir = self.create_log_dir()
        self.writer = SummaryWriter(log_dir=self.log_dir)#"./logs")
        self.reset_parameters()

    def create_log_dir(self):
        now = datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")
        log_dir = os.path.join("./logs", now)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        while self.step < self.train_num_steps:
            accumulated_loss = []
            for i in range(self.gradient_accumulate_every):
                if self.with_condition:
                    data = next(self.dl)
                    input_tensors = data['input'].cuda()
                    target_tensors = data['target'].cuda()
                    if self.use_cfg and random.random() < self.cond_drop_prob:
                        # 构造一个“空条件”，shape 完全一致
                        null_condition =-1*torch.ones_like(input_tensors)
                        loss = self.model(
                            target_tensors,
                            condition_tensors=null_condition
                        )
                    else:
                        loss = self.model(target_tensors, condition_tensors=input_tensors)
                else:
                    data = next(self.dl).cuda()
                    loss = self.model(data)
                loss = loss.sum()/self.batch_size
                print(f'{self.step}: {loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)
                accumulated_loss.append(loss.item())

            # Record here
            average_loss = np.mean(accumulated_loss)
            end_time = time.time()
            self.writer.add_scalar("training_loss", average_loss, self.step)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0 or DEBUG:
                milestone = self.step // self.save_and_sample_every
                batches = num_to_groups(1, self.batch_size)
                cond_batches = [
                        self.ds.sample_conditions(batch_size=n).cuda()
                        for n in batches
                    ]
                if self.with_condition:
                    all_images_list = []
                    for n, cond in zip(batches, cond_batches):
                        out = self.ema_model.sample_same_shape(
                            batch_size=n,
                            condition_tensors=cond
                        )
                        all_images_list.append(out)
                    all_images = torch.cat(all_images_list, dim=0)
                    if self.use_cfg:
                        all_images_cfg_list = []
                        for n, cond in zip(batches, cond_batches):
                            out_cfg = self.ema_model.sample_same_shape(
                                batch_size=n,
                                condition_tensors=cond,
                                guidance_scale=getattr(self, "guidance_scale", 0.0),  # 自己在 __init__ 里设好
                                mode=getattr(self, "cfg_mode", "ddpm"),               # 或 'ddpm'
                                eta=getattr(self, "cfg_eta", 0.0)                     # DDIM 时可用
                            )
                            all_images_cfg_list.append(out_cfg)
                        all_images_cfg = torch.cat(all_images_cfg_list, dim=0)
                else:
                    all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                    all_images = torch.cat(all_images_list, dim=0)

                all_images = all_images.squeeze()
                sampleImage = all_images.cpu().numpy()
                nifti_img = nib.Nifti1Image(sampleImage, affine=np.eye(4))
                sampledir=self.results_folder/'sample'
                os.makedirs(sampledir,exist_ok=True)
                nib.save(nifti_img, str(sampledir / f'sample-{milestone}.nii.gz'))
                self.save(milestone)
                if self.with_condition and self.use_cfg:
                    all_images_cfg = all_images_cfg.squeeze()
                    sampleImage_cfg = all_images_cfg.cpu().numpy()
                    nifti_img_cfg = nib.Nifti1Image(sampleImage_cfg, affine=np.eye(4))
                    cfgdir = self.results_folder / 'sample_cfg'
                    os.makedirs(cfgdir, exist_ok=True)
                    nib.save(nifti_img_cfg, str(cfgdir / f'sample-cfg-{milestone}.nii.gz'))
                if self.with_condition and self.eds is not None:
                    images=self.eds.sample_all_evaluate()
                    for i,image in enumerate(images):
                        image=image.cuda()
                        out= self.ema_model.sample_same_shape(batch_size=1,condition_tensors=image.unsqueeze(0).repeat(1,2,1,1,1))
                        nifti_img = nib.Nifti1Image(out.squeeze().cpu().numpy(), affine=np.eye(4))
                        evaldir=self.results_folder/'evalute'
                        os.makedirs(evaldir, exist_ok=True)
                        nib.save(nifti_img, str(evaldir / f'evalute-{milestone}-{i}.nii.gz'))
                    if self.use_cfg:
                        out_cfg = self.ema_model.sample_same_shape(
                            batch_size=1,
                            condition_tensors=cond,
                            guidance_scale=getattr(self, "guidance_scale", 0.0),
                            mode=getattr(self, "cfg_mode", "ddpm"),
                            eta=getattr(self, "cfg_eta", 0.0),
                        )
                        nifti_img_cfg = nib.Nifti1Image(out_cfg.squeeze().cpu().numpy(), affine=np.eye(4))
                        evaldir_cfg = self.results_folder / 'evalute_cfg'
                        os.makedirs(evaldir_cfg, exist_ok=True)
                        nib.save(nifti_img_cfg, str(evaldir_cfg / f'evalute-cfg-{milestone}-{i}.nii.gz'))
            self.step += 1

        print('training completed')
        end_time = time.time()
        execution_time = (end_time - start_time)/3600
        self.writer.add_hparams(
            {
                "lr": self.train_lr,
                "batchsize": self.train_batch_size,
                "image_size":self.image_size,
                "depth_size":self.depth_size,
                "execution_time (hour)":execution_time
            },
            {"last_loss":average_loss}
        )
        self.writer.close()
    def sample_as_train(self):
        
        images=self.ds.sample_all_evaluate()
        traindir=self.results_folder/'train'
        os.makedirs(traindir, exist_ok=True)
        for i,image in enumerate(images):
            if i%10!=0:
                continue
            image=image.cuda()
            name='-'.join(self.ds.combined_pairs[i][0].split('/')[5:7])
            DEBUG_test_save(image,f'train-input-{name}.nii.gz',save_dir=str(traindir))
            out= self.ema_model.sample_same_shape(batch_size=1,condition_tensors=image.unsqueeze(0).repeat(1,2,1,1,1),debug_save_dir=str(traindir/f'{name}'))
            DEBUG_test_save(out,f'train-ouput-{name}.nii.gz',save_dir=str(traindir))
        if self.eds is not None:
            images=self.eds.sample_all_evaluate()
            evaldir=self.results_folder/'evalute'
            os.makedirs(evaldir, exist_ok=True)
            for i,image in enumerate(images):
                image=image.cuda()
                name='-'.join(self.eds.combined_pairs[i][0].split('/')[5:7])
                DEBUG_test_save(image,f'eval-input-{name}.nii.gz',save_dir=str(evaldir))
                out= self.ema_model.sample_same_shape(batch_size=1,condition_tensors=image.unsqueeze(0).repeat(1,2,1,1,1),debug_save_dir=str(evaldir/f'{name}'))
                DEBUG_test_save(out,f'eval-ouput-{name}.nii.gz',save_dir=str(evaldir))