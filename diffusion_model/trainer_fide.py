from .trainer import GaussianDiffusion,default
from nesvor.image import Volume
import torch
from tqdm import tqdm
import os
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from nesvor import svr
import torchio as tio
try:
    from nesvor.inr.train import fide_coarse_train,fide_refine_train
except:
    pass
from nesvor.cli.commands import _register,_sample_inr
from argparse import Namespace
from nesvor.image import Volume, Slice,load_volume
from typing import List, Tuple
from nesvor.inr import models,data
from nesvor.transform import axisangle2mat,RigidTransform
from nesvor.slice_acquisition import slice_acquisition,slice_acquisition_adjoint
from nesvor.utils import get_PSF
def dot(x, y):
    return torch.dot(x.flatten(), y.flatten())

def CG(A, b, x0, n_iter):
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    while True:
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p  # alpha ~ 0.1 - 1
        i += 1
        if i == n_iter:
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new

def PSFreconstruction(transforms, slices, slices_mask, vol_mask, params):
    return slice_acquisition_adjoint(transforms, params['psf'], slices, slices_mask, vol_mask, params['volume_shape'], params['res_s'] / params['res_r'], params['interp_psf'], True)
    
class SRR(nn.Module):
    def __init__(self, n_iter=10, use_CG=False, alpha=0.5, beta=0.02, delta=0.1):
        super().__init__()
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta * delta * delta
        self.delta = delta
        self.use_CG = use_CG

    def forward(self, theta, slices, volume, params, p=None, mu=0, z=None, vol_mask=None, slices_mask=None):
        if len(theta.shape) == 2:
            transforms = axisangle2mat(theta)
        else:
            transforms = theta

        A = lambda x:self.A(transforms, x, vol_mask, slices_mask, params)
        At = lambda x:self.At(transforms, x, slices_mask, vol_mask, params)
        AtA = lambda x:self.AtA(transforms, x, vol_mask, slices_mask, p, params, mu, z)

        x = volume
        y = slices
        
        if self.use_CG:
            b = At(y * p if p is not None else y)
            if mu and z is not None:
                b = b + mu*z
            x = CG(AtA, b, volume, self.n_iter) 
        else:
            for _ in range(self.n_iter):
                err = A(x) - y
                if p is not None:
                    err = p * err
                g = At(err)
                if self.beta:
                    dR = self.dR(x, self.delta)
                    g.add_(dR, alpha=self.beta)
                x.add_(g, alpha=-self.alpha)
        return F.relu(x, True)
    

    def A(self, transforms, x, vol_mask, slices_mask, params):
        return slice_acquisition(transforms, x, vol_mask, slices_mask, params['psf'], params['slice_shape'], params['res_s'] / params['res_r'], False, params['interp_psf'])

    def At(self, transforms, x, slices_mask, vol_mask, params):
        return slice_acquisition_adjoint(transforms, params['psf'], x, slices_mask, vol_mask, params['volume_shape'], params['res_s'] / params['res_r'], params['interp_psf'], False)

    def AtA(self, transforms, x, vol_mask, slices_mask, p, params, mu, z):
        slices = self.A(transforms, x, vol_mask, slices_mask, params)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, slices_mask, vol_mask, params)
        if mu and z is not None:
            vol = vol + mu * x
        return vol

    def dR(self, v, delta):
        g = torch.zeros_like(v)
        D, H, W = v.shape[-3:]
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    v0 = v[:, :, 1:D-1, 1:H-1, 1:W-1]
                    v1 = v[:, :, 1+dz:D-1+dz, 1+dy:H-1+dy, 1+dx:W-1+dx]
                    dv = v0 - v1
                    dv_ = dv * (1 / (dx*dx + dy*dy + dz*dz) / (delta*delta))
                    g[:, :, 1:D-1, 1:H-1, 1:W-1] += dv_ / torch.sqrt(1 + dv * dv_)
        return g
def crop_nonzero_tensor(tensor):
    # 找到不为0的元素的索引
    non_zero_indices = torch.nonzero(tensor)
    
    # 计算每个维度的最小和最大索引
    min_indices = non_zero_indices.min(dim=0).values
    max_indices = non_zero_indices.max(dim=0).values
    
    # 根据每个维度的最小和最大索引裁剪张量
    slices = [slice(min_indices[i].item(), max_indices[i].item() + 1) for i in range(tensor.dim())]
    cropped_tensor = tensor[slices]
    
    return cropped_tensor, non_zero_indices
def resize_target(img: torch.Tensor,prefix_shape:tuple,nz:torch.Tensor)-> torch.Tensor:

    min_indices = nz.min(dim=0).values
    max_indices = nz.max(dim=0).values
    cropped_shape=[max_indices[i].item() + 1-min_indices[i].item() for i in range(1,len(min_indices))]
    img_croped= F.interpolate(img, size=cropped_shape, mode='trilinear', align_corners=False)
    img_ori=torch.zeros(prefix_shape)
    img_ori[min_indices[0].item():max_indices[0].item()+1,min_indices[1].item():max_indices[1].item()+1,min_indices[2].item():max_indices[2].item()+1,min_indices[3].item():max_indices[3].item()+1 ]=img_croped
    return img_ori
def resize_img_mask(img: torch.Tensor,batch_size,shape) -> torch.Tensor:

    if img.dim() == 3:
        img = img.unsqueeze(0)  # 从 (H, W, D) -> (C, H, W, D)
        prefix_shape=img.shape
    elif img.dim() == 4:
        if img.shape[-1]==2 or img.shape[-1]==1:
            img=img.permute(3,0,1,2)
            if img.shape[0]==2:
                img=img[0][None,...]
        prefix_shape=img.shape  # 已经是 (C, H, W, D)
    elif img.dim() == 5:
        img=img.squeeze(0)
        prefix_shape=img.shape
    else:
        raise ValueError(f"不支持的张量形状 {img.shape}，期望3D或4D张量。")
    device=img.device
    img=img.cpu()
    img,nz=crop_nonzero_tensor(img)
    scalar_image = tio.ScalarImage(tensor=img)
    
    resize_transform = tio.Resize(tuple(shape))

    resized_image = resize_transform(scalar_image)

    resized_tensor = resized_image.tensor.to(device)

    if resized_tensor.dim() == 3:
        resized_tensor = resized_tensor.unsqueeze(0) # 从 (H_new, W_new, D_new) -> (batch_size,1,H_new, W_new, D_new)
    resized_tensor= resized_tensor.unsqueeze(0).repeat(batch_size,1,1,1,1)
    return resized_tensor,prefix_shape,nz
def standardize_and_contrast_stretch(image):
    # 均值-方差标准化
    mean = image.mean()
    std = image.std()
    standardized_image = (image - mean) / std
    
    # 对比度拉伸：将标准化的图像拉伸到 [0, 1]
    min_val = standardized_image.min()
    max_val = standardized_image.max()
    contrast_stretched_image = (standardized_image - min_val) / (max_val - min_val)
    
    return contrast_stretched_image
def normalize_and_standardize(image):
    # 归一化到 [-1, 1]
    min_val = image.min()
    max_val = image.max()
    normalized_image = 2 * (image - min_val) / (max_val - min_val) - 1
    
    # 均值-方差标准化
    mean = normalized_image.mean()
    std = normalized_image.std()
    standardized_image = (normalized_image - mean) / std
    
    return standardized_image
def save_path(img,path):
    nifti_img = nib.Nifti1Image(img.squeeze().cpu().numpy(), affine =np.eye(4))
    nib.save(nifti_img, path)
class FidePipe(object):
    def __init__(
        self, args: Namespace, bounding_box: torch.Tensor=None, spatial_scaling: float = 1.0,mode='svr'
    ) -> None:
        super().__init__()
        self.args = args
        self.mode=mode
        self._n_train=None
        self.srr=SRR(2)
    @property
    def n_train(self):
        return self._n_train
    @n_train.setter
    def n_train(self,ntrain):
        self._n_train=ntrain
    def __call__(self, x: Volume,slices:List[Slice],mask=None,USE_age=0) -> torch.Tensor:
        """
        inputs:
        x:volume from diffusion model
        slices:acquired slices
        
        outputs:
        x: fidelity volume
        """
        #preprocess
        sum=torch.tensor(0,device=slices[0].image.device,dtype=torch.float32)
        print(slices[0].image.max())
        for slice in slices:
            sum+=slice.v_masked.mean()
        slice_mean=sum/len(slices)
        x.rescale(intensity_mean=slice_mean.cpu())
        

        if self.mode=='nesvor':
            #x=x.image.to(self.args.device)
            #model_save_path = "/home/lvyao/git/med-ddpm/results/test_inr/nesvor_model.pth"
            model = self.init_volume(x,self.args)
            #torch.save(model.state_dict(), model_save_path)
            # refine volume to slices
            output_volume=self.refine_volume_nesvor(model, slices,mask=mask,USE_age=USE_age)
            #
            return output_volume
        elif self.mode=='svr':
            output_volume=self.refine_volume_svr(x,slices)
            return output_volume
        elif self.mode=='cg':
            theta=RigidTransform.cat([s.transformation for s in slices])
            res_s=slices[0].resolution_x
            res_r=self.args.output_resolution
            s_thick=slices[0].resolution_z
            psf = get_PSF(
                res_ratio=(res_s / res_r, res_s / res_r, s_thick / res_r),
                device=x.device,
            )
            params = {
            "psf": psf,
            "slice_shape": slices[0].img.shape,
            "interp_psf": False,
            "res_s": res_s,
            "res_r": res_r,
            "s_thick": s_thick,
            "volume_shape": x.shape,
        }
            output_volume=self.srr(theta, slices, x, params)
        else:
            raise NotImplementedError
    

    def init_volume(self, x: torch.Tensor,args:Namespace) -> torch.Tensor:
        model=fide_coarse_train(x,args,ntrain=self._n_train//2)
        return model
    def refine_volume_VFM(self,volume_img):
        feature_map = self.encoder(volume_img)

    def refine_volume_nesvor(self, model:models.NeSVoR, slices: List[Slice],mask,USE_age) -> Tuple[torch.Tensor, torch.Tensor]:
        inr,slices_transform,xyz_mask =fide_refine_train(slices=slices,args=self.args,model_trained=model,ntrain=self._n_train,mask_atlas=mask if USE_age else None)
        if not USE_age:
            xyz_mask=mask
        output_volume, simulated_slices = _sample_inr(
            self.args,
            inr,
            xyz_mask,
            slices_transform,
            getattr(self.args, "output_volume", None) is not None,
            getattr(self.args, "simulated_slices", None) is not None,
        )
        return output_volume
    def refine_volume_nn(self, model:models.NeSVoR, slices: List[Slice]) -> Volume:
        
        inr,slices_transform,xyz_mask =fide_refine_train(slices=slices,args=self.args,model=model)

    def refine_volume_svr(self, x: torch.Tensor, slices: List[Slice]) -> Tuple[torch.Tensor, torch.Tensor]:
        #slices = _register(self.args,slices)
        output_volume, output_slices, simulated_slices = svr.slice_to_volume_reconstruction_with_atlas(
            x,slices=slices, **vars(self.args)
        )
        return output_volume
class FideGaussianDiffusion(GaussianDiffusion):
    @property
    def prefix_shape_nz_index(self):
        return self._prefix_shape_nz_index
    @prefix_shape_nz_index.setter
    def prefix_shape_nz_index(self,prefix_shape_nz):
        self._prefix_shape_nz_index=prefix_shape_nz
    @property
    def slices(self):
        return self._slices
    @slices.setter
    def slices(self,slices):
        self._slices=slices
    @property
    def svr_args(self):
        return self._svr_args
    @svr_args.setter
    def svr_args(self,args):
        self._svr_args=args
    @property
    def condition(self):
        return self._condition_mask
    @condition.setter
    def condition_transformation(self,mask):
        self._condition_mask=mask
    #TODO: ablation:1.direct inr and diffusion 2. without add noise 3. choise of RI 
    def fide_sample_process(self,shape,img, i,condition_tensors = None):
        device = self.betas.device
        b = shape[0]
        batch_size=img.shape[0]
        t = torch.full((b,), i, device=device, dtype=torch.long)
        fidepipe=FidePipe(self.svr_args,mode='nesvor' if self.svr_args.ablation!='ablation_mibr' else 'svr')
        fidepipe.n_train=2000
        result_folder=os.path.dirname(self._svr_args.output_volume)
        img=(img+1)/2
        #img=standardize_and_contrast_stretch(img)
        # fidepipe.n_train-=200
        img=resize_target(img,*self.prefix_shape_nz_index).to(device)
        volume=Volume(image=img.squeeze(),mask=(resize_target((condition_tensors[:,:1,...]+1)/2,*self.prefix_shape_nz_index).squeeze()).bool().to(device),
                        transformation=self.condition.transformation,resolution_x=0.5,resolution_y=0.5,resolution_z=0.5)
        if i==0 :               
            return volume
        # output_volume, output_slices, simulated_slices = svr.slice_to_volume_reconstruction_with_atlas(
        #     volume,slices=self.slices, **vars(self.svr_args)
        # )
        if True:
            volume.save(os.path.join(result_folder,f'resample-diff_process_{i}.nii.gz'))
        output_volume=fidepipe(volume,slices=self.slices,mask=self.condition,USE_age=self.svr_args.age)
        #img=normalize_and_standardize(output_volume.image)
        if True:
            output_volume.save(os.path.join(result_folder,f'resample-fide_{i}.nii.gz'))
        img,_,_=resize_img_mask(output_volume.image,batch_size,(self.image_size,self.image_size,self.depth_size))
        img=img*2-1
        save_path(img,os.path.join(result_folder,f'resample-vt_{i}.nii.gz'))
        if self.svr_args.ablation=='ablation_addnoise' or self.svr_args.ablation=='ablation_mibr':
            return img
        noise = default(None, lambda: torch.randn_like(img))
        img = self.q_sample(x_start=img, t=t, noise=noise)
        if True:
            volume=Volume(image=img.squeeze(),
                        transformation=self.condition.transformation,resolution_x=0.5,resolution_y=0.5,resolution_z=0.5)
            volume.save(os.path.join(result_folder,f'resample-diff_refine_{i}.nii.gz'))
        return img
    def p_sample_loop(self, shape, condition_tensors = None):  
            device = self.betas.device
            b = shape[0]
            img = torch.randn(shape, device=device)
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                if self.with_condition:        
                    with torch.no_grad():
                        t = torch.full((b,), i, device=device, dtype=torch.long)
                        img = self.p_sample(img, t, condition_tensors=condition_tensors)
                    if i%90==0 and not ( self.svr_args.ablation=='ablation_nesvor'):
                        img=self.fide_sample_process(shape,img, i,condition_tensors = condition_tensors)
                

                else:
                    img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
            return img