#TODO: #1register #2single-mask #experiment
from argparse import Namespace
import argparse
import torch
import yaml
import os
import inspect
import nibabel as nib
import numpy as np
from tqdm import tqdm
import time
import json
import torch.nn.functional as F
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.unet import create_model
from dataset import Subject  
import dataset
from utils.image_process import upsample_3d,downsample_3d
from utils.segment import _segment_stack
from utils.registration import _register,forward_process,Volume, _register_pe,RigidTransform,forward_reconstruction,svr_reconstruction,cg_reconstruction
from utils.reconstruction import INR_super_resolution,optimization_inr
from diffusion_model.trainer import GaussianDiffusion,CFGDiffusion,DPCPDiffusion
DEBUG=True
def DEBUG_test_save(img,name,save_dir='/home/lvyao/git/med-ddpm/results/DM_results/figures'):
    if not DEBUG:
        return
    os.makedirs(save_dir,exist_ok=True)
    out=img.cpu().detach().numpy()
    nifti_img = nib.Nifti1Image(out.squeeze(), affine=np.eye(4))
    nib.save(nifti_img,os.path.join(save_dir,name)+'.nii.gz')
input_transform = Compose([
    Lambda(lambda t: t.float()),
    Lambda(lambda t: (t-t.min())/(t.max()-t.min())),
    Lambda(lambda t: t.transpose(4,2)),
    Lambda(lambda t: (t * 2) - 1),
])

reverse_transform= Compose([
    Lambda(lambda t: (t + 1) /2),
    Lambda(lambda t: t.transpose(4,2)),
])

def save(img,args,target_shape,name):
    save_dir=args.save_dir
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    res=args.resolution_volume
    affine = np.eye(4, dtype=np.float32)
    # 设置分辨率
    affine[0, 0] = res   # X轴（宽度）
    affine[1, 1] = res   # Y轴（高度）
    affine[2, 2] = res   # Z轴（深度）
    # 调整原点到图像中心
    affine[0, 3] = - (target_shape[2] - 1) * res / 2  # W方向中心
    affine[1, 3] = - (target_shape[1] - 1) * res / 2  # H方向中心
    affine[2, 3] = - (target_shape[0] - 1) * res / 2  # D方向中心
    img_np = img.detach().cpu().numpy()
    img_nii = nib.Nifti1Image(img_np, affine)
    #
    save_path=os.path.join(args.save_dir,name)+'.nii.gz'
    nib.save(img_nii, save_path)

def pad_to_even(tensor: torch.Tensor) -> torch.Tensor:

    
    dim3, dim2, dim1 = tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]
    

    pad1 = 1 if dim1 %2== 1 else 0  
    pad2 = 1 if dim2 %2== 1 else 0  
    pad3 = 1 if dim3 %2== 1 else 0  
    
    padding = (0, pad1, 0, pad2, 0, pad3)
    
    # 执行padding（constant模式默认填充0）
    padded_tensor = F.pad(tensor, padding, mode='constant', value=0)
    
    return padded_tensor
def pad_to_multiple_of_4(tensor: torch.Tensor) -> torch.Tensor:
    dim3, dim2, dim1 = tensor.shape[-3], tensor.shape[-2], tensor.shape[-1]

    pad1 = (4 - dim1 % 4) % 4
    pad2 = (4 - dim2 % 4) % 4
    pad3 = (4 - dim3 % 4) % 4

    padding = (0, pad1, 0, pad2, 0, pad3)

    # constant 模式，默认补 0
    padded_tensor = F.pad(tensor, padding, mode='constant', value=0)

    return padded_tensor
class DiffusionModelManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_yaml(config_path)
        self.device=torch.device(f"cuda:{self.config.dataset.device}")
        self._initialize_dataset()
        if hasattr(self.config,'pretrained_model'):
            self._initialize_model()

    def load_yaml(self, file_path: str):
        """加载YAML配置文件并将其转换为Namespace"""
        with open(file_path, 'r') as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        
        # 将字典转换为 Namespace
        config_namespace = self.dict_to_namespace(config_dict)
        return config_namespace

    def dict_to_namespace(self, d):
        """将字典递归地转换为 Namespace"""
        if isinstance(d, dict):
            # 递归转换字典的每个元素
            ns = Namespace()
            for key, value in d.items():
                setattr(ns, key, self.dict_to_namespace(value))  # 使用递归
            return ns
        else:
            return d  # 如果是非字典元素，直接返回

    def _initialize_model(self):
        """初始化扩散模型并加载权重"""
        pretrained_config = self.config.pretrained_model
        
        # 加载模型的参数
        input_size = pretrained_config.input_size
        num_channels = pretrained_config.num_channels
        num_res_blocks = pretrained_config.num_res_blocks
        in_channels= pretrained_config.in_channels
        out_channels = pretrained_config.out_channels
        
        # 创建并初始化模型
        self.model = create_model(input_size, num_channels, num_res_blocks, 
                                  in_channels=in_channels, out_channels=out_channels).to(self.device)
        
        # 创建扩散模型
        if self.config.task.name=='DMplug':
            self.diffusion = GaussianDiffusion(
                self.model,
                image_size=input_size,
                depth_size=pretrained_config.depth_size,
                timesteps=pretrained_config.timesteps,  # number of steps
                loss_type=pretrained_config.loss_type,
                with_condition=True,
            ).to(self.device)
        elif self.config.task.name=='cfg_DMplug':
            self.diffusion = CFGDiffusion(
                self.model,
                image_size=input_size,
                depth_size=pretrained_config.depth_size,
                timesteps=pretrained_config.timesteps,  # number of steps
                loss_type=pretrained_config.loss_type,
                with_condition=True,
            ).to(self.device)
        elif self.config.task.name=='dpcp' or self.config.task.name=='fide' or self.config.task.name=='dpas' or self.config.task.name=='no_restart':
                self.diffusion = DPCPDiffusion(
                self.model,
                image_size=input_size,
                depth_size=pretrained_config.depth_size,
                timesteps=pretrained_config.timesteps,  # number of steps
                loss_type=pretrained_config.loss_type,
                with_condition=True,
            ).to(self.device)
        # 加载预训练模型的权重
        weight_path = pretrained_config.weights_path
        self._load_model_weights(weight_path)
    
    def _load_model_weights(self, weight_path: str):
        """加载预训练模型权重"""
        if not os.path.exists(weight_path):
            raise ValueError(f"Weight file {weight_path} does not exist.")
        
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.diffusion.load_state_dict(checkpoint['ema'])
        print(f"Loaded weights from {weight_path}")

    def _initialize_dataset(self):
        """加载待测试的数据集（动态根据配置初始化对应数据集类）"""
        dataset_config = self.config.dataset
        
        # 步骤1：获取dataset模块下所有的数据集类（过滤掉非数据集类如Subject）
        # 遍历dataset模块中所有成员，筛选出：是类 + 不是Subject + 模块归属为dataset
        dataset_classes = {}
        for name, obj in inspect.getmembers(dataset, inspect.isclass):
            # 过滤条件：排除Subject类，且类定义在dataset模块中（避免导入的外部类）
            if obj is not Subject and obj.__module__ == dataset.__name__:
                dataset_classes[name] = obj  # 键为类名（如"FetalStackDataset"），值为类本身
        
        # 步骤2：根据配置的name查找对应的数据集类，不存在则抛错
        dataset_class_name = dataset_config.name
        if dataset_class_name not in dataset_classes:
            raise ValueError(
                f"数据集类 {dataset_class_name} 不存在！"
                f"当前dataset模块下可用的数据集类有：{list(dataset_classes.keys())}"
            )
        
        # 步骤3：初始化对应的数据集类（参数与原代码保持一致）
        target_dataset_class = dataset_classes[dataset_class_name]
        if hasattr(dataset_config, '__dict__'):
            # 处理类对象类型的配置（如 argparse.Namespace、自定义Config类）
            config_dict = {k: v for k, v in dataset_config.__dict__.items() if not k.startswith('_')}
        else:
            # 处理字典类型的配置
            config_dict = dataset_config.copy()
        
        # 移除name字段，剩下的全部作为初始化参数
        init_kwargs = {k: v for k, v in config_dict.items() if k != "name"}

        # 步骤4：初始化数据集类（直接传递所有独有参数）
        try:
            self.dataset = target_dataset_class(**init_kwargs)
        except TypeError as e:
            # 增强错误提示，方便定位参数问题
            raise TypeError(
                f"初始化数据集类 {dataset_class_name} 失败！\n"
                f"传递的参数：{list(init_kwargs.keys())}\n"
                f"参数值：{init_kwargs}\n"
                f"错误详情：{str(e)}"
            ) from e
            
    def eval_model(self):
        """设置模型为评估模式"""
        self.model.eval()

    def run_inference(self, input_data):
        """执行模型推理"""
        with torch.no_grad():
            output = self.diffusion(input_data)
        return output

    def save_results(self, saved_images,subject):
        """保存结果到指定目录"""
        if isinstance(saved_images,list):
            for i,image in enumerate(saved_images):
                name=subject.get_name+f'_mid{i}' if i <len(saved_images)-1 else subject.get_name
                image=image.squeeze().transpose(2,0)
                save(image,self.config.DM_process,image.shape,name)
            print("Results saved.")
        else:
            image=saved_images
            image=image.squeeze().transpose(2,0)
            # 根据需求保存结果（例如保存图片或其他数据）
            save(image,self.config.DM_process,image.shape,subject.get_name)
            print("Results saved.")

    def pad(self,x, m=None,target_shape=None, multiple=16, mode='symmetric'):
        """
        x: 输入图像张量 [B, C, D, H, W]
        target_shape: 目标尺寸 [D, H, W]，如果指定，强制 pad 到这个尺寸
        multiple: pad 到的倍数（例如 16）
        mode: 'symmetric' 为对称 pad, 'fixed' 为 pad 到指定尺寸
        """
        D, H, W = x.shape[-3:]

        if mode == 'symmetric':
            def get_symmetric_pad(size):
                pad = (multiple - size % multiple) % multiple
                left = pad // 2
                right = pad - left
                return left, right

            d0, d1 = get_symmetric_pad(D)
            h0, h1 = get_symmetric_pad(H)
            w0, w1 = get_symmetric_pad(W)

            x_pad = F.pad(x, (w0, w1, h0, h1, d0, d1))
            if m is not None:
                assert m.shape==x.shape,"mask shape must equals image shape"
                m_pad = F.pad(m, (w0, w1, h0, h1, d0, d1))
                return x_pad,m_pad, (d0, d1, h0, h1, w0, w1)
            return x_pad, (d0, d1, h0, h1, w0, w1)

        elif mode == 'fixed' and target_shape is not None:
            target_D, target_H, target_W = target_shape

            # 确保目标尺寸比输入尺寸大
            if any([target_D < D, target_H < H, target_W < W]):
                raise ValueError("目标尺寸不能小于输入图像尺寸")

            # 计算每个维度需要 pad 的大小
            pad_D = target_D - D
            pad_H = target_H - H
            pad_W = target_W - W

            d0, d1 = pad_D // 2, pad_D - pad_D // 2
            h0, h1 = pad_H // 2, pad_H - pad_H // 2
            w0, w1 = pad_W // 2, pad_W - pad_W // 2

            x_pad = F.pad(x, (w0, w1, h0, h1, d0, d1))
            if m is not None:
                assert m.shape==x.shape,"mask shape must equals image shape"
                m_pad = F.pad(m, (w0, w1, h0, h1, d0, d1))
                return x_pad,m_pad, (d0, d1, h0, h1, w0, w1)
            return x_pad, (d0, d1, h0, h1, w0, w1)

        else:
            raise ValueError("mode 必须是 'symmetric' 或 'fixed'，并且 target_shape 必须提供。")
    def unpad(self,x, pads):
        d0, d1, h0, h1, w0, w1 = pads
        return x[:, :, d0 : x.shape[2] - d1,
                    h0 : x.shape[3] - h1,
                    w0 : x.shape[4] - w1]
    def run_subject_dmplug(self,subject):
        lr=0.1
        epochs=100
        target_shape=[]
        stacks=subject.input_stacks
        #stacks=_segment_stack(self.config.segment,stacks)
        
        slices,volumes = _register( self.config.registration,stacks) 
        #slices,volumes = _register_pe( self.config.registration,stacks) 
        if self.config.DM_process.mask_init=='svr_init':
            volume_svr,_,_=svr_reconstruction(slices,device=self.device)
            volumes=pad_to_even(volume_svr.image.view((1,1)+volume_svr.image.shape))
            full_mask=pad_to_even(volume_svr.mask.view((1,1)+volume_svr.image.shape))
        elif self.config.DM_process.mask_init=='template':
            # template,full_mask=subject.template()
            # full_mask=pad_to_multiple_of_4(full_mask.view((1,1)+template.shape))
            # volumes=pad_to_multiple_of_4(template.view((1,1)+template.shape))
            template,full_mask=subject.template(volumes.shape)
            full_mask=full_mask.view((1,1)+full_mask.shape)
            volumes[full_mask==False]=0
        else:
            full_mask=volumes>0.02
            volumes[full_mask==0]=0
        
        volumes_downsample,mask_downsample=downsample_3d(volumes,full_mask,scale=4)
        volumes_padded,mask_downsample,pads=self.pad(volumes_downsample,mask_downsample,target_shape)
        
        input_mask=input_transform(mask_downsample)
        input_volume=input_transform(volumes_padded)
        #z=torch.rand_like(input_mask,dtype=input_mask.dtype,device=input_mask.device,requires_grad=True)
        #init_volume=torch.tensor(input_volume.cpu().numpy(),device=input_volume.device)
        input_volume.requires_grad = True
        z=input_volume
        params_group1 = {'params': z, 'lr': lr}
        optimizer = torch.optim.Adam([params_group1]) 
        for iter in range(epochs):
            self.diffusion.eval()
            optimizer.zero_grad()
            volumes_output=self.diffusion.sample_from_z(z,batch_size = 1,condition_tensors = torch.cat([input_mask,input_mask],dim=1),device=self.device,mode='ddim',ddim_steps=5,eta=0.)
            # DEBUG_test_save(volumes_output,f'diff_out_{iter}.nii.gz')
            volumes_output=reverse_transform(volumes_output)
            volumes_unpad=self.unpad(volumes_output,pads)
            volumes_upsample,_=upsample_3d(volumes_unpad,scale=4)

            forward_volume=Volume(volumes_upsample.squeeze(),mask=full_mask.squeeze(),
                                resolution_x=self.config.registration.resolution_volume)
            #forward_volume.save('/home/lvyao/git/med-ddpm/results/DM_results/test_pipeline/foward_v.nii.gz')
            err=forward_process(forward_volume,slices,self.config.DM_process)
            loss=err.slices.abs().mean()
            loss.backward()
            optimizer.step()
            print(loss.item())
        return volumes_upsample

    def run_subject_dpcp(self,subject):
        saved_volumes=[]
        epochs=1000
        K=self.config.task.K 
        t1=self.config.task.T_1 if isinstance(self.config.task.T_1,int) else [int(s) for s in self.config.task.T_1.split(',')]
 
        try:
            if self.config.dataset.direction !='all':
                stacks=subject.input_stacks_dir(direct=self.config.dataset.direction)
            else:
                stacks=subject.input_stacks
            slices,volumes = _register( self.config.registration,stacks)
        except:
            print(f"error loading stack of {subject.get_name}")
            return None
        
        
        if self.config.DM_process.mask_init=='svr_init':
            volume_svr,_,_=svr_reconstruction(slices,device=self.device)
            volumes=pad_to_even(volume_svr.image.view((1,1)+volume_svr.image.shape))
            full_mask=pad_to_even(volume_svr.mask.view((1,1)+volume_svr.image.shape))
        elif self.config.DM_process.mask_init=='template':
            template,full_mask=subject.template(volumes.shape)
            full_mask=full_mask.view((1,1)+full_mask.shape)
            volumes[full_mask==False]=0
        else:
            full_mask=volumes>0.02
            volumes[full_mask==0]=0
        #     slices,volumes = _register( self.config.registration,stacks)
        
        DEBUG_test_save(volumes,f'svort_out')
        for k in range(K-1):
            volumes_downsample,mask_downsample=downsample_3d(volumes,full_mask,scale=2)
            volumes_padded,mask_padded,pads=self.pad(volumes_downsample,mask_downsample)
            max_val,min_val=volumes_padded.max(),volumes_padded.min()
            input_volumes,input_mask=input_transform(volumes_padded),input_transform(mask_padded)
            DEBUG_test_save(input_volumes,f'purified_in_{k}')
            purified_volume=self.diffusion.diffusion_purification(img=input_volumes,
                            T1=t1,
                            K=K,
                            k=k,
                            condition_tensors=input_mask.repeat(1,2,1,1,1),
                            clip_denoised=True,
                            use_tweedie_one_step=self.config.DM_process.use_tweedie if hasattr(self.config.DM_process,'use_tweedie')  else False,
                        )

            DEBUG_test_save(purified_volume,f'purified_out_{k}')
            output_volume=reverse_transform(purified_volume)
            #output_volume=output_volume*(max_val-min_val)+min_val
            
            DEBUG_test_save(output_volume,f'rever_tran_out_{k}')
            volumes_unpad=self.unpad(output_volume,pads)
            volumes_upsample,_=upsample_3d(volumes_unpad,scale=2)
            #update 
            DEBUG_test_save(volumes_upsample,f'diffusion_out_upsample_{k}')
            saved_volumes.append(volumes_upsample)

            if self.config.DM_process.sr=='svr' and k<K-1 :

                v=Volume(volumes_upsample.squeeze().contiguous(),full_mask.squeeze(),resolution_x=self.config.registration.resolution_volume,
                        transformation=RigidTransform(torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],device=volumes.device,dtype=torch.float32))) 
                no_local_exclusion=self.config.DM_process.no_local_exclusion if hasattr(self.config.DM_process,'no_local_exclusion') else False
                psf=self.config.DM_process.psf if hasattr(self.config.DM_process,'psf') else "gaussian"
                no_regular=self.config.DM_process.no_regular if hasattr(self.config.DM_process,'no_regular') else False
                v,_,_=forward_reconstruction(slices=slices,init_volume=v,device=self.device,n_iter=1,no_local_exclusion=no_local_exclusion,no_regular=no_regular,psf=psf,k_schedule=k)
                volumes=v.image.view((1,1)+v.shape)
                saved_volumes.append(volumes)
                DEBUG_test_save(volumes,f'diffusion_out_register_{k}')
            elif self.config.DM_process.sr=='inr' and k<K-1:
                volumes_upsample[full_mask==0]=0
                v=Volume(volumes_upsample.squeeze().contiguous(),full_mask.squeeze().bool(),resolution_x=self.config.registration.resolution_volume,
                        transformation=RigidTransform(torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],device=volumes.device,dtype=torch.float32))) 
                inr,volumes_INR,mask_volume=INR_super_resolution(slices,v,self.config.inr,scale=2)
                inr=None
                if self.config.inr.use_inr == 'stackaftervolume':
                    inr,volumes_INR,mask_volume=optimization_inr(v,slices,inr,self.config.inr)

                volumes_uncorrected=volumes_INR.image.view((1,1)+v.shape)
                volumes=DiffusionModelManager.sigma3_correction(volumes_uncorrected,full_mask)
                saved_volumes.append(volumes)
                DEBUG_test_save(volumes,f'inr_{k}')
            elif self.config.DM_process.sr=='cg' and k<K-1:
                volumes_upsample[full_mask==0]=0
                v=Volume(volumes_upsample.squeeze().contiguous(),full_mask.squeeze().bool(),resolution_x=self.config.registration.resolution_volume,
                        transformation=RigidTransform(torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],device=volumes.device,dtype=torch.float32))) 
                v=cg_reconstruction(slices,v)
                volumes=v.image.view((1,1)+v.shape)
                DEBUG_test_save(volumes,f'diffusion_out_register_{k}')
            elif self.config.DM_process.sr=='generator':
                from utils.unet import optimize_volume_from_slices_pipeline
                v=Volume(volumes_unpad.squeeze().contiguous(),resolution_x=self.config.registration.resolution_volume*2,
                        transformation=RigidTransform(torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],device=volumes.device,dtype=torch.float32))) 
                v_updated=optimize_volume_from_slices_pipeline(v,slices,device=self.device)
                volumes=v_updated
                volumes,_=upsample_3d(volumes,scale=2)
                DEBUG_test_save(volumes,f'diffusion_out_register_{k}')
        return saved_volumes
    def run_subject_dpas(self,subject):
        K=self.config.task.K
        target_shape=[]
        try:
            if self.config.dataset.direction !='all':
                stacks=subject.input_stacks_dir(direct=self.config.dataset.direction)
            else:
                stacks=subject.input_stacks
            slices,volumes = _register( self.config.registration,stacks)
        except:
            print(f"error loading stack of {subject.get_name}")
            return None
        if self.config.DM_process.mask_init=='template':
            template,full_mask=subject.template(volumes.shape)
            full_mask=full_mask.view((1,1)+full_mask.shape)
            volumes[full_mask==False]=0
        else:
            full_mask=volumes>0.02
            volumes[full_mask==0]=0
        DEBUG_test_save(volumes,f'svort_out')
        
        T=self.diffusion.num_timesteps - 1
        self.annealing_scheduler = [
            int(round(T * (1 - k / (K - 1)))) if K > 1 else T
            for k in range(K)
        ]
        for k in range(K-1):
            volumes_downsample,mask_downsample=downsample_3d(volumes,full_mask,scale=2)
            volumes_padded,mask_padded,pads=self.pad(volumes_downsample,mask_downsample,target_shape)
            max_val,min_val=volumes_padded.max(),volumes_padded.min()
            input_volumes,input_mask=input_transform(volumes_padded),input_transform(mask_padded)
            z=torch.rand_like(input_volumes,device=self.device)
            with torch.no_grad():
                volumes_output=self.diffusion.sample_from_z_t(z,t_start=self.annealing_scheduler[k],condition_tensors = torch.cat([input_mask,input_mask],dim=1))
            output_volume=reverse_transform(volumes_output)
            output_volume=output_volume*(max_val-min_val)+min_val
            DEBUG_test_save(output_volume,f'rever_tran_out_{k}')
            volumes_unpad=self.unpad(output_volume,pads)
            volumes_upsample,_=upsample_3d(volumes_unpad,scale=2)
            v=Volume(volumes_upsample.squeeze().contiguous(),full_mask.squeeze(),resolution_x=self.config.registration.resolution_volume,
                transformation=RigidTransform(torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],device=volumes.device,dtype=torch.float32))) 
            v,_,_=forward_reconstruction(slices=slices,init_volume=v,device=self.device,n_iter=1)

            volumes=v.image.view((1,1)+v.shape)
            if k==K-2:
                volumes[full_mask==0]=0   
                return volumes
            z= self.diffusion.q_sample(x_start=volumes, t=torch.full((1,), self.annealing_scheduler[k+1], device=self.device, dtype=torch.long), noise=torch.randn_like(volumes))
            #z=v0y + torch.randn_like(v0y) * self.annealing_scheduler.sigma_steps[k + 1]
    def run_subject_fide(self,subject):
        saved_volumes=[]
        try:
            if self.config.dataset.direction !='all':
                stacks=subject.input_stacks_dir(direct=self.config.dataset.direction)
            else:
                stacks=subject.input_stacks
            slices,volumes = _register( self.config.registration,stacks)
        except:
            print(f"error loading stack of {subject.get_name}")
            return None
        batch_size = 1
        full_mask=volumes>0.02
        volumes_downsample,mask_downsample=downsample_3d(volumes,full_mask,scale=2)
        volumes_padded,mask_padded,pads=self.pad(volumes_downsample,mask_downsample)
        input_volumes,input_mask=input_transform(volumes_padded),input_transform(mask_padded)
        img=torch.randn(input_volumes.shape, device=input_volumes.device)
        with torch.no_grad():
            for i in tqdm(reversed(range(0, self.diffusion.num_timesteps)), desc='sampling loop time step', total=self.diffusion.num_timesteps):
                
                t = torch.full((batch_size,), i, device=volumes.device, dtype=torch.long)
                img = self.diffusion.p_sample(img, t, condition_tensors=torch.cat([input_mask,input_mask],dim=1))
                ##fide_process
                if i%400==0:
                    output_volume=reverse_transform(img)
                    volumes_unpad=self.unpad(output_volume,pads)
                    volumes_upsample,_=upsample_3d(volumes_unpad,scale=2)
                    DEBUG_test_save(volumes_upsample,f'diff_{i}.nii.gz')
                    v=Volume(volumes_upsample.squeeze().contiguous(),full_mask.squeeze(),resolution_x=self.config.registration.resolution_volume,
                    transformation=RigidTransform(torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],device=volumes.device,dtype=torch.float32))) 
                    v,_,_=forward_reconstruction(slices=slices,init_volume=v,device=self.device,n_iter=1)
                    
                    volumes=v.image.view((1,1)+v.shape)
                    DEBUG_test_save(volumes,f'register_{i}.nii.gz')
                    if i==0:
                        volumes[full_mask==0]=0   
                        return volumes
                    volumes_downsample,mask_downsample=downsample_3d(volumes,full_mask,scale=2)
                    volumes_padded,mask_padded,pads=self.pad(volumes_downsample,mask_downsample)
                    img,input_mask=input_transform(volumes_padded),input_transform(mask_padded)
                    img = self.diffusion.q_sample(x_start=img, t=t, noise=torch.randn_like(img))
                    DEBUG_test_save(img,f'noisy_{i}.nii.gz')
    def run_subject_dpcp_no_restart(self,subject):
        saved_volumes=[]
        epochs=1000
        K=self.config.task.K
        t1=self.config.task.T_1

        try:
            if self.config.dataset.direction !='all':
                stacks=subject.input_stacks_dir(direct=self.config.dataset.direction)
            else:
                stacks=subject.input_stacks
            slices,volumes = _register( self.config.registration,stacks)
        except:
            print(f"error loading stack of {subject.get_name}")
            return None
        
        
        if self.config.DM_process.mask_init=='svr_init':
            volume_svr,_,_=svr_reconstruction(slices,device=self.device)
            volumes=pad_to_even(volume_svr.image.view((1,1)+volume_svr.image.shape))
            full_mask=pad_to_even(volume_svr.mask.view((1,1)+volume_svr.image.shape))
        elif self.config.DM_process.mask_init=='template':
            template,full_mask=subject.template(volumes.shape)
            full_mask=full_mask.view((1,1)+full_mask.shape)
            volumes[full_mask==False]=0
        else:
            full_mask=volumes>0.02
            volumes[full_mask==0]=0
        #     slices,volumes = _register( self.config.registration,stacks)
        
        DEBUG_test_save(volumes,f'svort_out')
        for k in range(K-1):
            volumes_downsample,mask_downsample=downsample_3d(volumes,full_mask,scale=2)
            volumes_padded,mask_padded,pads=self.pad(volumes_downsample,mask_downsample)
            max_val,min_val=volumes_padded.max(),volumes_padded.min()
            input_volumes,input_mask=input_transform(volumes_padded),input_transform(mask_padded)
            DEBUG_test_save(input_volumes,f'purified_in_{k}')
            purified_volume=self.diffusion.diffusion_purification(img=input_volumes,
                            T1=t1,
                            K=K,
                            k=0,
                            condition_tensors=input_mask.repeat(1,2,1,1,1),
                            clip_denoised=True,
                        )

            DEBUG_test_save(purified_volume,f'purified_out_{k}')
            output_volume=reverse_transform(purified_volume)
            #output_volume=output_volume*(max_val-min_val)+min_val
            
            DEBUG_test_save(output_volume,f'rever_tran_out_{k}')
            volumes_unpad=self.unpad(output_volume,pads)
            volumes_upsample,_=upsample_3d(volumes_unpad,scale=2)
            #update 
            DEBUG_test_save(volumes_upsample,f'diffusion_out_upsample_{k}')
            saved_volumes.append(volumes_upsample)

            if self.config.DM_process.sr=='svr' and k<K-1 :

                v=Volume(volumes_upsample.squeeze().contiguous(),full_mask.squeeze(),resolution_x=self.config.registration.resolution_volume,
                        transformation=RigidTransform(torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],device=volumes.device,dtype=torch.float32))) 
                no_local_exclusion=self.config.DM_process.no_local_exclusion if hasattr(self.config.DM_process,'no_local_exclusion') else False
                psf=self.config.DM_process.psf if hasattr(self.config.DM_process,'psf') else "gaussian"
                no_regular=self.config.DM_process.no_regular if hasattr(self.config.DM_process,'no_regular') else False
                v,_,_=forward_reconstruction(slices=slices,init_volume=v,device=self.device,n_iter=1,no_local_exclusion=no_local_exclusion,no_regular=no_regular,psf=psf)
                volumes=v.image.view((1,1)+v.shape)
                saved_volumes.append(volumes)
                DEBUG_test_save(volumes,f'diffusion_out_register_{k}')
        return saved_volumes
    def run_subject_svort(self,subject):
        stacks=subject.input_stacks_dir(direct=self.config.dataset.direction)
        slices,volumes = _register( self.config.registration,stacks)
        return volumes
    
    @staticmethod
    def sigma3_correction(
        x: torch.Tensor,
        mask = None,
        k: float = 3.0,
        stats_over_mask: bool = True,     # 统计 μ,σ 是否只在 mask 内做
        correct_only_mask: bool = False,  # 是否只校正 mask 内的体素
        eps: float = 1e-8,
        return_stats: bool = False,
    ) -> torch.Tensor :
        """
        3σ 异常值校正:
            I' = clamp(I, μ-kσ, μ+kσ)

        Args:
            x: 输入张量，支持形状
            - [D, H, W]
            - [C, D, H, W]
            - [B, C, D, H, W]
            mask: 可选 bool mask，形状可与 x 广播兼容（最好与空间维一致）
            k: 默认 3.0
            stats_over_mask: True 表示 μ,σ 只在 mask=True 的体素上统计
            correct_only_mask: True 表示只对 mask=True 的体素进行截断，其余保持原值
            eps: 避免 σ=0
            return_stats: 是否返回 (y, μ, σ)

        Returns:
            y 或 (y, μ, σ)
        """
        if x.numel() == 0:
            raise ValueError("Input tensor is empty.")

        x_float = x.float()

        # 准备用于统计的掩码
        if mask is not None:
            mask_bool = mask.bool()
            mask_b=mask_bool
        else:
            mask_b = None

        # 统计 μ, σ
        if (mask_b is not None) and stats_over_mask:
            vals = x_float[mask_b]
            if vals.numel() == 0:
                # mask 全空就退化为全局统计
                vals = x_float.reshape(-1)
        else:
            vals = x_float.reshape(-1)

        mu = vals.mean()
        sigma = vals.std(unbiased=False)  # 更稳定，且对大体素数更常用
        sigma = torch.clamp(sigma, min=eps)

        low = mu - k * sigma
        high = mu + k * sigma

        # 校正（等价于你的分段函数）
        y = torch.clamp(x_float, low, high)

        # 是否只在 mask 内更新
        if (mask_b is not None) and correct_only_mask:
            y = x_float.clone()
            y[mask_b] = torch.clamp(x_float[mask_b], low, high)

        # 保持 dtype
        y = y.to(dtype=x.dtype)

        if return_stats:
            return y, float(mu.item()), float(sigma.item())
        return y

def main():
    # init
    parser = argparse.ArgumentParser(description='初始化扩散模型管理器，支持指定配置文件路径')

    parser.add_argument(
        '--config_path', 
        '-c',            
        type=str,         
        default='/home/lvyao/git/med-ddpm/config/public_config/public_init_daps.yaml',  # 默认值，不输入时使用
        help='扩散模型配置文件的路径，默认值：git/med-ddpm/config/dm_plug_config.yaml'
    )
    
    # 3. 解析命令行参数
    config_file = parser.parse_args()
    manager = DiffusionModelManager(config_path=config_file.config_path)
    time_records = []
    input_data_index = 0

    # 进行推理并保存结果
    for input_data in manager.dataset:
        start_time = time.time()
        # if input_data.get_name!='30W-2010081819':
        #     print(input_data.get_name)
        #     continue
        if manager.config.task.name=='DMplug':
            output = manager.run_subject_dmplug(input_data)
        elif manager.config.task.name=='dpcp':
            output = manager.run_subject_dpcp(input_data)
        elif manager.config.task.name=='no_restart':
            output = manager.run_subject_dpcp_no_restart(input_data)
        elif manager.config.task.name=='dpcp_debug':
            try:
                output = manager.run_subject_dpcp(input_data)
            except :
                
                print('incorrect mask')
                continue
        elif manager.config.task.name=='svort':
            output = manager.run_subject_svort(input_data)


        elif manager.config.task.name=='fide':
            output = manager.run_subject_fide(input_data)
        elif manager.config.task.name=='dpas':
            output = manager.run_subject_dpas(input_data)
        else:
            raise NotImplementedError
        elapsed_time = round(time.time() - start_time, 6)
        time_records.append({
            "input_data_index": input_data_index,
            "elapsed_time_seconds": elapsed_time
        })
        
        # 索引自增
        input_data_index += 1

        if output is not None:
            manager.save_results(output,input_data)
    if time_records:
        total_time = sum([record["elapsed_time_seconds"] for record in time_records])
        avg_time = round(total_time / len(time_records), 6)
    else:
        avg_time = 0.0  # 无数据时均值为0

    save_data = {
        "task_name": manager.config.task.name,
        "total_input_data_count": len(time_records),
        "average_elapsed_time_seconds": avg_time,
        "per_input_data_time": time_records
    }
    os.makedirs(manager.config.DM_process.save_dir, exist_ok=True)
    save_path = os.path.join(manager.config.DM_process.save_dir, "execution_time.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4)  

if __name__ == "__main__":
    main()
