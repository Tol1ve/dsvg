from nesvor import svr
from nesvor.image import Slice,Volume,load_volume
from nesvor.inr import models
from nesvor.cli.commands import _register,_segment_stack
from nesvor.cli.parsers import main_parser
from nesvor.cli.io import inputs
#from nesvor.transform import RigidTransform
import os
from typing import List, Tuple
import torch
from argparse import Namespace
import sys
from nesvor.cli.main import run
import torchio as tio
from monai.utils import optional_import
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")
from diffusion_model.trainer_fide import FideGaussianDiffusion,resize_target
from diffusion_model.unet import create_model
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from utils.script_process import Subject,DataSet,datesetloader
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.squeeze(0)),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
    Lambda(lambda t: t.transpose(3, 1)),
])
input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: t.squeeze(0) if t.ndim==5 else t),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),

])
def save_tensor_as_nii(tensor, filename, affine=None):
    """
    将三维张量保存为 NIfTI 文件 (.nii 或 .nii.gz)。
    
    Args:
        tensor (torch.Tensor or np.ndarray): 三维数据张量，形状为 (D, H, W)。
        filename (str): 保存的文件路径（需包含 .nii 或 .nii.gz 后缀）。
        affine (np.ndarray, optional): 4x4 仿射矩阵，用于定义图像坐标系。
                                       默认为单位矩阵。
    
    Returns:
        None
    """
    # 检查输入数据类型
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()  # 将 PyTorch 张量转换为 NumPy 数组
    
    if tensor.ndim != 3:
        raise ValueError("输入张量必须是三维的，形状应为 (D, H, W)") 
    import numpy as np
    # 默认仿射矩阵（单位矩阵）
    if affine is None:
        affine = np.eye(4)
    import nibabel as nib
    # 创建 NIfTI 图像对象
    nifti_image = nib.Nifti1Image(tensor, affine)
    
    # 保存为 NIfTI 文件
    nib.save(nifti_image, filename)
    print(f"文件已保存为: {filename}")
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
def label2masks(masked_img,input_channel,batch_size=1):
    result_img = torch.ones(masked_img.shape + (input_channel - 1,)).to(masked_img.device)
    result_img[...,0][masked_img==0] = 0
    #result_img[masked_img==LabelEnum.TUMORAREA.value, 1] = 1
    return result_img
def resize_img_mask(img: torch.Tensor,batch_size,shape) -> torch.Tensor:
    mask_ones=None
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
    if mask_ones:
        
        resized_tensor=torch.cat([resized_tensor,-1*torch.ones_like(resized_tensor)],dim=0)
    return resized_tensor,prefix_shape,nz
class dpmRecon(object):
    def __init__(
        self, args: Namespace,input_size,num_channels,num_res_blocks,num_class_labels,out_channels,depth_size,weightfile,mode='svr',
    ) -> None:
        super().__init__()
        self.args = args
        self.mode=mode
        self.device=args.device
        self.batch_size=1
        self.channels=1
        self.depth_size=depth_size
        self.image_size=input_size

        model = create_model(input_size, num_channels, num_res_blocks, in_channels=num_class_labels, out_channels=out_channels).cuda()
        self.diffusion = FideGaussianDiffusion(
            model,
            image_size = input_size,
            depth_size = depth_size,
            timesteps = args.timesteps,   # number of steps
            loss_type = 'L1', 
            with_condition=True,
        ).to(self.args.device)
        self.diffusion.load_state_dict(torch.load(weightfile)['ema'])
        print("Model Loaded!")
        
    def run(self,slices:List[Slice]):
        USE_atlas=False
        agestr=str(self.args.age) +'exp' if self.args.age>=36 else  str(self.args.age) 
        if self.args.age:
            path_vol='//home/lvyao/local/atlas/CRL_FetalBrainAtlas_2017v3/STA'+agestr+'.nii.gz'
        if self.args.age and USE_atlas:
            condition=load_volume(
            path_vol=path_vol,
            path_mask='use_region_greater_zero',
            device=self.device,
        )   
            if self.args.dilate:
                condition.dilate_mask_3d(num_iterations=self.args.dilate)
                condition.image=condition.mask.float()
        elif self.args.age:
            atlas_condition=load_volume(
            path_vol=path_vol,
            device=self.device,
        )
            condition=svr._initial_mask(slices,output_resolution=0.5,
                                            sample_orientation=path_vol,
                                            device=self.device,
                                            mask_atlas=atlas_condition.image,
                                            )[0]
            #condition.transformation=atlas_condition.transformation

            #condition=condition.resample(resolution_new=0.5)
        else:
            path_vol='/home/lvyao/local/atlas/CRL_FetalBrainAtlas_2017v3/STA35.nii.gz'
            condition=svr._initial_mask(slices,output_resolution=0.5,
                                            sample_orientation=path_vol,
                                            device=self.device)[0]
            if self.args.dilate:
                condition.dilate_mask_3d(num_iterations=self.args.dilate)
                condition.image=condition.mask.float()
        condition_mask=condition.image.float()
        condition.save('./init_mask.nii.gz',masked=True)
        batch_size=self.batch_size#目前只支持batch_size=1.
        channels=self.channels
        depth_size=self.depth_size 
        image_size=self.image_size
        shape=(batch_size, channels, depth_size, image_size, image_size)
        condition_mask=label2masks(condition_mask,input_channel=3)
        condition_mask,prefix_shape,nz=resize_img_mask(condition_mask,batch_size,(image_size,image_size,depth_size))
        condition_mask=input_transform(condition_mask)
        self.diffusion.prefix_shape_nz_index=(prefix_shape,nz)
        self.diffusion.slices=slices
        self.diffusion.svr_args=self.args
        self.diffusion.condition_transformation=condition
        #refine process
        img=self.diffusion.sample(batch_size = 1, condition_tensors = torch.cat([condition_mask,-1*torch.ones_like(condition_mask)],dim=1))
        #post processing
        if isinstance(img,Volume):
            #post process
            #img.recontrast(mode='unsharp')
            img.rescale(self.args.output_intensity_mean,masked=True)
            img.save(path=self.args.output_volume,masked=True)
            reference_mask=svr._initial_mask(slices,output_resolution=0.5,
                                sample_orientation=path_vol,
                                device=self.device)[0]
            ref_nz,_=crop_nonzero_tensor(reference_mask.image.cpu())
            img_nz,_,nz=resize_img_mask(img.image.cpu(),batch_size,ref_nz.shape)
            mask_nz,_,_=resize_img_mask(img.mask.cpu(),batch_size,ref_nz.shape)
            image_reshape=resize_target(img_nz[None,...],reference_mask.image[None,...].shape,nz)
            mask_reshape=resize_target(mask_nz[None,...].float(),reference_mask.image[None,...].shape,nz)
            img_reshape=Volume(image=image_reshape[0].to(img.device),mask=mask_reshape[0].bool().to(img.device),transformation=img.transformation,
                               resolution_x=img.resolution_x,resolution_y=img.resolution_y,resolution_z=img.resolution_z)
            img_reshape.save(path=self.args.output_volume.replace(".nii.gz","_reshape.nii.gz"),masked=True)
        elif isinstance(img,torch.Tensor):
            img=(img+1)/2
            volume=Volume(image=img.squeeze(),mask=condition_mask.squeeze()>0,
                transformation=condition.transformation,resolution_x=0.5,resolution_y=0.5,resolution_z=0.5)
            volume.save(path=self.args.output_volume.replace(".nii.gz","_ablation.nii.gz"),masked=True)


def test():
    #just test
    print(models.USE_TORCH)
    sys.argv.pop(0)
    sys.argv.insert(0,"reconstruct")
    sys.argv.insert(0,"nesvor")
    parser, subparsers = main_parser()
    args = parser.parse_args()
    args.dtype = torch.float32 if args.single_precision else torch.float16
    args.timesteps=250
    args.age=0
    args.dilate=3
    args.gap=90
    args.ablation=''
    input_dict, args = inputs(args)
    slices = _register(args,input_dict['input_stacks'])
    DR=dpmRecon(
        args,input_size=128,depth_size=128,num_channels=64,num_class_labels=3,num_res_blocks=1,out_channels = 1,
        weightfile='/home/lvyao/git/med-ddpm/results/model-8.pt')
    DR.run(slices)

def main(dataset:DataSet):
    for i in range(len(dataset)):
        command=dataset.run(i)
        if command is None:
            print("no need for reconstructed subject in resume process")
            continue
        sys.argv.clear()
        #TODO: different command
        sys.argv.extend(['nesvor','segment-stack'])
        sys.argv.extend(command)
        parser, subparsers = main_parser()
        args = parser.parse_args()

        try:
            run(args)
        except Exception as e:
            print(e)
            print("error in subject:",dataset[i].get_name)
            continue
def main_fide(dataset:DataSet):
    for i in range(len(dataset)):

        command=dataset.run(i)
        if command is None:
            print("no need for reconstructed subject in resume process")
            continue
        sys.argv.clear()
        #TODO: different command
        sys.argv.extend(['nesvor','reconstruct'])
        sys.argv.extend(command)
        parser, subparsers = main_parser()
        args = parser.parse_args()
        # some new paras
        args.device=3
        args.dtype = torch.float32 if args.single_precision else torch.float16
        args.timesteps=250
        args.age=dataset.age(i)
        args.age=0
        args.dilate=3
        args.gap=90
        args.ablation=''

        #print(args)
        input_dict, args = inputs(args)

        if args.segmentation:
            input_dict['input_stacks']=_segment_stack(args,input_dict['input_stacks'])
        slices,svort_volume = _register(args,input_dict['input_stacks'])

        svort_volume.save(path=args.output_volume.replace(".nii.gz","_init.nii.gz"))
        DR=dpmRecon(
        args,input_size=128,depth_size=128,num_channels=64,num_class_labels=3,num_res_blocks=1,out_channels = 1,
        weightfile='/home/lvyao/git/med-ddpm/results/results_other_versions/model-8.pt')
        DR.run(slices)
        # try:
        #     DR.run(slices)
        # except Exception as e:
        #     print(e)
        #     print("error in subject:",dataset[i].get_name)
        #     continue
def main_nesvor(dataset:DataSet):
    from nesvor.cli.main import run
    for i in range(len(dataset)):
        command=dataset.run(i)
        if command is None:
            print("no need for reconstructed subject in resume process")
            continue
        sys.argv.clear()
        #TODO: different command
        sys.argv.extend(['nesvor','reconstruct'])
        sys.argv.extend(command)
        parser, subparsers = main_parser()
        args = parser.parse_args()
        run(args)

def sim_dataset_get(sim_in,mode='fide',sim_out=None,mask_folder=None):
    
    if sim_out is None:
        sim_out='/home/lvyao/local/sim/result/'+mode+'_noatlas_dr3'
    resume=False
    return datesetloader(sim_in,sim_out,mode=mode,resume=resume,maskfolder=mask_folder)

if __name__ == "__main__":

    # dataset=sim_dataset_get(sim_in='/home/lvyao/local/dataset/fetal_clinical/clinival_lv_stacks_in/data',
    #                         sim_out='/home/lvyao/local/dataset/fetal_clinical/clinival_lv_stacks_in/result',
    #                         mask_folder='/home/lvyao/local/dataset/fetal_clinical/clinival_lv_stacks_in/mask(115)')
    
    dataset=sim_dataset_get(sim_in='/home/lvyao/local/dataset/fetal_clinical/shengfy_0203/input_stack',
                            sim_out='/home/lvyao/local/dataset/fetal_clinical/shengfy_0203/result',
                            mask_folder='/home/lvyao/local/dataset/fetal_clinical/shengfy_0203/mask')
    dataset.FBS_Seg=True
    main_fide(dataset)

