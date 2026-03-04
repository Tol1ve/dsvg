import sys
import os
import inspect
import torch
import yaml
from argparse import Namespace
import argparse
import nibabel as nib
import subprocess
import time
import numpy as np
import torch.nn.functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# 此时可以直接导入b模块
from dataset import Subject
import dataset
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
    img_np = img.cpu().numpy()
    img_nii = nib.Nifti1Image(img_np, affine)
    #
    save_path=os.path.join(args.save_dir,name)
    nib.save(img_nii, save_path)

class DiffusionModelManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_yaml(config_path)
        self.device=torch.device(f"cuda:{self.config.dataset.device}")
        self._initialize_dataset()
        if hasattr(self.config,'pretrained_model'):
            self._initialize_model()
    def _initialize_model(self):
        from diffusion_model.unet import create_model
        
        from diffusion_model.trainer import GaussianDiffusion,CFGDiffusion,DPCPDiffusion
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

   
    def _load_model_weights(self, weight_path: str):
        """加载预训练模型权重"""
        if not os.path.exists(weight_path):
            raise ValueError(f"Weight file {weight_path} does not exist.")
        
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.diffusion.load_state_dict(checkpoint['ema'])
        print(f"Loaded weights from {weight_path}")
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
    def run_subject_svrtk(self,subject):
        save_dir=self.config.DM_process.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        outpath=os.path.join(save_dir,f"{subject.get_name}.nii.gz")
        filelist_dir=[s for s in subject.file_list if (self.config.dataset.direction in s) ] 
        if len(filelist_dir)==0:
            return
        command=['mirtk','reconstruct',outpath,str(len(filelist_dir))]+filelist_dir

        command+=["-thickness"]+["4"]*len(filelist_dir)
        command+=["-svr_only","-resolution",f"{self.config.DM_process.resolution_volume}","-iterations", "3","-with_background"] 

        print(command)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    def run_subject_niftymic(self,subject):
        save_dir=self.config.DM_process.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filelist_dir=[s for s in subject.file_list if (self.config.dataset.direction in s) ] 
        if len(filelist_dir)==0:
            return
        command=['niftymic_reconstruct_volume','--filenames']+filelist_dir
        outpath=os.path.join(save_dir,f"{subject.get_name}.nii.gz")
        command+=["--output",outpath]
        print(command)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout = result.stdout
        stderr = result.stderr
        print(stdout)
        print(stderr)
    def run_exp(self,subject):
        import csv
        import matplotlib.pyplot as plt
        from registration import _register,svr_reconstruction,forward_process,Volume,RigidTransform
        from DMpipe import pad_to_even,downsample_3d,DEBUG_test_save,input_transform,reverse_transform,upsample_3d
        mse_records = []
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
        full_mask=volumes>0.02
        # volume_svr,_,_=svr_reconstruction(slices,device=self.device)
        # volumes=pad_to_even(volume_svr.image.view((1,1)+volume_svr.image.shape))
        # full_mask=pad_to_even(volume_svr.mask.view((1,1)+volume_svr.image.shape))
        for i in range(99,self.config.pretrained_model.timesteps,100):
            volumes_downsample,mask_downsample=downsample_3d(volumes,full_mask,scale=2)
            volumes_padded,mask_padded,pads=self.pad(volumes_downsample,mask_downsample)
            max_val,min_val=volumes_padded.max(),volumes_padded.min()
            input_volumes,input_mask=input_transform(volumes_padded),input_transform(mask_padded)
            purified_volume=self.diffusion.diffusion_purification(img=input_volumes,
                            T1=i,
                            K=K,
                            k=0,
                            condition_tensors=input_mask.repeat(1,2,1,1,1),
                            clip_denoised=True,
                        )
            output_volume=reverse_transform(purified_volume)
            #output_volume=output_volume*(max_val-min_val)+min_val
            
            volumes_unpad=self.unpad(output_volume,pads)
            volumes_upsample,_=upsample_3d(volumes_unpad,scale=2)
            #update 
            v=Volume(volumes_upsample.squeeze().contiguous(),full_mask.squeeze(),resolution_x=self.config.registration.resolution_volume,
                        transformation=RigidTransform(torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,1,0]]],device=volumes.device,dtype=torch.float32))) 
            err=forward_process(v,slices,self.config.forward_process).slices


            mse = (err.float() ** 2).mean()

            # 记录（转成 python float，便于写 csv）
            mse_val = float(mse.detach().cpu().item())
            mse_records.append({"i": int(i), "mse": mse_val})

            # （可选）打印一下监控

        # ====== 新增：保存 CSV ======
        with open(self.config.DM_process.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["i", "mse"])
            writer.writeheader()
            writer.writerows(mse_records)

        # ====== 新增：绘图并保存 ======
        xs = [r["i"] for r in mse_records]
        ys = [r["mse"] for r in mse_records]

        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel("timestep i")
        plt.ylabel("mean(err^2)")
        plt.title(f"MSE curve ")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.tight_layout()
        plt.savefig(self.config.DM_process.save_plt, dpi=200)
        plt.close()

        return 
            
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
def main():
    # init
    parser = argparse.ArgumentParser(description='初始化扩散模型管理器，支持指定配置文件路径')

    parser.add_argument(
        '--config_path', 
        '-c',            
        type=str,         
        default='/home/lvyao/git/med-ddpm/config/test/dm_svrtk_config.yaml',  # 默认值，不输入时使用
        help='扩散模型配置文件的路径，默认值：git/med-ddpm/config/dm_plug_config.yaml'
    )
    import time
    import json
    time_records = []
    input_data_index = 0
    # 3. 解析命令行参数
    config_file = parser.parse_args()
    manager = DiffusionModelManager(config_path=config_file.config_path)
    for input_data in manager.dataset:
        start_time = time.time()
        if manager.config.task.name=='svrtk':
            output=manager.run_subject_svrtk_new(input_data)
        elif    manager.config.task.name=='niftymic':
             output=manager.run_subject_niftymic(input_data)
        elif manager.config.task.name=='experiment':
            output=manager.run_exp(input_data)
        elapsed_time = round(time.time() - start_time, 6)
        time_records.append({
            "input_data_index": input_data_index,
            "elapsed_time_seconds": elapsed_time
        })

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
