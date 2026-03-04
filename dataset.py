#-*- coding:utf-8 -*-
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
#from torchvision.transforms import Compose, ToTensor, Lambda
from glob import glob
#from utils.dtypes import LabelEnum
import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
import numpy as np
import torch
import re
import os
import torch.nn.functional as F
import random
from utils.image_process import downsample_3d,upsample_3d

class NiftiImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, depth_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.depth_size = depth_size
        self.inputfiles = glob(os.path.join(imagefolder, '*.nii.gz'))
        self.scaler = MinMaxScaler()
        self.transform = transform

    def read_image(self, file_path):
        img = nib.load(file_path).get_fdata()
        img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot_samples(self, n_slice=15, n_row=4):
        samples = [self[index] for index in np.random.randint(0, len(self), n_row*n_row)]
        for i in range(n_row):
            for j in range(n_row):
                sample = samples[n_row*i+j]
                sample = sample[0]
                plt.subplot(n_row, n_row, n_row*i+j+1)
                plt.imshow(sample[:, :, n_slice])
        plt.show()

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        inputfile = self.inputfiles[index]
        img = self.read_image(inputfile)
        h, w, d= img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(inputfile)
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]

        if self.transform is not None:
            img = self.transform(img)
        return img

class NiftiPairImageGenerator(Dataset):
    def __init__(self,
            input_folder: str,
            target_folder: str,
            input_size: int,
            depth_size: int,
            input_channel: int = 2,
            transform=None,
            target_transform=None,
            full_channel_mask=False,
            combine_output=False,
            nozero=False,
            nesvor_volume=False,
            train=True,
            use_resized=False,
            padded=False,
            downsample=True,
            with_condition = True, 
        ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.train=train
        self.nesvor_volume=nesvor_volume
        self.pair_files = self.pair_file()
        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output
        self.nonzero=nozero
        self.use_resized=use_resized
        self.padded=padded
        self.downsample=downsample
        self.with_condition = with_condition 
    def pair_file(self):
        raise NotImplementedError
    def crop_nonzero(self,input_img):
        non_zero_indices = np.argwhere(input_img != 0)

        # 找到不为0的元素的索引

        if non_zero_indices.size == 0:
            cropped_tensor=input_img
            print("err")
        else:
        # 获取每个维度的最小和最大索引
            min_indices = non_zero_indices.min(axis=0)
            max_indices = non_zero_indices.max(axis=0)

            # 使用每个维度的范围裁剪张量
            slices = tuple(slice(min_indices[i], max_indices[i] + 1) for i in range(input_img.ndim))
            cropped_tensor = input_img[slices]
    # 裁剪张量
        return cropped_tensor
    def label2masks(self, masked_img):
        result_img = np.ones(masked_img.shape)
        result_img[masked_img==0] = 0
        #result_img[masked_img==LabelEnum.TUMORAREA.value, 1] = 1
        return result_img

    def read_image(self, file_path, pass_scaler=False):
        img = nib.load(file_path).get_fdata()
        if not pass_scaler:
            img = self.scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # 0 -> 1 scale
        return img

    def plot(self, index, n_slice=30):
        data = self[index]
        input_img = data['input']
        target_img = data['target']
        plt.subplot(1, 2, 1)
        plt.imshow(input_img[:, :, n_slice])
        plt.subplot(1, 2, 2)
        plt.imshow(target_img[:, :, n_slice])
        plt.show()

    def resize_img(self, img):
        if not self.use_resized:
            return img
        if self.nonzero:
            img=self.crop_nonzero(img)
        h, w, d = img.shape
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(cop(img))[0]
        return img

    def resize_img_4d(self, input_img):
        if not self.use_resized:
            return input_img
        h, w, d, c = input_img.shape
        result_img = np.zeros((self.input_size, self.input_size, self.depth_size, 2))
        if h != self.input_size or w != self.input_size or d != self.depth_size:
            for ch in range(c):
                if self.nonzero:
                    buff=self.crop_nonzero(input_img[..., ch])
                else:   
                    buff = input_img.copy()[..., ch]

                img = tio.ScalarImage(tensor=buff[np.newaxis, ...])
                cop = tio.Resize((self.input_size, self.input_size, self.depth_size))
                img = np.asarray(cop(img))[0]
                result_img[..., ch] += img
            return result_img
        else:
            return input_img

    def sample_conditions(self, batch_size: int,index=None):
        indexes = np.random.randint(0, len(self), batch_size) if index is None else np.array([index]*batch_size)
        input_files = [self.pair_files[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) 
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
            if self.padded:
                input_img,_=self.pad(input_img)
        return torch.cat(input_tensors, 0).cuda()
    def sample_all_evaluate(self):
        input_files = [self.pair_files[index][0] for index in len(self.pair_files)]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img)
            if self.transform is not None:
                input_img = self.transform(input_img).unsqueeze(0)
                input_tensors.append(input_img)
            if self.padded:
                input_img,_=self.pad(input_img)
        return input_tensors
    def __len__(self):
        return len(self.pair_files)
    def pad(self,x, mask=None,multiple=16):
        D,H,W=x.shape[-3:]

        def get_symmetric_pad(size):
            pad = (multiple - size % multiple) % multiple
            left = pad // 2
            right = pad - left
            return left, right

        d0, d1 = get_symmetric_pad(D)
        h0, h1 = get_symmetric_pad(H)
        w0, w1 = get_symmetric_pad(W)

        x_pad = F.pad(x, (w0, w1, h0, h1, d0, d1),value=-1)
            # mask 不为空 → 一起 pad
        if mask is not None:
            # 保证 mask 有 batch 维度与 x 对齐
            if mask.shape[0] != x.shape[0]:
                raise ValueError("mask 和 x 的 batch size 不一致")
            mask_pad = F.pad(mask, (w0, w1, h0, h1, d0, d1),value=-1)
            return x_pad, mask_pad, (d0, d1, h0, h1, w0, w1)
        return x_pad, (d0, d1, h0, h1, w0, w1)
    def __getitem__(self, index):
        input_file, target_file = self.pair_files[index]
        target_img = self.read_image(target_file)
        target_img = self.resize_img(target_img)
        if input_file==target_file:
            input_img = (target_img > 0).astype(np.float32)[..., None]  # 将其转换为 0 或 1 的浮点数，并在最后添加一个维度
            input_img = np.concatenate([input_img, np.zeros_like(input_img)], axis=-1)
        else:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) 

        if self.transform is not None:
            input_img = self.transform(input_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)

        if self.combine_output:
            return torch.cat([target_img, input_img], 0)
        if self.downsample:
            target_img,input_img=downsample_3d(target_img,input_img)
        if self.padded:
            input_img,target_img,_=self.pad(input_img,target_img)
        if not self.with_condition:
            return target_img
        return {'input':input_img, 'target':target_img}


class DHCPDataSet(NiftiPairImageGenerator):
    # def __init__(self, imagefolder, input_size, depth_size, transform=None):
    #     super().__init__(imagefolder, input_size, depth_size, transform)
    def pair_file(self):

        self.root=self.input_folder
        pairs=[]
        name_prefix='_T2w_brain_affine_WENXUAN.nii.gz'
        train_folder=os.path.join(self.root,"train")
        for sub in os.listdir(train_folder):
            input_file=os.path.join(train_folder,sub)
            target_file=os.path.join(input_file,sub+name_prefix)
            mask_file=target_file
            pairs.append((target_file, mask_file))
        valid_folder=os.path.join(self.root,"valid")
        for sub in os.listdir(train_folder):
            input_file=os.path.join(train_folder,sub)
            target_file=os.path.join(input_file,sub+name_prefix)
            mask_file=target_file
            pairs.append((target_file, mask_file))    
        if not self.train:
            train_folder=os.path.join(self.root,"test")
            for sub in os.listdir(train_folder):
                input_file=os.path.join(train_folder,sub)
                target_file=os.path.join(input_file,sub+name_prefix)
                mask_file=target_file
                pairs.append((target_file, mask_file))

        return pairs
class FeTADataSet(NiftiPairImageGenerator):
    def pair_file(self):
        self.root=self.input_folder
        self.label_file = '%s_rec-%s_dseg_reg.nii.gz'
        self.image_file = '%s_rec-%s_T2w_norm_reg.nii.gz'
        pairs = []
        with open(os.path.join('/home/lvyao/git/med-ddpm/config', 'train'), 'r') as f:
            path_names = [p.strip() for p in f.readlines()]
        for p in path_names:
            #assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            input_file=os.path.join(self.root, p, 'anat', self.label_file % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) 
            target_file=os.path.join(self.root, p, 'anat', self.image_file % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) 
            pairs.append((input_file, target_file))
        
        if  self.nesvor_volume:

            for sub in os.listdir('/home/lvyao/git/dataset/selected'):
                input_file=os.path.join('/home/lvyao/git/dataset/selected',sub)
                target_file=input_file
                pairs.append((input_file, target_file))
        if not self.train:
                pairs=[]
                with open(os.path.join('/home/lvyao/git/med-ddpm/config', 'val'), 'r') as f:
                    path_names = [p.strip() for p in f.readlines()]
                for p in path_names:
                    #assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
                    input_file=os.path.join(self.root, p, 'anat', self.image_file % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) 
                    target_file=os.path.join(self.root, p, 'anat', self.label_file % (p, 'mial' if p < 'sub-041' else 'irtk' if p < 'sub-081' else 'nmic')) 
                    pairs.append((input_file, target_file))
        return pairs
class CRLDataSet(NiftiPairImageGenerator):
    def __init__(self,
                 input_folder: str,
                 target_folder: str,
                 input_size: int,
                 depth_size: int,
                 input_channel: int = 2,
                 transform=None,
                 target_transform=None,
                 full_channel_mask=False,
                 combine_output=False,
                 nozero=False,
                 train=True,
                 use_resized=False,
                 padded=False,
                 downsample=True,
                 split_config_path: str = "/home/lvyao/git/svr/datasets/crl_split.txt",
                 with_condition=True):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.train=train

        self.input_size = input_size
        self.depth_size = depth_size
        self.input_channel = input_channel
        self.scaler = MinMaxScaler()
        self.transform = transform
        self.target_transform = target_transform
        self.full_channel_mask = full_channel_mask
        self.combine_output = combine_output
        self.nonzero=nozero
        self.use_resized=use_resized
        self.padded=padded
        self.split_config_path = split_config_path
        self.all_week_pairs = self._generate_all_week_pairs()
        self.pair_files = self.pair_file()
        self.downsample=downsample
        self.with_condition = with_condition 
        self.nesvor_volume=False
    def _generate_all_week_pairs(self):
        week_pairs = {}
        for week in range(23, 39):
            week_str = f"{week}" if week in range(23,36) else f"{week}exp"
            target_file  = os.path.join(self.input_folder, f"STA{week_str}.nii.gz")
            input_file = os.path.join(self.input_folder, f"STA{week_str}_tissue.nii.gz")
            week_pairs[week_str] = (input_file, target_file)
        return week_pairs

    def _create_default_split_config(self):
        random.seed(42)
        all_weeks = list(self.all_week_pairs.keys())
        random.shuffle(all_weeks)
        split_idx = int(len(all_weeks) * 0.8)
        train_weeks = sorted(all_weeks[:split_idx])
        val_weeks = sorted(all_weeks[split_idx:])
        os.makedirs(os.path.dirname(self.split_config_path), exist_ok=True)
        with open(self.split_config_path, 'w', encoding='utf-8') as f:
            f.write("[train]\n")
            f.write('\n'.join(train_weeks) + '\n\n')
            f.write("[val]\n")
            f.write('\n'.join(val_weeks))
        
        print(f"配置文件已生成：{self.split_config_path}")
        print(f"训练集：{train_weeks} | 验证集：{val_weeks}")

    def _read_split_config(self):
        if not os.path.exists(self.split_config_path):
            self._create_default_split_config()
        split_config = {'train': [], 'val': []}
        current_section = None
        with open(self.split_config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                # 识别section（如[train]）
                if line.startswith('[') and line.endswith(']'):
                    current_section = line.strip('[]')
                    if current_section not in split_config:
                        raise ValueError(f"无效的section：{current_section}（仅支持[train]/[val]）")
                # 识别孕周（当前section下的行）
                elif current_section:
                    split_config[current_section].append(line)

        # 校验孕周合法性
        valid_weeks = set(self.all_week_pairs.keys())
        invalid_train = [w for w in split_config['train'] if w not in valid_weeks]
        invalid_val = [w for w in split_config['val'] if w not in valid_weeks]
        if invalid_train:
            raise ValueError(f"训练集包含无效孕周：{invalid_train}（仅支持{valid_weeks}）")
        if invalid_val:
            raise ValueError(f"验证集包含无效孕周：{invalid_val}（仅支持{valid_weeks}）")
        
        return split_config
    def pair_file(self):

        split_config = self._read_split_config()
        target_weeks = split_config['train'] if self.train else split_config['val']
        pairs = [self.all_week_pairs[week] for week in target_weeks]
        return pairs
class CHNDataSet(CRLDataSet):
    def _generate_all_week_pairs(self):
        week_pairs = {}
        for week in range(23, 39):
            week_str = f"{week}w"
            target_file= os.path.join(self.input_folder, f"Atlas_{week_str}.nii.gz")
            input_file  = os.path.join(self.input_folder, f"Atlas_{week_str}_tissue_labels.nii.gz")
            week_pairs[week_str] = (input_file, target_file)
        return week_pairs
class KCLDataSet(CRLDataSet):
    
    def _generate_all_week_pairs(self):
        """重写：适配KCL数据集的文件匹配逻辑
        - 影像文件：image文件夹下的`t2-tXX.00.nii.gz`
        - 掩码文件：label文件夹下的`tissue-tXX.00_dhcp-19.nii.gz`
        """
        week_pairs = {}
        # 1. 筛选image文件夹中所有t2开头的NIfTI文件
        image_folder=os.path.join(self.input_folder,'image_resample')
        label_folder=os.path.join(self.input_folder,'label_resample')
        image_files = [
            f for f in os.listdir(image_folder)
            if f.startswith("t2-") and f.endswith(".nii.gz")
        ]
        if not image_files:
            raise FileNotFoundError(f"在image文件夹{self.input_folder}中未找到t2开头的NIfTI文件")

        # 2. 逐个匹配影像与掩码文件
        for img_file in image_files:
            # 提取胎龄标识（如从`t2-t21.00.nii.gz`中提取`t21.00`）
            age_tag = img_file.split("-")[1].replace(".nii.gz", "")  # 结果为`t21.00`/`t22.00`等
            
            # 构造影像文件完整路径
            target_file = os.path.join(image_folder, img_file)
            # 构造对应掩码文件完整路径
            input_file = os.path.join(label_folder, f"tissue-t2-{age_tag}.nii.gz")

            # 验证掩码文件是否存在
            if not os.path.exists(target_file):
                raise FileNotFoundError(f"影像文件{img_file}对应的掩码文件不存在：{target_file}")

            # 保存（以胎龄标识为key，方便后续划分）
            week_pairs[age_tag] = (input_file, target_file)

        return week_pairs
class CombinedDataSet(NiftiPairImageGenerator):
    def __init__(self,
                 dataset_list):

        if not dataset_list:
            raise ValueError("dataset_list不能为空，请传入至少一个NiftiPairImageGenerator子类实例")
        for ds in dataset_list:
            if not isinstance(ds, NiftiPairImageGenerator):
                raise TypeError(f"dataset_list中的元素必须是NiftiPairImageGenerator子类，当前类型：{type(ds)}")
        
        # 从第一个数据集提取通用参数，初始化父类
        first_ds = dataset_list[0]
       
        self.input_folder=first_ds.input_folder # 仅占位，实际由子数据集管理
        self.target_folder=first_ds.target_folder
        self.input_size=first_ds.input_size
        self.depth_size=first_ds.depth_size
        self.input_channel=first_ds.input_channel
        self.transform=first_ds.transform
        self.target_transform=first_ds.target_transform
        self.full_channel_mask=first_ds.full_channel_mask
        self.combine_output=first_ds.combine_output
        self.nozero=first_ds.nonzero
        self.nesvor_volume=first_ds.nesvor_volume
        self.train=first_ds.train
        self.use_resized=first_ds.use_resized
        self.padded=first_ds.padded
        self.downsample=first_ds.downsample
        self.with_condition = first_ds.with_condition
        # 保存数据集列表和随机采样开关
        self.dataset_list = dataset_list

        
        # 合并所有数据集的pairs，并维护「全局索引→(数据集实例, 局部索引)」的映射
        self.combined_pairs = []
        self.index_mapping = []  # 每个元素：(dataset_instance, local_idx)
         
        # 遍历数据集列表，合并pairs和索引映射
        for ds in self.dataset_list:
            for local_idx, pair in enumerate(ds.pair_files):
                self.combined_pairs.append(pair)
                self.index_mapping.append((ds, local_idx))

    def pair_file(self):
        # 重写pair_file，返回合并后的pairs（兼容父类逻辑）
        return self.combined_pairs

    def __len__(self):
        # 总长度=所有子数据集长度之和
        return len(self.combined_pairs)

    def __getitem__(self, idx):
        # 根据索引映射，获取对应的子数据集和局部索引
        ds_instance, local_idx = self.index_mapping[idx]
        
        # 调用子数据集的__getitem__（复用其所有预处理逻辑）
        return ds_instance[local_idx]
    def sample_conditions(self, batch_size: int,index=None):
        indexes = np.random.randint(0, len(self), batch_size) if index is None else np.array([index]*batch_size)
        input_files = [self.combined_pairs[index][0] for index in indexes]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) 
            if self.transform is not None:
                input_img = self.transform(input_img)
                
            if self.downsample:
                input_img,_=downsample_3d(input_img)
            if self.padded:
                input_img,_=self.pad(input_img)
            input_tensors.append(input_img.unsqueeze(0).repeat(1,2,1,1,1))
        return torch.cat(input_tensors, 0).cuda()
    def sample_all_evaluate(self):
        input_files = [self.combined_pairs[index][0] for index in range(len(self.combined_pairs))]
        input_tensors = []
        for input_file in input_files:
            input_img = self.read_image(input_file, pass_scaler=self.full_channel_mask)
            input_img = self.label2masks(input_img) if self.full_channel_mask else input_img
            input_img = self.resize_img(input_img) 
            if self.transform is not None:
                input_img = self.transform(input_img)
            if self.downsample:
                input_img,_=downsample_3d(input_img)
            if self.padded:
                input_img,_=self.pad(input_img)
            input_tensors.append(input_img)
        return input_tensors
def create_dataset(inputfolder,
        CRLfolder,
        KCLfolder,
        CHNfolder,
        targetfolder,          
        input_size,
        depth_size,
        transform,
        input_transform,
        full_channel_mask,
        nozero,
        train,
        use_resized,
        addition_volume,
        padded,
        downsample,
        with_condition: bool = True):
        dataset_list=[]
        if inputfolder !="":
            feta=FeTADataSet(
                inputfolder,
                targetfolder,
                input_size=input_size,
                depth_size=depth_size,
                transform=input_transform,
                target_transform=transform,
                full_channel_mask=full_channel_mask,
                nozero=nozero,
                train=train,
                use_resized=use_resized,
                padded=padded,
                nesvor_volume=addition_volume,
                with_condition=with_condition,
                downsample=downsample,
                )
            dataset_list.append(feta)
        if CRLfolder !="":
            crl=CRLDataSet(            
                CRLfolder,
                targetfolder,
                input_size=input_size,
                depth_size=depth_size,
                transform=input_transform,
                target_transform=transform,
                full_channel_mask=full_channel_mask,
                nozero=nozero,
                train=train,
                use_resized=use_resized,
                padded=padded,
                split_config_path='/home/lvyao/git/med-ddpm/config/crl_split.txt',
                with_condition=with_condition,
                downsample=downsample,
                )
            dataset_list.append(crl)
        if KCLfolder !="":
            kcl=KCLDataSet(
                KCLfolder,
                targetfolder,
                input_size=input_size,
                depth_size=depth_size,
                transform=input_transform,
                target_transform=transform,
                full_channel_mask=full_channel_mask,
                nozero=nozero,
                train=train,
                use_resized=use_resized,
                padded=padded,
                split_config_path='/home/lvyao/git/med-ddpm/config/kcl_split.txt',
                with_condition=with_condition,
                downsample=downsample,
                )
            dataset_list.append(kcl)
        if CHNfolder !="":
            chn=CHNDataSet(
                CHNfolder,
                targetfolder,
                input_size=input_size,
                depth_size=depth_size,
                transform=input_transform,
                target_transform=transform,
                full_channel_mask=full_channel_mask,
                nozero=nozero,
                train=train,
                use_resized=use_resized,
                padded=padded,
                split_config_path='/home/lvyao/git/med-ddpm/config/chn_split.txt',
                with_condition=with_condition,
                downsample=downsample,
                )
            dataset_list.append(chn)
        dataset=CombinedDataSet(
                dataset_list
            )
        return dataset

class Subject:
    def __init__(self, folder_path,mask_path=None,device='cpu',thicknesses=None,ref_volume_path=None,age=None):
        """
        通过文件夹路径初始化Subject类
        :param folder_path: str, 文件夹路径
        """
        self.folder_path = folder_path
        self.file_list,self.mask_list = self.get_file_list(folder_path,mask_path)
        self.device=device
        self.thicknesses=thicknesses
        self._input_stacks=None
        self.ref_volume_path=ref_volume_path
        self._input_ref_volume=None
        self._age=age
    @property
    def get_name(self):
        """
        获取文件夹的最后一层名称作为name
        :return: str, 文件夹的最后一层名称
        """
        return os.path.basename(self.folder_path)

    @property
    def age(self):
        """
        提取文件夹名中的年龄信息（假设文件夹名包含年龄）
        :return: int, 年龄（单位：周）
        """
        if self._age is not None:
            return self._age
        import re
        match = re.search(r'(\d+)[Ww]', os.path.basename(self.folder_path))
        if match:
            return int(match.group(1))  # 提取数字部分并转换为整数
        else:
            match = re.search(r'(\d+)',  os.listdir(self.folder_path)[0])
            if match:
                return int(match.group(1))
            else:
                return 0

    def get_result(self, output_folder):
        """
        根据output_folder和name生成一个包含文件路径的字典作为result
        :param output_folder: str, 输出文件夹路径
        :return: dict, 包含不同路径的字典
        """
        name = self.get_name
        # 基础result文件夹路径
        result_folder = os.path.join(output_folder, name)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)
        # 生成结果路径的字典
        result_paths = {
            'volume': os.path.join(result_folder, f"{name}.nii.gz"),
            'model': os.path.join(result_folder, 'model'),
            'slice': os.path.join(result_folder, 'slice'),
            'sim-slice': os.path.join(result_folder, 'sim-slice')
        }
        return result_paths

    def get_mask_result(self, output_folder):
        """
        获取mask处理结果的文件路径
        :param output_folder: str, 输出文件夹路径
        :return: result_folder, mask_output_path
        """
        name = self.get_name
        # 基础result文件夹路径
        result_folder = os.path.join(output_folder, name)
        mask_output_path = []
        for filepath in self.file_list:  
            filename = 'mask_' + os.path.basename(filepath)
            mask_output_path.append(os.path.join(result_folder, filename))
        return result_folder, mask_output_path

    def is_processed(self, output_folder):
        """
        检查subject是否已经处理过
        :param output_folder: str, 输出文件夹路径
        :return: bool
        """
        name = self.get_name
        result_folder = os.path.join(output_folder, name)
        resample_volume = os.path.join(result_folder, "resample-vo.nii.gz")
        if os.path.exists(resample_volume) and os.path.getsize(resample_volume) > 0:
            return True
        result_volume = os.path.join(result_folder, f"{name}.nii.gz")
        if os.path.exists(result_volume) and os.path.getsize(result_volume) > 0:
            return True
        
        additional_path = os.path.join(output_folder, f"{name}.nii.gz")
        if os.path.exists(additional_path):
            return True
        return False


    def get_file_list(self, folder_path, mask_folder_path=None):
        """
        获取文件夹中所有的 .nii 或 .nii.gz 文件。如果提供了 mask_path，则加载对应的掩码文件。

        :param folder_path: str, 输入图像所在的文件夹路径
        :param mask_path: str, 可选，掩码文件所在的路径
        :return: tuple, 包含图像文件和掩码文件的列表
        """
        file_list = []  # 存储图像文件路径
        mask_list = None if mask_folder_path is None else []  # 存储掩码文件路径
        
        # 遍历文件夹中的所有文件
        root=folder_path
        files =os.listdir(folder_path)

        for file in files:
            # 检查文件扩展名是否为 .nii 或 .nii.gz
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                # 记录图像文件的完整路径
                img_path = os.path.join(root, file)
                file_list.append(img_path)

                # 如果提供了 mask_path，加载对应的掩码
                if mask_folder_path:
                    # 计算掩码文件的相对路径

                    #alter_mask_dir=os.path.join(mask_folder_path,file.split('-')[-1])
                    mask_file_path=os.path.join(mask_folder_path,file)
                    file_seg=file.replace('.nii.gz','_seg.nii.gz')
                    alter_mask_path1= os.path.join(mask_folder_path,file_seg)
                    alter_mask_path2=os.path.join(mask_file_path,f"mask_{file}")
                    # 检查掩码文件是否存在
                    if os.path.exists(mask_file_path):
                        mask_list.append(mask_file_path)
                    elif os.path.exists(alter_mask_path1):
                        mask_list.append(alter_mask_path1)
                    elif os.path.exists(alter_mask_path2):
                        mask_list.append(alter_mask_path2)
                    else:
                        # 如果没有找到对应的掩码文件，可以选择跳过，或者抛出警告
                        print(f"Warning: Mask file for {img_path} not found,except{mask_file_path}.")
        
        # 返回图像文件和掩码文件的路径列表
        return file_list, mask_list

    @property
    def input_stacks(self):
        """
        懒加载input_stacks属性：
        - 首次访问时，按路径加载所有stack并初始化厚度
        - 后续访问直接返回已加载的列表
        """
        # 1. 若未加载，则执行加载逻辑
        if self._input_stacks is None:
            self._input_stacks = []  # 初始化空列表
            from nesvor.cli.io import load_stack
            # 遍历所有stack路径，逐一生成stack对象
            for i, stack_path in enumerate(self.file_list):
                # 获取当前stack对应的mask路径（若存在）
                mask_path = None
                if self.mask_list is not None and i < len(self.mask_list):
                    mask_path = self.mask_list[i]
                
                # 加载单个stack（与参考逻辑一致）
                stack = load_stack(
                    stack_path,
                    mask_path,
                    device=self.device
                )
                
                # 若指定了厚度，赋值给stack
                if self.thicknesses is not None and i < len(self.thicknesses):
                    stack.thickness = self.thicknesses[i]
                
                stack.name+='_orientation='+self.findorientation(stack_path)
                # 将加载后的stack加入列表
                self._input_stacks.append(stack)
                    
        # 2. 已加载（或刚加载完成），返回结果
        return self._input_stacks
    def input_stacks_dir(self,direct):
        """
        懒加载input_stacks属性：
        - 首次访问时，按路径加载所有stack并初始化厚度
        - 后续访问直接返回已加载的列表
        """
        # 1. 若未加载，则执行加载逻辑
        if self._input_stacks is None:
            self._input_stacks = []  # 初始化空列表
            from nesvor.cli.io import load_stack
            # 遍历所有stack路径，逐一生成stack对象
            for i, stack_path in enumerate(self.file_list):
                # 获取当前stack对应的mask路径（若存在）
                mask_path = None
                if self.mask_list is not None and i < len(self.mask_list):
                    mask_path = self.mask_list[i]
                
                # 加载单个stack（与参考逻辑一致）
                stack = load_stack(
                    stack_path,
                    mask_path,
                    device=self.device
                )
                
                # 若指定了厚度，赋值给stack
                if self.thicknesses is not None and i < len(self.thicknesses):
                    stack.thickness = self.thicknesses[i]
                stack.name+='_orientation='+self.findorientation(stack_path)
                # 将加载后的stack加入列表
                self._input_stacks.append(stack)
        
        if direct=='random':
            return [random.choice(self._input_stacks)]
        output=[]
        for s in self._input_stacks:
            if s.name.split('_orientation=')[-1]==direct:
                output.append(s)
        # 2. 已加载（或刚加载完成），返回结果
        return output
    @staticmethod
    def findorientation(path):
        dirs=['sag','cor','tra','axi']
        for dir in dirs:
            if dir in path:
                return dir
        return 'unknown'
    @property
    def ref_volume(self):
        assert self.ref_volume_path is not None,'no ref_volume inited'
        if self._input_ref_volume is None:
            from nesvor.image import load_volume
            self._input_ref_volume=load_volume(self.ref_volume_path)
        return self._input_ref_volume
    def template(self,target_shape=None):

        from nesvor.cli.io import load_stack
        age=self.age
        template_root_dir='/home/lvyao/local/atlas/CRL_reaffined'
        template_path=os.path.join(template_root_dir,f"STA{age}exp.nii.gz" if age in range(36,39) else f"STA{age}.nii.gz")

        stack = load_stack(
                    template_path,
                    device=self.device
                )
        x=stack.slices.squeeze()
        if target_shape is None:
            return x,x>0
        D, H, W = x.shape[-3:]
        target_D, target_H, target_W = target_shape[-3:]
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
        return x_pad,x_pad>0
class FetalStackDataset:
    def __init__(self, root_dir,maskdir=None,device='cpu', is_classified=False):
        """
        初始化Dataset类
        :param root_dir: str, 数据集根目录
        :param is_classified: bool, 是否按照类别组织文件夹结构（例如，`class_name/subject_folder`）
        """
        self.root_dir = root_dir
        self.is_classified = is_classified
        self.device=device
        self.mask_path=maskdir
        self.subjects = self.load_subjects(root_dir)
        

    def load_subjects(self, root_dir):
        """
        根据根目录加载所有的subject。
        :param root_dir: str, 数据集根目录
        :return: list, 该目录下的所有Subject实例
        """
        subjects = []
        if self.is_classified:
            # 按照类别文件夹组织的结构
            for class_name in os.listdir(root_dir):
                class_folder = os.path.join(root_dir, class_name)
                if os.path.isdir(class_folder):
                    # 类别文件夹下可能有多个subject文件夹
                    for subject_folder in os.listdir(class_folder):
                        
                        subject_path = os.path.join(class_folder, subject_folder)
                        if os.path.isdir(subject_path):
                            if self.mask_path:
                                mask_class_path=os.path.join(self.mask_path, class_name)
                                mask_path=os.path.join(mask_class_path, subject_folder)
                                subjects.append(Subject(subject_path,mask_path,device=self.device))
                            else:
                                subjects.append(Subject(subject_path,device=self.device))
        else:
            # 按照直接在根目录下存放subject文件夹
            for subject_folder in os.listdir(root_dir):
                subject_path = os.path.join(root_dir, subject_folder)
                if os.path.isdir(subject_path):
                    if self.mask_path:
                        mask_path=os.path.join(self.mask_path, subject_folder)
                        subjects.append(Subject(subject_path,mask_path,device=self.device))
                    else:
                        subjects.append(Subject(subject_path,device=self.device))
        return subjects

    def get_all_subjects(self):
        """
        获取所有Subject实例
        :return: list, 所有Subject实例
        """
        return self.subjects

    def get_subject_by_name(self, name):
        """
        根据subject的名称获取对应的Subject实例
        :param name: str, subject的名称
        :return: Subject实例
        """
        for subject in self.subjects:
            if subject.get_name == name:
                return subject
        return None
    def __len__(self):
        return len(self.subjects)
    def __getitem__(self, idx):
        return self.subjects[idx]
class CRLStackDataset:
    def __init__(self, root_dir,maskdir=None,device='cpu', train=False,split_config_path='/home/lvyao/git/med-ddpm/config/crl_split.txt',ref_volume_folder=None,direction='all'):
        self.split_config_path = split_config_path
        self.ref_volume_folder=ref_volume_folder
        self.device=device
        self.train=train
        self.pair_files =[]
        self.direction=direction
        self._generate_all_week_pairs(root_dir)
        self.subjects=self.load_subjects()
    def _generate_all_week_pairs(self,root_dir):
        split_config = self._read_split_config()
        if self.train=='all':
            target_weeks =  split_config['train'] +split_config['val']
        else:
            target_weeks = split_config['train'] if self.train else split_config['val']
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            for subject_folder in os.listdir(class_folder):
                week=int(subject_folder.split('_')[0][3:5])
                week_str = f"{week}" if week in range(21,36) else f"{week}exp"
                if week_str in target_weeks:
                    stack_file=os.path.join(class_folder, subject_folder)
                    if self.ref_volume_folder:
                        volume_file=os.path.join(self.ref_volume_folder, f"STA{week_str}.nii.gz")
                        self.pair_files.append((stack_file,volume_file))
                    else:
                        self.pair_files.append(stack_file)
        return 
        
    def _read_split_config(self):
        if not os.path.exists(self.split_config_path):
            raise FileNotFoundError
        split_config = {'train': [], 'val': []}
        current_section = None
        with open(self.split_config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                # 识别section（如[train]）
                if line.startswith('[') and line.endswith(']'):
                    current_section = line.strip('[]')
                    if current_section not in split_config:
                        raise ValueError(f"无效的section：{current_section}（仅支持[train]/[val]）")
                # 识别孕周（当前section下的行）
                elif current_section:
                    split_config[current_section].append(line)

        # # 校验孕周合法性
        # valid_weeks = set(self.all_week_pairs.keys())
        # invalid_train = [w for w in split_config['train'] if w not in valid_weeks]
        # invalid_val = [w for w in split_config['val'] if w not in valid_weeks]
        # if invalid_train:
        #     raise ValueError(f"训练集包含无效孕周：{invalid_train}（仅支持{valid_weeks}）")
        # if invalid_val:
        #     raise ValueError(f"验证集包含无效孕周：{invalid_val}（仅支持{valid_weeks}）")
        
        return split_config
    def load_subjects(self):
        subjects = []
        for stack_path,volume_path in self.pair_files:
            subjects.append(Subject(stack_path,device=self.device,ref_volume_path=volume_path))
        return subjects
    def __len__(self):
        return len(self.subjects)
    def __getitem__(self, idx):
        return self.subjects[idx]
class CRLandFetaDataset:
    def __init__(self, crl_root_dir,feta_root_dir=None,device='cpu', train=False,ref_volume_folder=None,direction='all',
                 split_config_path='/home/lvyao/git/med-ddpm/config/crl_split.txt',feta_split_config_path='/home/lvyao/git/med-ddpm/config/val'):
        self.split_config_path = split_config_path
        self.ref_volume_folder=ref_volume_folder
        self.device=device
        self.train=train
        self.pair_files =[]
        self.direction=direction
        self._generate_all_week_pairs(crl_root_dir)
        self._generate_feta_pairs(feta_root_dir,feta_split_config_path)
        self.subjects=self.load_subjects()
    def _generate_all_week_pairs(self,root_dir):
        split_config = self._read_split_config()
        target_weeks = split_config['train'] if self.train else split_config['val']
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            for subject_folder in os.listdir(class_folder):
                week=int(subject_folder.split('_')[0][3:5])
                week_str = f"{week}" if week in range(21,36) else f"{week}exp"
                if week_str in target_weeks:
                    stack_file=os.path.join(class_folder, subject_folder)
                    if self.ref_volume_folder:
                        volume_file=os.path.join(self.ref_volume_folder, f"STA{week_str}.nii.gz")
                        self.pair_files.append((stack_file,volume_file))
                    else:
                        self.pair_files.append(stack_file)
        return 
    def get_age_by_id_single(self,file_path, target_id):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 跳过表头
                f.readline()
                # 逐行遍历
                for line_num, line in enumerate(f, start=2):
                    fields = line.strip().split('\t')
                    if len(fields) != 3:
                        continue
                    
                    participant_id = fields[0].strip()
                    # 找到目标ID，立即处理并返回
                    if participant_id == target_id:
                        try:
                            return int(round(float(fields[2])))
                        except ValueError:
                            print(f"错误：ID {target_id} 的孕周值不是有效数字")
                            return None
            
            # 遍历完未找到
            print(f"提示：未找到ID为'{target_id}'的记录")
            return None
        
        except FileNotFoundError:
            print(f"错误：文件'{file_path}'不存在")
            return None
    def _generate_feta_pairs(self,root_dir,split_file):
        with open(split_file, 'r') as f:
            path_names = [p.strip() for p in f.readlines()]
        for p in path_names:
            #assert int("".join(re.findall("\d", input_file))) == int("".join(re.findall("\d", target_file)))
            class_folder=os.path.join(root_dir, p)
            for  subject_folder in os.listdir(class_folder):
                stack_file=os.path.join(class_folder, subject_folder)
                age=self.get_age_by_id_single(os.path.join(root_dir,'participants.tsv'),p)
            self.pair_files.append((stack_file,age))
    def _read_split_config(self):
        if not os.path.exists(self.split_config_path):
            raise FileNotFoundError
        split_config = {'train': [], 'val': []}
        current_section = None
        with open(self.split_config_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                # 识别section（如[train]）
                if line.startswith('[') and line.endswith(']'):
                    current_section = line.strip('[]')
                    if current_section not in split_config:
                        raise ValueError(f"无效的section：{current_section}（仅支持[train]/[val]）")
                # 识别孕周（当前section下的行）
                elif current_section:
                    split_config[current_section].append(line)


        return split_config
    def load_subjects(self):
        subjects = []
        for stack_path,temp in self.pair_files:
            if isinstance(temp,int):
                subjects.append(Subject(stack_path,device=self.device,age=temp))
            else:
                subjects.append(Subject(stack_path,device=self.device,ref_volume_path=temp))
        return subjects
    def __len__(self):
        return len(self.subjects)
    def __getitem__(self, idx):
        return self.subjects[idx]
class ClinicalSimDataset:
    def __init__(self, root_dir,device='cpu', direction='all'):
        self.device=device
        self.pair_files =[]
        self.direction=direction
        self._generate_all_pairs(root_dir)
        self.subjects=self.load_subjects()
    def _generate_all_pairs(self,root_dir):
        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            for subject_folder in os.listdir(class_folder):
                stack_folder=os.path.join(class_folder, subject_folder)
                self.pair_files.append(stack_folder)
        return 
    def load_subjects(self):
        subjects = []
        for stack_path in self.pair_files:
            subjects.append(Subject(stack_path,device=self.device))
        return subjects
    def __len__(self):
        return len(self.subjects)
    def __getitem__(self, idx):
        return self.subjects[idx]
class lv_dataset(CRLStackDataset):
    def __init__(self, root_dir,maskdir=None,device='cpu', train=False,ref_volume_folder=None,direction='all'):
        self.ref_volume_folder=ref_volume_folder
        self.device=device
        self.train=train
        self.pair_files =[]
        self.direction=direction
        self._generate_all_week_pairs(root_dir,maskdir)
        self.subjects=self.load_subjects()
    def _generate_all_week_pairs(self,root_dir,maskdir):
        for subject_folder in os.listdir(root_dir):
            stack_file=os.path.join(root_dir,subject_folder)
            if maskdir is not None:
                mask_subject_folder=os.path.join(maskdir,subject_folder.split('-')[-1])
                self.pair_files.append((stack_file,mask_subject_folder))
            else:
                self.pair_files.append(stack_file)
        return 
    def load_subjects(self):
        subjects = []
        for stack_path,mask_path in self.pair_files:
            subjects.append(Subject(stack_path,mask_path,device=self.device))
        return subjects
class lv_yang_dataset(CRLStackDataset):
    def __init__(self, lv_root_dir,yang_root_dir,lv_maskdir=None,yang_maskdir=None,device='cpu', train=False,ref_volume_folder=None,direction='all'):
        self.ref_volume_folder=ref_volume_folder
        self.device=device
        self.train=train
        self.pair_files =[]
        self.direction=direction
        self._generate_all_week_pairs(lv_root_dir,lv_maskdir)
        self._generate_yang_pairs(yang_root_dir,yang_maskdir)
        self.subjects=self.load_subjects()
    def _generate_all_week_pairs(self,root_dir,maskdir):
        for subject_folder in os.listdir(root_dir):
            stack_file=os.path.join(root_dir,subject_folder)
            if maskdir is not None:
                mask_subject_folder=os.path.join(maskdir,subject_folder.split('-')[-1])
                self.pair_files.append((stack_file,mask_subject_folder))
            else:
                self.pair_files.append(stack_file)
        return 
    def _generate_yang_pairs(self,root_dir,maskdir):
        for subject_folder in os.listdir(root_dir):
            stack_file=os.path.join(root_dir,subject_folder)
            if maskdir is not None:
                mask_subject_folder=os.path.join(maskdir,subject_folder.split('-')[-1])
                self.pair_files.append((stack_file,mask_subject_folder))
            else:
                self.pair_files.append(stack_file)
        return 
    def load_subjects(self):
        subjects = []
        for stack_path,mask_path in self.pair_files:
            subjects.append(Subject(stack_path,mask_path,device=self.device))
        return subjects
class yang_dataset(CRLStackDataset):
    def __init__(self, root_dir,maskdir=None,device='cpu', train=False,ref_volume_folder=None,direction='all'):
        self.ref_volume_folder=ref_volume_folder
        self.device=device
        self.train=train
        self.pair_files =[]
        self.direction=direction
        self._generate_all_week_pairs(root_dir,maskdir)
        self.subjects=self.load_subjects()
    def _generate_all_week_pairs(self,root_dir,maskdir):
        for subject_folder in os.listdir(root_dir):
            stack_file=os.path.join(root_dir,subject_folder)
            if maskdir is not None:

                mask_file=os.path.join(maskdir, subject_folder)
                self.pair_files.append((stack_file,mask_file))
            else:
                self.pair_files.append(stack_file)
        return 
    def load_subjects(self):
        subjects = []
        for stack_path,mask_path in self.pair_files:
            subjects.append(Subject(stack_path,mask_path,device=self.device))
        return subjects