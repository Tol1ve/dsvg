#-*- coding:utf-8 -*-
# TODO: #1 cfg #2 steps #3 
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda
from diffusion_model.trainer import GaussianDiffusion, CFGDiffusion,Trainer,DEBUG
from diffusion_model.unet import create_model
from dataset import NiftiImageGenerator, create_dataset,DHCPDataSet
import argparse
import torch

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# -
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inputfolder', type=str, default="dataset/mask/")
parser.add_argument('--crldatasetfolder', type=str, default="")
parser.add_argument('--kcldatasetfolder', type=str, default="")
parser.add_argument('--chndatasetfolder', type=str, default="")
parser.add_argument('-t', '--targetfolder', type=str, default="dataset/image/")
parser.add_argument('--result', type=str, default="/home/lvyao/local/dhcp4ALDM/result")
parser.add_argument('--input_size', type=int, default=128)
parser.add_argument('--depth_size', type=int, default=128)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--num_res_blocks', type=int, default=1)
parser.add_argument('--num_class_labels', type=int, default=3)
parser.add_argument('--train_lr', type=float, default=1e-5)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50100) # epochs parameter specifies the number of training iterations
parser.add_argument('--timesteps', type=int, default=250)
parser.add_argument('--save_and_sample_every', type=int, default=5000)
parser.add_argument('--with_condition', action='store_true')
parser.add_argument('--DHCP', action='store_true')
parser.add_argument('-r', '--resume_weight', type=str, default=None)
parser.add_argument('-nz', '--no_background',action='store_true')
parser.add_argument('--resized' ,action='store_true')
parser.add_argument('--addition_volume' ,action='store_true')
parser.add_argument('--padded' ,action='store_true')
parser.add_argument('--downsample' ,action='store_true')
parser.add_argument('--evalwhiletrain' ,action='store_true')
parser.add_argument('--use_cfg' ,action='store_true')
args = parser.parse_args()

inputfolder = args.inputfolder
CRLfolder=args.crldatasetfolder
CHNfolder=args.chndatasetfolder
KCLfolder=args.kcldatasetfolder
targetfolder = args.targetfolder
input_size = args.input_size
depth_size = args.depth_size
num_channels = args.num_channels
num_res_blocks = args.num_res_blocks
num_class_labels = args.num_class_labels
save_and_sample_every = args.save_and_sample_every
with_condition = args.with_condition
resume_weight = args.resume_weight
train_lr = args.train_lr
use_dhcp=args.DHCP
result_folder=args.result
use_cfg=args.use_cfg
# input tensor: (B, 1, H, W, D)  value range: [-1, 1]
transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
])

input_transform = Compose([
    Lambda(lambda t: torch.tensor(t).float()),
    Lambda(lambda t: (t * 2) - 1),
    Lambda(lambda t: t.unsqueeze(0)),
])

if with_condition:
    if use_dhcp:
        dataset=DHCPDataSet(
            inputfolder,
            targetfolder,
            input_size=input_size,
            depth_size=depth_size,
            transform=input_transform if with_condition else transform,
            target_transform=transform,
            full_channel_mask=True,
            train=True,
        )
    else:
        dataset=create_dataset(
                inputfolder=inputfolder,CRLfolder=CRLfolder,KCLfolder=KCLfolder,CHNfolder=CHNfolder,\
                targetfolder=targetfolder,          
                input_size=input_size,
                depth_size=depth_size,
                transform=transform,
                input_transform=input_transform if with_condition else transform,
                full_channel_mask=True,
                nozero=args.no_background,
                train=True,
                use_resized=args.resized,
                addition_volume=args.addition_volume,
                padded=args.padded,
                downsample=args.downsample,
        )
        eval_dataset=create_dataset(
                inputfolder=inputfolder,
                CRLfolder=CRLfolder,
                KCLfolder=KCLfolder,
                CHNfolder=CHNfolder,
                targetfolder=targetfolder,          
                input_size=input_size,
                depth_size=depth_size,
                transform=transform,
                input_transform=input_transform if with_condition else transform,
                full_channel_mask=True,
                nozero=args.no_background,
                train=False,
                use_resized=args.resized,
                addition_volume=args.addition_volume,
                padded=args.padded,
                downsample=args.downsample,
        ) if args.evalwhiletrain else None
else:
    dataset=create_dataset(
                inputfolder=inputfolder,CRLfolder=CRLfolder,KCLfolder=KCLfolder,CHNfolder=CHNfolder,\
                targetfolder=targetfolder,          
                input_size=input_size,
                depth_size=depth_size,
                transform=transform,
                input_transform=input_transform if with_condition else transform,
                full_channel_mask=True,
                nozero=args.no_background,
                train=True,
                use_resized=args.resized,
                addition_volume=args.addition_volume,
                padded=args.padded,
                downsample=args.downsample,
                with_condition=with_condition,
        )
    eval_dataset=create_dataset(
                inputfolder=inputfolder,
                CRLfolder=CRLfolder,
                KCLfolder=KCLfolder,
                CHNfolder=CHNfolder,
                targetfolder=targetfolder,          
                input_size=input_size,
                depth_size=depth_size,
                transform=transform,
                input_transform=input_transform if with_condition else transform,
                full_channel_mask=True,
                nozero=args.no_background,
                train=False,
                use_resized=args.resized,
                addition_volume=args.addition_volume,
                padded=args.padded,
                with_condition=with_condition
        ) if args.evalwhiletrain else None
    # dataset = NiftiImageGenerator(
    #     inputfolder,
    #     input_size=input_size,
    #     depth_size=depth_size,
    #     transform=transform
    # )

print(len(dataset))

in_channels = num_class_labels if with_condition else 1
out_channels = 1 
model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels).cuda()
if use_cfg:
    diffusion = CFGDiffusion(
        model,
        image_size = input_size,
        depth_size = depth_size,
        timesteps = args.timesteps,   # number of steps
        loss_type = 'l1',    # L1 or L2
        with_condition=with_condition,
        channels=out_channels
    ).cuda()
else:
    diffusion = GaussianDiffusion(
        model,
        image_size = input_size,
        depth_size = depth_size,
        timesteps = args.timesteps,   # number of steps
        loss_type = 'l1',    # L1 or L2
        with_condition=with_condition,
        channels=out_channels
    ).cuda()

if resume_weight is not None and len(resume_weight) > 0:
    weight = torch.load(resume_weight, map_location='cuda')
    diffusion.load_state_dict(weight['ema'])
    print("Model Loaded!")

trainer = Trainer(
    diffusion,
    dataset,
    eval_dataset,
    image_size = input_size,
    depth_size = depth_size,
    train_batch_size = args.batchsize,
    train_lr = train_lr,
    train_num_steps = args.epochs,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False,#True,                       # turn on mixed precision training with apex
    with_condition=with_condition,
    save_and_sample_every = save_and_sample_every,
    results_folder=result_folder,
    use_cfg=use_cfg,
)
if DEBUG:
    trainer.sample_as_train()
trainer.train()
