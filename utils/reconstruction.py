from argparse import Namespace
from typing import List, Tuple
import time
import datetime
import torch
import torch.optim as optim
import logging
from nesvor.utils import MovingAverage, log_params, TrainLogger,resolution2sigma
from nesvor.inr.models import INR, NeSVoR,TwNeSVoR, NewNeSVoR,D_LOSS, S_LOSS, I_REG
from nesvor.transform import RigidTransform
from nesvor.image import Volume, Slice
from nesvor.inr.data import PointDataset,VolumeDataset,VolumeStackDataset,VolumeEncodeDataset,SimPointDataset
from nesvor.inr.sample import sample_points
from vfm.volume_dataset import VFMINRModel
def dataset_property(dataset):
    use_scaling = False
    use_centering = False
    # perform centering and scaling
    spatial_scaling = 30.0 if use_scaling else 1
    bb = dataset.bounding_box
    center = (bb[0] + bb[-1]) / 2 if use_centering else torch.zeros_like(bb[0])
    ax = (
        RigidTransform(torch.cat([torch.zeros_like(center), -center])[None])
        .compose(dataset.transformation)
        .axisangle()
    )
    ax[:, -3:] /= spatial_scaling
    transformation = RigidTransform(ax)

    boundding_box=(bb - center) / spatial_scaling
    resolution=dataset.resolution

    return transformation,boundding_box,resolution,spatial_scaling
def INR_super_resolution(slices: List[Slice],volume, args: Namespace,scale=2) -> Tuple[INR, List[Slice], Volume]:
    # create training dataset
    args.device=torch.device(f"cuda:{args.device}")  if isinstance(args.device,int) else args.device
    args.dtype=torch.float32 if args.dtype=="f32" else torch.float16
    if args.use_inr =='err_sim' :
        dataset = VolumeStackDataset(volume,slices,scale_factor=scale)
    elif args.use_inr =='Tw' or args.use_inr =='new_sim' :
        dataset =VolumeEncodeDataset(volume,slices,scale_factor=scale)
    else:
        dataset = VolumeDataset(volume)
    if getattr(args,'n_epochs',None) and args.n_epochs is not None:
        args.n_iter = args.n_epochs * (dataset.v.numel() // args.batch_size)
    transformation,boundding_box,resolution,spatial_scaling=dataset_property(dataset)
    dataset.xyz /= spatial_scaling
    if args.use_inr =='err_sim':
        model = NeSVoR(
            transformation,
            dataset.resolution / spatial_scaling,
            dataset.mean,
            boundding_box,
            spatial_scaling,
            args,
        )
    elif args.use_inr =='Tw':
        model=TwNeSVoR(
            transformation,
            dataset.resolution / spatial_scaling,
            dataset.mean,
            boundding_box,
            spatial_scaling,
            args,
        )
    elif args.use_inr =='new_sim':
        model=NewNeSVoR(
            transformation,
            dataset.resolution / spatial_scaling,
            dataset.mean,
            boundding_box,
            spatial_scaling,
            args,
        )
    else:
        model = VFMINRModel(args,dataset.bounding_box)
    # setup optimizer
    params_net = []
    params_encoding = []
    for name, param in model.named_parameters():
        if param.numel() > 0:
            if "_net" in name:
                params_net.append(param)
            else:
                params_encoding.append(param)
    # logging
    logging.debug(log_params(model))
    optimizer = torch.optim.AdamW(
        params=[
            {"name": "encoding", "params": params_encoding},
            {"name": "net", "params": params_net, "weight_decay": 1e-2},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    # setup scheduler for lr decay
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(1, len(args.milestones) + 1)),
        gamma=args.gamma,
    )
    decay_milestones = [int(m * args.n_iter) for m in args.milestones]
    # setup grad scalar for mixed precision training
    fp16 = not args.single_precision
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1.0,
        enabled=fp16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )
    # training
    model.train()
    loss_weights = {
        D_LOSS: 1,
        S_LOSS: 1,
        I_REG: 0.0001 if  args.use_inr =='stackaftervolume' else args.weight_image,

    }
    average = MovingAverage(1 - 0.001)
    # logging
    logging_header = False
    logging.info("NeSVoR training starts.")
    train_time = 0.0
    for i in range(1, args.n_iter + 1):
        train_step_start = time.time()
        # forward
        batch = dataset.get_batch(args.batch_size, args.device)
        with torch.cuda.amp.autocast(fp16):
            losses = model(**batch)
            loss = 0
            for k in losses:
                if k in loss_weights and loss_weights[k]:
                    if k == D_LOSS and args.use_inr =='new_sim':
                        alpha= 1 if i<args.n_iter//3 else ( (args.n_iter//10-i//10)/(args.n_iter//10)  if i<args.n_iter*2//3 else 0)
                        loss = loss + loss_weights[ D_LOSS] *( (1-alpha)*losses[D_LOSS]+alpha*losses['sim_LOSS'])
                    else:
                        loss = loss + loss_weights[k] * losses[k]
        # backward
        scaler.scale(loss).backward()
        if args.debug:  # check nan grad
            for _name, _p in model.named_parameters():
                if _p.grad is not None and not _p.grad.isfinite().all():
                    logging.warning("iter %d: Found NaNs in the grad of %s", i, _name)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_time += time.time() - train_step_start
        for k in losses:
            average(k, losses[k].item())
        if (decay_milestones and i >= decay_milestones[0]) or i == args.n_iter:
            # logging
            if not logging_header:
                train_logger = TrainLogger(
                    "time",
                    "epoch",
                    "iter",
                    *list(losses.keys()),
                    "lr",
                )
                logging_header = True
            train_logger.log(
                datetime.timedelta(seconds=int(train_time)),
                dataset.epoch,
                i,
                *[average[k] for k in losses],
                optimizer.param_groups[0]["lr"],
            )
            if i < args.n_iter:
                decay_milestones.pop(0)
                scheduler.step()
            # check scaler
            if scaler.is_enabled():
                current_scaler = scaler.get_scale()
                if current_scaler < 1 / (2**5):
                    logging.warning(
                        "Numerical instability detected! "
                        "The scale of GradScaler is %f, which is too small. "
                        "The results might be suboptimal. "
                        "Try to set --single-precision or run the command again with a different random seed."% current_scaler
                    )
                if i == args.n_iter:
                    logging.debug("Final scale of GradScaler = %f" % current_scaler)
    model.inr.bounding_box.copy_(dataset.bounding_box)
    dataset.xyz *= spatial_scaling
    mask = dataset.mask
    if args.use_inr =='err_sim'  :
        err_volume= sample_volume(
            model.inr,
            mask,
            args.output_resolution * args.output_psf_factor,
            args.inference_batch_size,
            args.n_inference_samples,
        )
        err_volume.image+=mask.image
        return model.inr,err_volume,mask 
    elif  args.use_inr =='new_sim':
        output_volume= sample_volume(
            model.inr,
            mask,
            args.output_resolution * args.output_psf_factor,
            args.inference_batch_size,
            args.n_inference_samples,
        )
        return model.inr,output_volume,mask 
    elif args.use_inr =='Tw':
        output_volume= sample_volume_atlas(
            model.inr,
            mask,
            args.output_resolution * args.output_psf_factor,
            args.inference_batch_size,
            args.n_inference_samples,
        )
        return model.inr,output_volume,mask 
    
    else:
        output_volume = sample_volume(
            model.inr,
            mask,
            args.output_resolution * args.output_psf_factor,
            args.inference_batch_size,
            args.n_inference_samples,
        )
        return model.inr,output_volume,mask 

def optimization_inr(volume,slices,inr,args):
    dataset = PointDataset(slices)
    # args.device=torch.device(f"cuda:{args.device}")  if isinstance(args.device,int) else args.device
    # args.dtype=torch.float32 if args.dtype=="f32"  else torch.float16
    #dataset = SimPointDataset(slices,volume)
    transformation,bounding_box,resolution,spatial_scaling=dataset_property(dataset)
    dataset.xyz /= spatial_scaling
    model = NeSVoR(
            transformation,
            dataset.resolution / spatial_scaling,
            dataset.mean,
            inr.bounding_box if inr is not None else bounding_box ,
            spatial_scaling,
            args,
            inr=inr,
        )
    params_net = []
    params_encoding = []
    for name, param in model.named_parameters():
        if param.numel() > 0:
            if "_net" in name:
                params_net.append(param)
            else:
                params_encoding.append(param)
    # logging
    logging.debug(log_params(model))
    optimizer = torch.optim.AdamW(
        params=[
            {"name": "encoding", "params": params_encoding},
            {"name": "net", "params": params_net, "weight_decay": 1e-2},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        eps=1e-15,
    )
    # setup scheduler for lr decay
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(range(1, len(args.milestones) + 1)),
        gamma=args.gamma,
    )
    decay_milestones = [int(m * args.n_iter) for m in args.milestones]
    # setup grad scalar for mixed precision training
    fp16 = not args.single_precision
    scaler = torch.cuda.amp.GradScaler(
        init_scale=1.0,
        enabled=fp16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    )
    # training
    model.train()
    loss_weights = {
        D_LOSS: 1,
        S_LOSS: 1,
        I_REG: args.weight_image,

    }
    average = MovingAverage(1 - 0.001)
    # logging
    logging_header = False
    logging.info("NeSVoR training starts.")
    train_time = 0.0
    for i in range(1, args.n_iter + 1):
        train_step_start = time.time()
        # forward
        batch = dataset.get_batch(args.batch_size, args.device)
        with torch.cuda.amp.autocast(fp16):
            losses = model(**batch)
            loss = 0
            for k in losses:
                if k in loss_weights and loss_weights[k]:
                    loss = loss + loss_weights[k] * losses[k]
        # backward
        scaler.scale(loss).backward()
        if args.debug:  # check nan grad
            for _name, _p in model.named_parameters():
                if _p.grad is not None and not _p.grad.isfinite().all():
                    logging.warning("iter %d: Found NaNs in the grad of %s", i, _name)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        train_time += time.time() - train_step_start
        for k in losses:
            average(k, losses[k].item())
        if (decay_milestones and i >= decay_milestones[0]) or i == args.n_iter:
            # logging
            if not logging_header:
                train_logger = TrainLogger(
                    "time",
                    "epoch",
                    "iter",
                    *list(losses.keys()),
                    "lr",
                )
                logging_header = True
            train_logger.log(
                datetime.timedelta(seconds=int(train_time)),
                dataset.epoch,
                i,
                *[average[k] for k in losses],
                optimizer.param_groups[0]["lr"],
            )
            if i < args.n_iter:
                decay_milestones.pop(0)
                scheduler.step()
            # check scaler
            if scaler.is_enabled():
                current_scaler = scaler.get_scale()
                if current_scaler < 1 / (2**5):
                    logging.warning(
                        "Numerical instability detected! "
                        "The scale of GradScaler is %f, which is too small. "
                        "The results might be suboptimal. "
                        "Try to set --single-precision or run the command again with a different random seed."% current_scaler
                    )
                if i == args.n_iter:
                    logging.debug("Final scale of GradScaler = %f" % current_scaler)
    #model.inr.bounding_box.copy_(dataset.bounding_box)
    dataset.xyz *= spatial_scaling
    mask = volume
    
    output_volume= sample_volume(
            model.inr,
            mask,
            args.output_resolution * args.output_psf_factor,
            args.inference_batch_size,
            args.n_inference_samples,
        )
    return model.inr,output_volume,mask 


def sample_volume(
    model: INR,
    mask: Volume,
    psf_resolution: float,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> Volume:
    model.eval()
    img = mask.clone()
    img.image[img.mask] = sample_points(
        model,
        img.xyz_masked,
        psf_resolution,
        batch_size,
        n_samples,
    )
    return img

def sample_points_atlas(
    model: INR,
    xyz: torch.Tensor,
    sim_v:torch.Tensor,
    resolution: float = 0,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> torch.Tensor:
    shape = xyz.shape[:-1]
    xyz = xyz.view(-1, 3)
    v = torch.empty(xyz.shape[0], dtype=torch.float32, device=xyz.device)
    with torch.no_grad():
        for i in range(0, xyz.shape[0], batch_size):
            xyz_batch = xyz[i : i + batch_size]
            v_batch=sim_v[i : i + batch_size]
            xyz_batch = model.sample_batch(
                xyz_batch,
                None,
                resolution2sigma(resolution, isotropic=True),
                0 if resolution <= 0 else n_samples,
            )
            v_batch=v_batch[:,None].repeat(1,n_samples,1)
            
            v_b = model(xyz_batch,v_batch).mean(-1)

            v[i : i + batch_size] = v_b
    return v.view(shape)
def sample_volume_atlas(
    model: INR,
    mask: Volume,
    psf_resolution: float,
    batch_size: int = 1024,
    n_samples: int = 128,
) -> Volume:
    model.eval()
    img = mask.clone()

    img.image[img.mask] = sample_points_atlas(
        model,
        img.xyz_masked,
        img.v_masked[:,None],
        psf_resolution,
        batch_size,
        n_samples,
    )
    return img