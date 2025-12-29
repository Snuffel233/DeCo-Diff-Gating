"""
DeCo-Diff 多卡并行推理脚本

支持使用多个 GPU 进行并行推理，显著加速评估过程。

用法:
    # 使用 2 个 GPU 进行评估
    torchrun --nnodes=1 --nproc_per_node=2 evaluation_DeCo_Diff_DDP.py --dataset mvtec --object-category bottle
    
    # 使用 4 个 GPU 进行评估
    torchrun --nnodes=1 --nproc_per_node=4 evaluation_DeCo_Diff_DDP.py --dataset mvtec --object-category all
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models
import argparse
import numpy as np
torch.set_grad_enabled(False)
import torch.nn.functional as F
from glob import glob

from torch.utils.data import DataLoader
from torchvision import transforms
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from scipy.ndimage import gaussian_filter

from anomalib import metrics
from sklearn.metrics import average_precision_score
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc


def setup_ddp():
    """初始化分布式环境"""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def gather_tensors(tensor, world_size):
    """从所有 GPU 收集张量"""
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def gather_numpy_arrays(local_array, world_size, rank):
    """从所有 GPU 收集 numpy 数组"""
    # 转换为 tensor
    local_tensor = torch.from_numpy(local_array).cuda()
    
    # 获取每个 rank 的数组大小
    local_size = torch.tensor([local_tensor.shape[0]], device='cuda')
    all_sizes = [torch.zeros(1, device='cuda', dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    
    max_size = max([s.item() for s in all_sizes])
    
    # Pad to max size
    if local_tensor.shape[0] < max_size:
        pad_size = max_size - local_tensor.shape[0]
        padding = torch.zeros((pad_size,) + local_tensor.shape[1:], device='cuda', dtype=local_tensor.dtype)
        local_tensor = torch.cat([local_tensor, padding], dim=0)
    
    # Gather
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    
    # Remove padding and concatenate
    result = []
    for i, (tensor, size) in enumerate(zip(gathered, all_sizes)):
        result.append(tensor[:size.item()].cpu().numpy())
    
    return np.concatenate(result, axis=0)


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR"""
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": [np.mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def calculate_metrics(ground_truth, prediction):
    flat_gt = ground_truth.flatten()
    flat_pred = prediction.flatten()
    

    auprc = metrics.AUPR()
    auprc_score = auprc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    aupro = metrics.AUPRO(fpr_limit=0.3)
    aupro_score = compute_pro(ground_truth, prediction)
    
    auroc = metrics.AUROC()
    auroc_score = auroc(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))

    f1max = metrics.F1Max()
    f1_max_score = f1max(torch.from_numpy(flat_pred), torch.from_numpy(flat_gt.astype(int)))
    
    ap = average_precision_score(ground_truth.flatten(), prediction.flatten())
    
    gt_list_sp = []
    pr_list_sp = []
    for idx in range(len(ground_truth)):
        gt_list_sp.append(np.max(ground_truth[idx]))
        sp_score = np.max(prediction[idx])
        pr_list_sp.append(sp_score)

    gt_list_sp = np.array(gt_list_sp).astype(np.int32)
    pr_list_sp = np.array(pr_list_sp)

    apsp = average_precision_score(gt_list_sp, pr_list_sp)
    aurocsp = auroc(torch.from_numpy(pr_list_sp), torch.from_numpy(gt_list_sp))
    f1sp = f1max(torch.from_numpy(pr_list_sp), torch.from_numpy(gt_list_sp))
    
    return auroc_score.numpy(), aupro_score ,f1_max_score.numpy(), ap, aurocsp.numpy(), apsp, f1sp.numpy()


def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


def calculate_anomaly_maps(x0_s, encoded_s, image_samples_s, latent_samples_s, center_size=256):
    """计算异常图"""
    pred_geometric = []
    pred_aritmetic = []
    image_differences = []
    latent_differences = []
    
    for x, encoded, image_samples, latent_samples in zip(x0_s, encoded_s, image_samples_s, latent_samples_s):
        image_difference = (((((torch.abs(image_samples-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
        image_difference = (np.clip(image_difference, 0.0, 0.4)) * 2.5
        image_difference = smooth_mask(image_difference, sigma=3)
        image_differences.append(image_difference)
        
        latent_difference = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
        latent_difference = (np.clip(latent_difference, 0.0, 0.2)) * 5
        latent_difference = smooth_mask(latent_difference, sigma=1)
        latent_difference = resize(latent_difference, (center_size, center_size))
        latent_differences.append(latent_difference)
        
        final_anomaly = image_difference * latent_difference
        final_anomaly = np.sqrt(final_anomaly)
        final_anomaly2 = 1/2*image_difference + 1/2*latent_difference
        pred_geometric.append(final_anomaly)
        pred_aritmetic.append(final_anomaly2)
            
    pred_geometric = np.stack(pred_geometric, axis=0)
    pred_aritmetic = np.stack(pred_aritmetic, axis=0)
    latent_differences = np.stack(latent_differences, axis=0)
    image_differences = np.stack(image_differences, axis=0)

    return {
        'anomaly_geometric': pred_geometric, 
        'anomaly_aritmetic': pred_aritmetic, 
        'latent_discrepancy': latent_differences, 
        'image_discrepancy': image_differences
    }


def evaluate_anomaly_maps(anomaly_maps, segmentation, rank=0):
    """评估异常图（仅在 rank 0 上执行）"""
    if rank != 0:
        return
    
    for key in anomaly_maps.keys():
        auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp = calculate_metrics(segmentation, anomaly_maps[key])
        auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp = (
            np.round(auroc_score, 4), np.round(aupro_score, 4), np.round(f1_max_score, 4), 
            np.round(ap, 4), np.round(aurocsp, 4), np.round(apsp, 4), np.round(f1sp, 4)
        )
        print('{}: auroc:{:.4f}, aupro:{:.4f}, f1_max:{:.4f}, ap:{:.4f}, aurocsp:{:.4f}, apsp:{:.4f}, f1sp:{:.4f}'.format(
            key, auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp))


def evaluation_ddp(args):
    """多卡并行评估"""
    # 初始化 DDP
    rank, world_size, device = setup_ddp()
    
    if rank == 0:
        print(f"使用 {world_size} 个 GPU 进行并行推理")
    
    # 加载模型
    vae_model = f"stabilityai/sd-vae-ft-{args.vae_type}"
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()
    
    try:
        if args.model_path != '.':
            ckpt = args.model_path
        else:
            path = f"./DeCo-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
            try:
                ckpt = sorted(glob(f'{path}/last.pt'))[-1]
            except:
                ckpt = sorted(glob(f'{path}/*/last.pt'))[-1]
    except:
        raise Exception("Please provide the trained model's path using --model_path")
    

    latent_size = int(args.center_size) // 8
    model = UNET_models[args.model_size](latent_size=latent_size, ncls=args.num_classes)
    
    state_dict = torch.load(ckpt, map_location='cpu')['model']
    if rank == 0:
        print(model.load_state_dict(state_dict, strict=False))
    else:
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.to(device)
    
    # 使用 DDP 包装模型（虽然是推理，但可以利用多卡并行）
    # 注意：推理时不需要 DDP 包装，直接使用模型即可
    
    if rank == 0:
        print('model loaded')
        print('=='*30)
        print('Starting Multi-GPU Evaluation...')
        print('=='*30)

    for category in args.categories:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
            
        # Create diffusion object
        diffusion = create_diffusion(f'ddim{args.reverse_steps}', predict_deviation=True, sigma_small=False, predict_xstart=False, diffusion_steps=10)

        # 创建数据集
        if args.dataset == 'mvtec':
            test_dataset = MVTECDataset(
                'test', object_class=category, rootdir=args.data_dir, transform=transform, 
                normal=False, anomaly_class=args.anomaly_class, 
                image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop
            )
        else:
            test_dataset = VISADataset(
                'test', object_class=category, rootdir=args.data_dir, transform=transform, 
                normal=False, anomaly_class=args.anomaly_class, 
                image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop
            )
        
        # 使用 DistributedSampler 分片数据
        sampler = DistributedSampler(
            test_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False  # 测试时不需要 shuffle
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            sampler=sampler,
            num_workers=4, 
            drop_last=False,
            pin_memory=True
        )

        # 本地结果收集
        local_encoded_s = []
        local_image_samples_s = []
        local_latent_samples_s = []
        local_x0_s = []
        local_segmentation_s = []
        
        for ii, (x, seg, object_cls) in enumerate(test_loader):
            with torch.no_grad():
                # Map input images to latent space + normalize latents
                encoded = vae.encode(x.to(device)).latent_dist.mean.mul_(0.18215)
                model_kwargs = {
                    'context': object_cls.to(device).unsqueeze(1),
                    'mask': None
                }
                latent_samples = diffusion.ddim_deviation_sample_loop(
                    model, encoded.shape, noise=encoded, clip_denoised=False, 
                    start_t=args.reverse_steps,
                    model_kwargs=model_kwargs, progress=False, device=device,
                    eta=0
                )

                image_samples = vae.decode(latent_samples / 0.18215).sample 
                x0 = vae.decode(encoded / 0.18215).sample 

            local_segmentation_s += [_seg.squeeze().numpy() for _seg in seg]
            local_encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            local_image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            local_latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_samples]
            local_x0_s += [_x0.unsqueeze(0) for _x0 in x0]

        # 计算本地异常图
        local_anomaly_maps = calculate_anomaly_maps(
            local_x0_s, local_encoded_s, local_image_samples_s, local_latent_samples_s, 
            center_size=args.center_size
        )
        local_segmentation = np.stack(local_segmentation_s, axis=0)

        # 同步所有 GPU
        dist.barrier()

        # 收集所有 GPU 的结果（仅在 rank 0 上进行最终评估）
        if rank == 0:
            print(f"\nCategory: {category}")
        
        # 收集分割标签和异常图
        all_segmentation = gather_numpy_arrays(local_segmentation, world_size, rank)
        
        all_anomaly_maps = {}
        for key in local_anomaly_maps.keys():
            all_anomaly_maps[key] = gather_numpy_arrays(local_anomaly_maps[key], world_size, rank)
        
        # 在 rank 0 上评估
        if rank == 0:
            evaluate_anomaly_maps(all_anomaly_maps, all_segmentation, rank=0)
            print('=='*30)
        
        dist.barrier()

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeCo-Diff Multi-GPU Evaluation")
    parser.add_argument("--dataset", type=str, choices=['mvtec','visa'], default="mvtec")
    parser.add_argument("--data-dir", type=str, default='./mvtec-dataset/')
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_L')
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8, help="每个 GPU 的 batch size")
    parser.add_argument("--object-category", type=str, default='all')
    parser.add_argument("--model-path", type=str, default='.')
    parser.add_argument("--anomaly-class", type=str, default='all')
    parser.add_argument("--reverse-steps", type=int, default=5)
    
    args = parser.parse_args()
    
    if args.dataset == 'mvtec':
        args.num_classes = 15
    elif args.dataset == 'visa':
        args.num_classes = 12
    
    args.results_dir = f"./DeCo-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
    if args.center_crop:
        args.results_dir += "_CenterCrop"
        args.actual_image_size = args.center_size
    else:
        args.actual_image_size = args.image_size

    if args.object_category == 'all' and args.dataset == 'mvtec':
        args.categories = [
            "bottle", "cable", "capsule", "hazelnut", "metal_nut",
            "pill", "screw", "toothbrush", "transistor", "zipper",
            "carpet", "grid", "leather", "tile", "wood",
        ]
    elif args.object_category == 'all' and args.dataset == 'visa':
        args.categories = [
            "candle", "cashew", "fryum", "macaroni2", "pcb2", "pcb4",
            "capsules", "chewinggum", "macaroni1", "pcb1", "pcb3", "pipe_fryum"
        ]
    else:
        args.categories = [args.object_category]
        
    evaluation_ddp(args)
