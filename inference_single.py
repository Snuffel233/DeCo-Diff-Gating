"""
DeCo-Diff 单张图片推理和可视化脚本

支持门控机制（Gating）的异常检测，可以提取并可视化 DoD-Gating 和 Skip-Gating 的输出。

用法:
    python inference_single.py --image ./test_image.jpg --model-path ./checkpoints/last.pt --output ./result.png
    
    # 使用门控融合
    python inference_single.py --image ./test_image.jpg --model-path ./checkpoints/last.pt --use-gate-fusion true
"""

import torch
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models
import argparse
import numpy as np
torch.set_grad_enabled(False)
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

from torchvision import transforms
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import os


def smooth_mask(mask, sigma=1.0):
    """高斯平滑"""
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask


def extract_gate_maps(model, encoded_batch, t, model_kwargs, device='cuda'):
    """
    从模型中提取门控图。
    
    Args:
        model: UNet 模型
        encoded_batch: 编码后的 latent 批次
        t: 时间步
        model_kwargs: 模型参数
        device: 设备
    
    Returns:
        dod_gate: DoD 门控图
        skip_gates: Skip 门控图列表
    """
    with torch.no_grad():
        # 调用模型并请求返回门控图
        output = model(encoded_batch, t, return_gate=True, **model_kwargs)
        if isinstance(output, tuple) and len(output) >= 2:
            _, dod_gate, skip_gates = output
            return dod_gate, skip_gates
        else:
            return None, None


def calculate_anomaly_map(x0, encoded, image_samples, latent_samples, gate_map=None, center_size=256):
    """
    计算单张图片的异常图。
    
    Args:
        x0: 原始重建图像
        encoded: 编码后的 latent
        image_samples: 采样后的图像
        latent_samples: 采样后的 latent
        gate_map: 可选的门控图（来自 DoD-Gating）
        center_size: 中心裁剪尺寸
    
    Returns:
        包含多种异常图的字典
    """
    # 图像差异
    image_difference = (((((torch.abs(image_samples-x0))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
    image_difference = (np.clip(image_difference, 0.0, 0.4)) * 2.5
    image_difference = smooth_mask(image_difference, sigma=3)
    
    # Latent 差异
    latent_difference = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
    latent_difference = (np.clip(latent_difference, 0.0, 0.2)) * 5
    latent_difference = smooth_mask(latent_difference, sigma=1)
    latent_difference = resize(latent_difference, (center_size, center_size))
    
    # 几何均值融合
    final_anomaly = image_difference * latent_difference
    final_anomaly = np.sqrt(final_anomaly)
    
    # 算术均值
    final_anomaly2 = 0.5 * image_difference + 0.5 * latent_difference
    
    result = {
        'anomaly_geometric': final_anomaly,
        'anomaly_aritmetic': final_anomaly2,
        'image_discrepancy': image_difference,
        'latent_discrepancy': latent_difference
    }
    
    # 如果有门控图，添加门控融合结果
    if gate_map is not None:
        # 将门控图处理成异常图格式
        gate_map_np = gate_map[0].mean(dim=0).detach().cpu().numpy()  # 平均通道
        gate_map_np = resize(gate_map_np, (center_size, center_size))
        gate_map_np = smooth_mask(gate_map_np, sigma=2)
        
        # 门控加权融合
        gate_fused = 0.6 * final_anomaly + 0.4 * gate_map_np
        result['gate_map'] = gate_map_np
        result['gate_fused'] = gate_fused
    
    return result


def visualize_results(original_img, reconstructed_img, anomaly_maps, output_path, threshold=0.3, use_gate=False):
    """
    可视化结果并保存。
    
    Args:
        original_img: 原始图片 (PIL Image)
        reconstructed_img: 重建图片 (PIL Image)
        anomaly_maps: 异常图字典
        output_path: 输出路径
        threshold: 异常判断阈值
        use_gate: 是否使用门控图
    """
    if use_gate and 'gate_map' in anomaly_maps:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图片
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Input Image', fontsize=12)
    axes[0, 0].axis('off')
    
    # 重建图片
    axes[0, 1].imshow(reconstructed_img)
    axes[0, 1].set_title('Reconstructed Image', fontsize=12)
    axes[0, 1].axis('off')
    
    # 异常图（热力图）
    anomaly_map = anomaly_maps['anomaly_geometric']
    im = axes[0, 2].imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title('Anomaly Heatmap (Geometric)', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 图像差异图
    axes[1, 0].imshow(anomaly_maps['image_discrepancy'], cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Image Discrepancy', fontsize=12)
    axes[1, 0].axis('off')
    
    # Latent 差异图
    axes[1, 1].imshow(anomaly_maps['latent_discrepancy'], cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Latent Discrepancy', fontsize=12)
    axes[1, 1].axis('off')
    
    # 异常检测叠加图
    original_array = np.array(original_img.resize((anomaly_map.shape[1], anomaly_map.shape[0])))
    overlay = original_array.copy().astype(float) / 255.0
    
    # 将异常区域标红
    mask = anomaly_map > threshold
    overlay[mask, 0] = np.clip(overlay[mask, 0] + anomaly_map[mask] * 0.5, 0, 1)
    overlay[mask, 1] = overlay[mask, 1] * 0.5
    overlay[mask, 2] = overlay[mask, 2] * 0.5
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f'Anomaly Overlay (threshold={threshold})', fontsize=12)
    axes[1, 2].axis('off')
    
    # 如果使用门控图
    if use_gate and 'gate_map' in anomaly_maps:
        # 门控图
        axes[0, 3].imshow(anomaly_maps['gate_map'], cmap='jet', vmin=0, vmax=1)
        axes[0, 3].set_title('DoD Gate Map', fontsize=12)
        axes[0, 3].axis('off')
        
        # 门控融合图
        gate_fused = anomaly_maps['gate_fused']
        im2 = axes[1, 3].imshow(gate_fused, cmap='jet', vmin=0, vmax=1)
        axes[1, 3].set_title('Gate Fused Anomaly', fontsize=12)
        axes[1, 3].axis('off')
        plt.colorbar(im2, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        # 使用门控融合的异常分数
        anomaly_score = gate_fused.max()
    else:
        anomaly_score = anomaly_map.max()
    
    is_anomalous = anomaly_score > threshold
    
    # 添加判断结果
    result_text = f"Anomaly Score: {anomaly_score:.4f} | "
    result_text += f"Threshold: {threshold} | "
    result_text += f"Result: {'ANOMALOUS ⚠️' if is_anomalous else 'NORMAL ✓'}"
    
    fig.suptitle(result_text, fontsize=14, fontweight='bold', 
                 color='red' if is_anomalous else 'green', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Anomaly Score: {anomaly_score:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Result: {'ANOMALOUS' if is_anomalous else 'NORMAL'}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return anomaly_score, is_anomalous


def load_and_preprocess_image(image_path, image_size=288, center_size=256, center_crop=True):
    """
    加载并预处理图片。
    
    Args:
        image_path: 图片路径
        image_size: 目标尺寸
        center_size: 中心裁剪尺寸
        center_crop: 是否中心裁剪
        
    Returns:
        original_img: PIL Image
        img_tensor: 预处理后的 tensor
    """
    original_img = Image.open(image_path).convert('RGB')
    
    # Resize
    img = original_img.resize((image_size, image_size), Image.BILINEAR)
    
    # Center crop
    if center_crop:
        left = (image_size - center_size) // 2
        top = (image_size - center_size) // 2
        img = img.crop((left, top, left + center_size, top + center_size))
    
    # Transform（与评估代码一致）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    
    return img, img_tensor


def inference(args):
    """单张图片推理"""
    print(f"Device: {device}")
    print(f"Loading model from {args.model_path}...")
    
    # 加载 VAE
    vae_model = f"stabilityai/sd-vae-ft-{args.vae_type}"
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()
    
    # 加载 UNet 模型
    latent_size = int(args.center_size) // 8
    model = UNET_models[args.model_size](latent_size=latent_size, ncls=args.num_classes)
    
    state_dict = torch.load(args.model_path, map_location='cpu',weights_only=False)['model']
    print(model.load_state_dict(state_dict, strict=False))
    model.eval()
    model.to(device)
    print('Model loaded successfully!')
    
    # 创建 diffusion 对象（与评估代码一致）
    diffusion = create_diffusion(
        f'ddim{args.reverse_steps}', 
        predict_deviation=True, 
        sigma_small=False, 
        predict_xstart=False, 
        diffusion_steps=10
    )
    
    # 加载并预处理图片
    print(f"Processing image: {args.image}")
    original_img, img_tensor = load_and_preprocess_image(
        args.image, 
        image_size=args.image_size, 
        center_size=args.center_size,
        center_crop=args.center_crop
    )
    
    with torch.no_grad():
        # Map input images to latent space + normalize latents（与评估代码一致）
        encoded = vae.encode(img_tensor.to(device)).latent_dist.mean.mul_(0.18215)
        
        model_kwargs = {
            'context': torch.tensor([[args.class_id]]).to(device),
            'mask': None
        }
        
        # 运行 DDIM deviation sampling（与评估代码一致）
        latent_samples = diffusion.ddim_deviation_sample_loop(
            model, encoded.shape, noise=encoded, clip_denoised=False,
            start_t=args.reverse_steps,
            model_kwargs=model_kwargs, progress=True, device=device,
            eta=0
        )
        
        # 提取门控图（如果启用）
        gate_map = None
        if args.use_gate_fusion:
            t = torch.zeros(encoded.shape[0], dtype=torch.long, device=device)
            gate_map, skip_gates = extract_gate_maps(model, encoded, t, model_kwargs, device)
            if gate_map is not None:
                print(f"Gate map extracted, shape: {gate_map.shape}")
        
        # 解码回图像空间
        image_samples = vae.decode(latent_samples / 0.18215).sample
        x0 = vae.decode(encoded / 0.18215).sample
    
    # 计算异常图
    anomaly_maps = calculate_anomaly_map(
        x0, encoded, image_samples, latent_samples,
        gate_map=gate_map,
        center_size=args.center_size
    )
    
    # 获取重建图像用于可视化
    reconstructed = image_samples[0].cpu().numpy().transpose(1, 2, 0)
    reconstructed = np.clip((reconstructed + 1) * 127.5, 0, 255).astype(np.uint8)
    reconstructed_img = Image.fromarray(reconstructed)
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.image)
        output_path = f"{base}_result.png"
    
    # 可视化并保存
    anomaly_score, is_anomalous = visualize_results(
        original_img, reconstructed_img, anomaly_maps,
        output_path, threshold=args.threshold,
        use_gate=args.use_gate_fusion
    )
    
    return anomaly_score, is_anomalous


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeCo-Diff Single Image Inference with Gating Support")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--model-path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output", type=str, default='', help="输出结果图片路径（默认在输入图片同目录生成）")
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_L')
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--reverse-steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3, help="异常判断阈值（0-1之间）")
    parser.add_argument("--class-id", type=int, default=0, help="类别 ID（MVTec: 0-14, ViSA: 0-11）")
    parser.add_argument("--num-classes", type=int, default=15, help="类别总数（MVTec=15, ViSA=12）")
    parser.add_argument("--use-gate-fusion", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=False,
                        help="是否使用门控图融合进行异常检测（需要使用带 Gating 的模型）")
    
    args = parser.parse_args()
    
    if args.center_crop:
        args.actual_image_size = args.center_size
    else:
        args.actual_image_size = args.image_size
    
    inference(args)
