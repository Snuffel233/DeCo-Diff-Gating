"""
DeCo-Diff 单张图片推理和可视化脚本

用法:
    python inference_single.py --image ./test_image.jpg --model-path ./checkpoints/last.pt --output ./result.png
"""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models
from scipy.ndimage import gaussian_filter
from torchvision import transforms
import os

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def smooth_mask(mask, sigma=1.0):
    """高斯平滑"""
    return gaussian_filter(mask, sigma=sigma)


def load_image(image_path, image_size=288, center_size=256, center_crop=True):
    """加载并预处理图片"""
    img = Image.open(image_path).convert('RGB')
    
    # Resize
    img = img.resize((image_size, image_size), Image.BILINEAR)
    
    # Center crop
    if center_crop:
        left = (image_size - center_size) // 2
        top = (image_size - center_size) // 2
        img = img.crop((left, top, left + center_size, top + center_size))
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    
    return img, img_tensor


def calculate_anomaly_map(x0, encoded, image_samples, latent_samples, center_size=256):
    """计算单张图片的异常图"""
    # 图像差异
    image_difference = torch.abs(image_samples - x0).float().mean(dim=1)  # [1, H, W]
    image_difference = image_difference[0].cpu().numpy()
    image_difference = np.clip(image_difference, 0.0, 0.4) * 2.5
    image_difference = smooth_mask(image_difference, sigma=3)
    
    # Latent 差异
    latent_difference = torch.abs(latent_samples - encoded).float().mean(dim=1)  # [1, H, W]
    latent_difference = latent_difference[0].cpu().numpy()
    latent_difference = np.clip(latent_difference, 0.0, 0.2) * 5
    latent_difference = smooth_mask(latent_difference, sigma=1)
    latent_difference = resize(latent_difference, (center_size, center_size))
    
    # 几何均值融合
    anomaly_map = np.sqrt(image_difference * latent_difference)
    
    return {
        'anomaly_map': anomaly_map,
        'image_diff': image_difference,
        'latent_diff': latent_difference
    }


def visualize_results(original_img, reconstructed_img, anomaly_maps, output_path, threshold=0.3):
    """可视化结果并保存"""
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
    anomaly_map = anomaly_maps['anomaly_map']
    im = axes[0, 2].imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title('Anomaly Heatmap', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 图像差异图
    axes[1, 0].imshow(anomaly_maps['image_diff'], cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Image Difference', fontsize=12)
    axes[1, 0].axis('off')
    
    # Latent 差异图
    axes[1, 1].imshow(anomaly_maps['latent_diff'], cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Latent Difference', fontsize=12)
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
    
    # 计算异常分数
    anomaly_score = anomaly_map.max()
    is_anomalous = anomaly_score > threshold
    
    # 添加判断结果
    result_text = f"Anomaly Score: {anomaly_score:.4f}\n"
    result_text += f"Threshold: {threshold}\n"
    result_text += f"Result: {'ANOMALOUS ⚠️' if is_anomalous else 'NORMAL ✓'}"
    
    fig.suptitle(result_text, fontsize=14, fontweight='bold', 
                 color='red' if is_anomalous else 'green', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*50}")
    print(f"Anomaly Score: {anomaly_score:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Result: {'ANOMALOUS' if is_anomalous else 'NORMAL'}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*50}\n")
    
    return anomaly_score, is_anomalous


def inference(args):
    """单张图片推理"""
    print(f"Loading model from {args.model_path}...")
    
    # 加载 VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae_type}").to(device)
    vae.eval()
    
    # 加载 UNet 模型
    latent_size = args.center_size // 8
    model = UNET_models[args.model_size](latent_size=latent_size, ncls=args.num_classes)
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    print("Model loaded successfully!")
    
    # 创建 diffusion 对象
    diffusion = create_diffusion(
        f'ddim{args.reverse_steps}', 
        predict_deviation=True, 
        sigma_small=False, 
        predict_xstart=False, 
        diffusion_steps=10
    )
    
    # 加载图片
    print(f"Processing image: {args.image}")
    original_img, img_tensor = load_image(
        args.image, 
        image_size=args.image_size, 
        center_size=args.center_size,
        center_crop=args.center_crop
    )
    
    with torch.no_grad():
        # 编码到 latent 空间
        encoded = vae.encode(img_tensor.to(device)).latent_dist.mean.mul_(0.18215)
        
        # 准备模型参数
        model_kwargs = {
            'context': torch.tensor([[args.class_id]]).to(device),
            'mask': None
        }
        
        # 运行 diffusion 采样
        latent_samples = diffusion.ddim_deviation_sample_loop(
            model, encoded.shape, noise=encoded, clip_denoised=False,
            start_t=args.reverse_steps,
            model_kwargs=model_kwargs, progress=True, device=device,
            eta=0
        )
        
        # 解码回图像空间
        image_samples = vae.decode(latent_samples / 0.18215).sample
        x0 = vae.decode(encoded / 0.18215).sample
    
    # 计算异常图
    anomaly_maps = calculate_anomaly_map(
        x0, encoded, image_samples, latent_samples, 
        center_size=args.center_size
    )
    
    # 获取重建图像用于可视化
    reconstructed = image_samples[0].cpu().numpy().transpose(1, 2, 0)
    reconstructed = np.clip((reconstructed + 1) * 127.5, 0, 255).astype(np.uint8)
    reconstructed_img = Image.fromarray(reconstructed)
    
    # 可视化并保存
    output_path = args.output if args.output else args.image.replace('.', '_result.')
    anomaly_score, is_anomalous = visualize_results(
        original_img, reconstructed_img, anomaly_maps, 
        output_path, threshold=args.threshold
    )
    
    return anomaly_score, is_anomalous


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeCo-Diff Single Image Inference")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--model-path", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output", type=str, default='', help="输出结果图片路径")
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_L')
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--reverse-steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3, help="异常判断阈值（0-1之间）")
    parser.add_argument("--class-id", type=int, default=0, help="类别 ID（MVTec: 0-14, ViSA: 0-11）")
    parser.add_argument("--num-classes", type=int, default=15, help="类别总数（MVTec=15, ViSA=12）")
    
    args = parser.parse_args()
    inference(args)
