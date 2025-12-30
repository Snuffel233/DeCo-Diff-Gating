"""
DeCo-Diff 自动类别检测推理脚本

结合分类器模型自动识别图像类别，无需手动指定 class_id。

用法:
    # 自动检测类别
    python inference_auto.py --image ./test.jpg --model-path ./checkpoints/last.pt --classifier-path ./classifier_mvtec.pth
    
    # 批量处理
    python inference_auto.py --image-dir ./test_images/ --model-path ./checkpoints/last.pt --classifier-path ./classifier_mvtec.pth
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

from torchvision import transforms, models
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from glob import glob
import torch.nn as nn

# ===================== 类别映射 =====================
MVTEC_CLASS_MAP = {
    "capsule": 0, "bottle": 1, "grid": 2, "leather": 3, "metal_nut": 4,
    "tile": 5, "transistor": 6, "zipper": 7, "cable": 8, "carpet": 9,
    "hazelnut": 10, "pill": 11, "screw": 12, "toothbrush": 13, "wood": 14,
}
ID_TO_CLASS = {v: k for k, v in MVTEC_CLASS_MAP.items()}
NUM_CLASSES = len(MVTEC_CLASS_MAP)


# ===================== 分类器模型 =====================
class MVTecClassifier(nn.Module):
    """基于 ResNet 的分类器"""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ===================== 工具函数 =====================
def smooth_mask(mask, sigma=1.0):
    return gaussian_filter(mask, sigma=sigma)


def extract_gate_maps(model, encoded_batch, t, model_kwargs, device='cuda'):
    with torch.no_grad():
        output = model(encoded_batch, t, return_gate=True, **model_kwargs)
        if isinstance(output, tuple) and len(output) >= 2:
            _, dod_gate, skip_gates = output
            return dod_gate, skip_gates
    return None, None


def calculate_anomaly_map(x0, encoded, image_samples, latent_samples, gate_map=None, center_size=256):
    """计算异常图"""
    image_difference = (((((torch.abs(image_samples-x0))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
    image_difference = (np.clip(image_difference, 0.0, 0.4)) * 2.5
    image_difference = smooth_mask(image_difference, sigma=3)
    
    latent_difference = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
    latent_difference = (np.clip(latent_difference, 0.0, 0.2)) * 5
    latent_difference = smooth_mask(latent_difference, sigma=1)
    latent_difference = resize(latent_difference, (center_size, center_size))
    
    final_anomaly = np.sqrt(image_difference * latent_difference)
    
    result = {
        'anomaly_geometric': final_anomaly,
        'image_discrepancy': image_difference,
        'latent_discrepancy': latent_difference
    }
    
    if gate_map is not None:
        gate_map_np = gate_map[0].mean(dim=0).detach().cpu().numpy()
        gate_map_np = resize(gate_map_np, (center_size, center_size))
        gate_map_np = smooth_mask(gate_map_np, sigma=2)
        result['gate_map'] = gate_map_np
        result['gate_fused'] = 0.6 * final_anomaly + 0.4 * gate_map_np
    
    return result


def visualize_results(original_img, reconstructed_img, anomaly_maps, output_path, 
                      threshold, use_gate, class_name, class_confidence):
    """可视化结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f'输入图像\n检测类别: {class_name} ({class_confidence:.1%})', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reconstructed_img)
    axes[0, 1].set_title('重建图像', fontsize=11)
    axes[0, 1].axis('off')
    
    anomaly_map = anomaly_maps.get('gate_fused', anomaly_maps['anomaly_geometric']) if use_gate else anomaly_maps['anomaly_geometric']
    im = axes[0, 2].imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].set_title('异常热力图', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    axes[1, 0].imshow(anomaly_maps['image_discrepancy'], cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('图像差异', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(anomaly_maps['latent_discrepancy'], cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Latent差异', fontsize=11)
    axes[1, 1].axis('off')
    
    # 异常叠加
    original_array = np.array(original_img.resize((anomaly_map.shape[1], anomaly_map.shape[0])))
    overlay = original_array.copy().astype(float) / 255.0
    mask = anomaly_map > threshold
    overlay[mask, 0] = np.clip(overlay[mask, 0] + anomaly_map[mask] * 0.5, 0, 1)
    overlay[mask, 1] = overlay[mask, 1] * 0.5
    overlay[mask, 2] = overlay[mask, 2] * 0.5
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f'异常区域 (阈值={threshold:.2f})', fontsize=11)
    axes[1, 2].axis('off')
    
    anomaly_score = anomaly_map.max()
    is_anomalous = anomaly_score > threshold
    
    result_text = f"类别: {class_name} | 异常分数: {anomaly_score:.4f} | 阈值: {threshold} | 结果: {'异常 ⚠️' if is_anomalous else '正常 ✓'}"
    fig.suptitle(result_text, fontsize=13, fontweight='bold', 
                 color='red' if is_anomalous else 'green', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return anomaly_score, is_anomalous


class AutoInference:
    """自动推理类"""
    
    def __init__(self, model_path, classifier_path, model_size='UNet_L', 
                 vae_type='ema', center_size=256, reverse_steps=5):
        self.device = device
        self.center_size = center_size
        self.reverse_steps = reverse_steps
        
        print("正在加载模型...")
        
        # 加载分类器
        print("  加载分类器...")
        self.classifier = MVTecClassifier(NUM_CLASSES, pretrained=False).to(device)
        classifier_ckpt = torch.load(classifier_path, map_location='cpu', weights_only=False)
        self.classifier.load_state_dict(classifier_ckpt['model_state_dict'])
        self.classifier.eval()
        print(f"  分类器加载完成 (准确率: {classifier_ckpt.get('best_acc', 'N/A')}%)")
        
        # 分类器预处理
        self.classifier_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载 VAE
        print("  加载 VAE...")
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
        self.vae.eval()
        
        # 加载 DeCo-Diff 模型
        print("  加载 DeCo-Diff 模型...")
        latent_size = center_size // 8
        self.model = UNET_models[model_size](latent_size=latent_size, ncls=NUM_CLASSES)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)['model']
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model.to(device)
        
        # Diffusion
        self.diffusion = create_diffusion(
            f'ddim{reverse_steps}', predict_deviation=True, 
            sigma_small=False, predict_xstart=False, diffusion_steps=10
        )
        
        # DeCo-Diff 预处理
        self.inference_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print("所有模型加载完成！\n")
    
    def classify_image(self, image):
        """分类图像"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.classifier_transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.classifier(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            class_id = outputs.argmax(1).item()
            confidence = probs[0, class_id].item()
        
        return class_id, ID_TO_CLASS[class_id], confidence
    
    def preprocess_for_decodiff(self, image):
        """为 DeCo-Diff 预处理图像"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img = image.convert('RGB')
        img = img.resize((self.center_size, self.center_size), Image.BILINEAR)
        img_tensor = self.inference_transform(img).unsqueeze(0)
        
        return img, img_tensor
    
    def detect_anomaly(self, image_path, threshold=0.35, use_gate_fusion=False, output_path=None):
        """检测单张图片的异常"""
        # 加载图像
        original_image = Image.open(image_path).convert('RGB')
        
        # 1. 自动分类
        class_id, class_name, class_confidence = self.classify_image(original_image)
        print(f"检测到类别: {class_name} (ID: {class_id}, 置信度: {class_confidence:.2%})")
        
        # 2. 预处理
        processed_img, img_tensor = self.preprocess_for_decodiff(original_image)
        
        with torch.no_grad():
            # 3. 编码到 latent 空间
            encoded = self.vae.encode(img_tensor.to(self.device)).latent_dist.mean.mul_(0.18215)
            
            model_kwargs = {
                'context': torch.tensor([[class_id]]).to(self.device),
                'mask': None
            }
            
            # 4. Diffusion 采样
            latent_samples = self.diffusion.ddim_deviation_sample_loop(
                self.model, encoded.shape, noise=encoded, clip_denoised=False,
                start_t=self.reverse_steps,
                model_kwargs=model_kwargs, progress=False, device=self.device, eta=0
            )
            
            # 5. 提取门控图
            gate_map = None
            if use_gate_fusion:
                t = torch.zeros(1, dtype=torch.long, device=self.device)
                gate_map, _ = extract_gate_maps(self.model, encoded, t, model_kwargs, self.device)
            
            # 6. 解码
            image_samples = self.vae.decode(latent_samples / 0.18215).sample
            x0 = self.vae.decode(encoded / 0.18215).sample
        
        # 7. 计算异常图
        anomaly_maps = calculate_anomaly_map(
            x0, encoded, image_samples, latent_samples,
            gate_map=gate_map, center_size=self.center_size
        )
        
        # 8. 重建图像
        reconstructed = image_samples[0].cpu().numpy().transpose(1, 2, 0)
        reconstructed = np.clip((reconstructed + 1) * 127.5, 0, 255).astype(np.uint8)
        reconstructed_img = Image.fromarray(reconstructed)
        
        # 9. 可视化
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_result.png"
        
        anomaly_score, is_anomalous = visualize_results(
            processed_img, reconstructed_img, anomaly_maps,
            output_path, threshold, use_gate_fusion,
            class_name, class_confidence
        )
        
        print(f"异常分数: {anomaly_score:.4f} | 阈值: {threshold} | 结果: {'异常' if is_anomalous else '正常'}")
        print(f"结果保存到: {output_path}\n")
        
        return {
            'image_path': image_path,
            'class_name': class_name,
            'class_id': class_id,
            'class_confidence': class_confidence,
            'anomaly_score': anomaly_score,
            'threshold': threshold,
            'is_anomalous': is_anomalous,
            'output_path': output_path
        }
    
    def batch_detect(self, image_dir, threshold=0.35, use_gate_fusion=False, output_dir=None):
        """批量检测"""
        # 查找所有图片
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob(os.path.join(image_dir, ext)))
            image_paths.extend(glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        image_paths = list(set(image_paths))  # 去重
        print(f"找到 {len(image_paths)} 张图片\n")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for img_path in image_paths:
            try:
                if output_dir:
                    output_path = os.path.join(output_dir, os.path.basename(img_path).replace('.', '_result.'))
                else:
                    output_path = None
                
                result = self.detect_anomaly(img_path, threshold, use_gate_fusion, output_path)
                results.append(result)
            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")
        
        # 统计
        print("\n" + "="*60)
        print("批量检测完成！")
        print(f"总计: {len(results)} 张")
        print(f"异常: {sum(1 for r in results if r['is_anomalous'])} 张")
        print(f"正常: {sum(1 for r in results if not r['is_anomalous'])} 张")
        print("="*60)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="DeCo-Diff 自动类别检测推理")
    parser.add_argument("--image", type=str, default='', help="单张图片路径")
    parser.add_argument("--image-dir", type=str, default='', help="批量处理目录")
    parser.add_argument("--model-path", type=str, required=True, help="DeCo-Diff 模型路径")
    parser.add_argument("--classifier-path", type=str, required=True, help="分类器模型路径")
    parser.add_argument("--output", type=str, default='', help="输出路径")
    parser.add_argument("--output-dir", type=str, default='', help="批量输出目录")
    parser.add_argument("--model-size", type=str, default='UNet_L', choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'])
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--vae-type", type=str, default="ema", choices=["ema", "mse"])
    parser.add_argument("--reverse-steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35, help="异常判断阈值")
    parser.add_argument("--use-gate-fusion", action='store_true', help="使用门控融合")
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = AutoInference(
        model_path=args.model_path,
        classifier_path=args.classifier_path,
        model_size=args.model_size,
        vae_type=args.vae_type,
        center_size=args.center_size,
        reverse_steps=args.reverse_steps
    )
    
    if args.image:
        # 单张推理
        inferencer.detect_anomaly(
            args.image, 
            threshold=args.threshold,
            use_gate_fusion=args.use_gate_fusion,
            output_path=args.output if args.output else None
        )
    elif args.image_dir:
        # 批量推理
        inferencer.batch_detect(
            args.image_dir,
            threshold=args.threshold,
            use_gate_fusion=args.use_gate_fusion,
            output_dir=args.output_dir if args.output_dir else None
        )
    else:
        print("错误: 请指定 --image 或 --image-dir")


if __name__ == "__main__":
    main()
