"""
DeCo-Diff Auto Inference Gradio Web Interface

Features:
- Automatic class recognition
- Configurable parameters
- Display results from each processing stage

Launch:
    python app_gradio_auto.py --model-path ./checkpoints/last.pt --classifier-path ./classifier_mvtec.pth
    
    # Public access
    python app_gradio_auto.py --model-path ./checkpoints/last.pt --classifier-path ./classifier_mvtec.pth --share
"""

import torch
import gradio as gr
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models
import numpy as np
torch.set_grad_enabled(False)
from torchvision import transforms, models
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
# Set Times font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

import argparse
import torch.nn as nn
import io

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== Class Mapping =====================
MVTEC_CLASS_MAP = {
    "capsule": 0, "bottle": 1, "grid": 2, "leather": 3, "metal_nut": 4,
    "tile": 5, "transistor": 6, "zipper": 7, "cable": 8, "carpet": 9,
    "hazelnut": 10, "pill": 11, "screw": 12, "toothbrush": 13, "wood": 14,
}
ID_TO_CLASS = {v: k for k, v in MVTEC_CLASS_MAP.items()}
NUM_CLASSES = len(MVTEC_CLASS_MAP)


# ===================== Classifier Model =====================
class MVTecClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    
    def forward(self, x):
        return self.backbone(x)


# ===================== Global Models =====================
classifier = None
vae = None
decodiff_model = None
diffusion = None
current_config = {}


def smooth_mask(mask, sigma=1.0):
    return gaussian_filter(mask, sigma=sigma)


def load_models(model_path, classifier_path, model_size, vae_type, center_size, reverse_steps):
    """Load all models"""
    global classifier, vae, decodiff_model, diffusion, current_config
    
    status_messages = []
    
    new_config = {
        'model_path': model_path,
        'classifier_path': classifier_path,
        'model_size': model_size,
        'vae_type': vae_type,
        'center_size': center_size,
        'reverse_steps': reverse_steps
    }
    
    if current_config == new_config and classifier is not None:
        return "✅ Models already loaded"
    
    try:
        status_messages.append("Loading classifier...")
        classifier = MVTecClassifier(NUM_CLASSES).to(device)
        ckpt = torch.load(classifier_path, map_location='cpu',weights_only=False)
        classifier.load_state_dict(ckpt['model_state_dict'])
        classifier.eval()
        status_messages.append(f"✅ Classifier loaded (Acc: {ckpt.get('best_acc', 'N/A'):.1f}%)")
        
        status_messages.append("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
        vae.eval()
        status_messages.append("✅ VAE loaded")
        
        status_messages.append("Loading DeCo-Diff model...")
        latent_size = center_size // 8
        decodiff_model = UNET_models[model_size](latent_size=latent_size, ncls=NUM_CLASSES)
        state_dict = torch.load(model_path, map_location='cpu',weights_only=False)['model']
        decodiff_model.load_state_dict(state_dict, strict=False)
        decodiff_model.eval()
        decodiff_model.to(device)
        status_messages.append("✅ DeCo-Diff loaded")
        
        diffusion = create_diffusion(
            f'ddim{reverse_steps}', predict_deviation=True,
            sigma_small=False, predict_xstart=False, diffusion_steps=10
        )
        status_messages.append("✅ All models ready")
        
        current_config = new_config
        return "\n".join(status_messages)
    
    except Exception as e:
        return f"❌ Load failed: {str(e)}"


def classify_image(image, center_size):
    """Classify image"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = classifier(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        class_id = outputs.argmax(1).item()
        confidence = probs[0, class_id].item()
    
    top3_probs, top3_ids = torch.topk(probs, 3, dim=1)
    top3_results = [(ID_TO_CLASS[idx.item()], prob.item()) for idx, prob in zip(top3_ids[0], top3_probs[0])]
    
    return class_id, ID_TO_CLASS[class_id], confidence, top3_results


def create_results_figure(processed_img, reconstructed, image_diff, latent_diff, 
                          anomaly_map, threshold, class_name, confidence, anomaly_score):
    """Create a comprehensive results figure with 2x3 grid"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Row 1: Input, Reconstructed, Difference
    axes[0, 0].imshow(processed_img)
    axes[0, 0].set_title(f'Input Image\nClass: {class_name} ({confidence:.1%})', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(reconstructed)
    axes[0, 1].set_title('Reconstructed Image', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Image difference
    im1 = axes[0, 2].imshow(image_diff, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Image Discrepancy', fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Latent diff, Heatmap, Overlay
    im2 = axes[1, 0].imshow(latent_diff, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Latent Discrepancy', fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 1].imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Anomaly Heatmap (Score: {anomaly_score:.3f})', fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Overlay
    overlay = np.array(processed_img).astype(float) / 255.0
    mask = anomaly_map > threshold
    overlay[mask, 0] = np.clip(overlay[mask, 0] + anomaly_map[mask] * 0.6, 0, 1)
    overlay[mask, 1] = overlay[mask, 1] * 0.4
    overlay[mask, 2] = overlay[mask, 2] * 0.4
    
    axes[1, 2].imshow(overlay)
    is_anomalous = anomaly_score > threshold
    status = "ANOMALY DETECTED" if is_anomalous else "NORMAL"
    color = '#D32F2F' if is_anomalous else '#388E3C'
    axes[1, 2].set_title(f'Detection Result: {status}', fontweight='bold', color=color)
    axes[1, 2].axis('off')
    
    plt.tight_layout(pad=2.0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    buf.seek(0)
    result = np.array(Image.open(buf))
    plt.close()
    
    return result


def create_classification_text(class_name, confidence, top3_results):
    """创建分类结果文本"""
    text = f"### 🏷️ 识别类别: **{class_name}**\n"
    text += f"### 📊 置信度: **{confidence:.1%}**\n\n"
    text += "| 排名 | 类别 | 置信度 |\n"
    text += "|------|------|--------|\n"
    for i, (cls, prob) in enumerate(top3_results):
        marker = "✓" if cls == class_name else ""
        text += f"| {i+1} | {cls} {marker} | {prob:.1%} |\n"
    return text


def run_inference(image, threshold, use_gate_fusion, center_size, reverse_steps):
    """Run full inference pipeline"""
    global classifier, vae, decodiff_model, diffusion
    
    if classifier is None or vae is None or decodiff_model is None:
        return None, None, "❌ Please load models first!"
    
    if image is None:
        return None, None, "❌ Please upload an image!"
    
    try:
        # Stage 1: Classification
        if isinstance(image, np.ndarray):
            original_image = Image.fromarray(image)
        else:
            original_image = image
        
        class_id, class_name, confidence, top3_results = classify_image(original_image, center_size)
        classification_text = create_classification_text(class_name, confidence, top3_results)
        
        # Stage 2: Preprocessing (consistent with inference_single.py)
        # First resize to image_size (288), then center crop to center_size (256)
        image_size = 288  # Same as inference_single.py default
        img = original_image.convert('RGB')
        img = img.resize((image_size, image_size), Image.BILINEAR)
        
        # Center crop
        left = (image_size - center_size) // 2
        top = (image_size - center_size) // 2
        processed_img = img.crop((left, top, left + center_size, top + center_size))
        
        # Transform (consistent with inference_single.py)
        inference_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        img_tensor = inference_transform(processed_img).unsqueeze(0)
        
        with torch.no_grad():
            # Stage 3: Encode
            encoded = vae.encode(img_tensor.to(device)).latent_dist.mean.mul_(0.18215)
            
            model_kwargs = {
                'context': torch.tensor([[class_id]]).to(device),
                'mask': None
            }
            
            # Stage 4: Diffusion sampling
            latent_samples = diffusion.ddim_deviation_sample_loop(
                decodiff_model, encoded.shape, noise=encoded, clip_denoised=False,
                start_t=reverse_steps,
                model_kwargs=model_kwargs, progress=False, device=device, eta=0
            )
            
            # Stage 5: Gate map extraction (consistent with inference_single.py)
            gate_map = None
            gate_map_np = None
            if use_gate_fusion:
                t = torch.zeros(encoded.shape[0], dtype=torch.long, device=device)
                output = decodiff_model(encoded, t, return_gate=True, **model_kwargs)
                if isinstance(output, tuple) and len(output) >= 2:
                    _, gate_map, skip_gates = output
                    if gate_map is not None:
                        # Process gate map same as inference_single.py
                        gate_map_np = gate_map[0].mean(dim=0).detach().cpu().numpy()
                        gate_map_np = resize(gate_map_np, (center_size, center_size))
                        gate_map_np = smooth_mask(gate_map_np, sigma=2)
            
            # Stage 6: Decode
            image_samples = vae.decode(latent_samples / 0.18215).sample
            x0 = vae.decode(encoded / 0.18215).sample
        
        # Reconstructed image
        reconstructed = image_samples[0].cpu().numpy().transpose(1, 2, 0)
        reconstructed = np.clip((reconstructed + 1) * 127.5, 0, 255).astype(np.uint8)
        
        # Stage 7: Anomaly map calculation (consistent with inference_single.py)
        # Image discrepancy: mean over batch, transpose, then max over channels
        image_diff = (((((torch.abs(image_samples-x0))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
        image_diff = (np.clip(image_diff, 0.0, 0.4)) * 2.5
        image_diff = smooth_mask(image_diff, sigma=3)
        
        # Latent discrepancy: mean over batch, transpose, then mean over channels
        latent_diff = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
        latent_diff = (np.clip(latent_diff, 0.0, 0.2)) * 5
        latent_diff = smooth_mask(latent_diff, sigma=1)
        latent_diff = resize(latent_diff, (center_size, center_size))
        
        # Geometric mean fusion
        anomaly_map = image_diff * latent_diff
        anomaly_map = np.sqrt(anomaly_map)
        
        # Gate fusion (if enabled)
        if use_gate_fusion and gate_map_np is not None:
            anomaly_map = 0.6 * anomaly_map + 0.4 * gate_map_np
        
        # Stage 8: Result
        anomaly_score = float(anomaly_map.max())
        is_anomalous = anomaly_score > threshold
        
        # Create results figure
        results_figure = create_results_figure(
            np.array(processed_img), reconstructed, image_diff, latent_diff,
            anomaly_map, threshold, class_name, confidence, anomaly_score
        )
        
        # Result report (Chinese)
        status = "🔴 检测到异常" if is_anomalous else "🟢 正常"
        result_text = f"""
## 检测结果: {status}

| 指标 | 值 |
|------|------|
| **识别类别** | {class_name} |
| **类别置信度** | {confidence:.2%} |
| **异常分数** | {anomaly_score:.4f} |
| **判断阈值** | {threshold} |
| **门控融合** | {'是' if use_gate_fusion else '否'} |

{'⚠️ **检测到异常！** 请检查标红区域。' if is_anomalous else '✅ **图像正常**，未检测到异常。'}
"""
        
        return classification_text, results_figure, result_text
    
    except Exception as e:
        import traceback
        return "", None, f"❌ 错误: {str(e)}\n{traceback.format_exc()}"


def create_app(args):
    """创建 Gradio 应用"""
    
    with gr.Blocks(title="DeCo-Diff 异常检测", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔬 DeCo-Diff-Gating 自动异常检测系统
        自动识别图像类别并进行异常检测，可视化展示检测结果。
        """)
        
        with gr.Row():
            # ========== 左侧：输入和配置 ==========
            with gr.Column(scale=1):
                gr.Markdown("### 📷 图像上传")
                image_input = gr.Image(label="上传图片", type="numpy", height=300)
                
                gr.Markdown("### ⚙️ 参数设置")
                threshold = gr.Slider(0.1, 0.8, value=0.30, step=0.05, label="异常阈值（越高越不易误报）")
                use_gate = gr.Checkbox(label="使用门控融合", value=False)
                
                run_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")
                
                with gr.Accordion("🔧 模型配置", open=False):
                    model_path = gr.Textbox(label="DeCo-Diff 模型路径", value=args.model_path)
                    classifier_path = gr.Textbox(label="分类器路径", value=args.classifier_path)
                    
                    with gr.Row():
                        model_size = gr.Dropdown(
                            choices=['UNet_XS', 'UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'],
                            value=args.model_size, label="模型大小"
                        )
                        vae_type = gr.Dropdown(choices=['ema', 'mse'], value='ema', label="VAE 类型")
                    
                    with gr.Row():
                        center_size = gr.Slider(128, 512, value=256, step=32, label="图像尺寸")
                        reverse_steps = gr.Slider(1, 10, value=5, step=1, label="反向步数")
                    
                    load_btn = gr.Button("🔄 加载模型", variant="secondary")
                    load_status = gr.Textbox(label="状态", interactive=False, lines=3)
            
            # ========== 右侧：结果展示 ==========
            with gr.Column(scale=2):
                gr.Markdown("### 🏷️ 类别识别结果")
                classification_output = gr.Markdown()
                
                gr.Markdown("### 🔍 检测结果可视化")
                results_output = gr.Image(label="分析结果")
                
                result_report = gr.Markdown(label="检测报告")
        
        # ========== 事件绑定 ==========
        load_btn.click(
            load_models,
            inputs=[model_path, classifier_path, model_size, vae_type, center_size, reverse_steps],
            outputs=load_status
        )
        
        run_btn.click(
            run_inference,
            inputs=[image_input, threshold, use_gate, center_size, reverse_steps],
            outputs=[classification_output, results_output, result_report]
        )
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeCo-Diff Auto Inference Gradio App")
    parser.add_argument("--model-path", type=str, default='', help="DeCo-Diff model path")
    parser.add_argument("--classifier-path", type=str, default='', help="Classifier model path")
    parser.add_argument("--model-size", type=str, default='UNet_L')
    parser.add_argument("--share", action="store_true", help="Create public sharing link")
    parser.add_argument("--port", type=int, default=7860)
    
    args = parser.parse_args()
    
    demo = create_app(args)
    demo.launch(share=args.share, server_port=args.port)
