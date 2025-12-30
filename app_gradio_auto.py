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


def create_classification_plot(top3_results, predicted_class):
    """Create classification bar chart"""
    fig, ax = plt.subplots(figsize=(5, 3), facecolor='white')
    fig.patch.set_facecolor('white')
    
    classes = [r[0] for r in top3_results]
    probs = [r[1] * 100 for r in top3_results]
    colors = ['#4CAF50' if c == predicted_class else '#64B5F6' for c in classes]
    
    bars = ax.barh(classes, probs, color=colors, edgecolor='white', height=0.6)
    ax.set_xlabel('Confidence (%)', fontweight='bold')
    ax.set_title('Category Prediction (Top-3)', fontweight='bold', pad=10)
    ax.set_xlim(0, 105)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    result = np.array(Image.open(buf))
    plt.close()
    
    return result


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
        classification_plot = create_classification_plot(top3_results, class_name)
        
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
        
        # Result report
        status = "🔴 ANOMALY DETECTED" if is_anomalous else "🟢 NORMAL"
        result_text = f"""
## Detection Result: {status}

| Metric | Value |
|--------|-------|
| **Predicted Class** | {class_name} |
| **Class Confidence** | {confidence:.2%} |
| **Anomaly Score** | {anomaly_score:.4f} |
| **Threshold** | {threshold} |
| **Gate Fusion** | {'Yes' if use_gate_fusion else 'No'} |

{'⚠️ **Anomaly detected!** Please check the highlighted regions.' if is_anomalous else '✅ **Image is normal.** No anomalies detected.'}
"""
        
        return classification_plot, results_figure, result_text
    
    except Exception as e:
        import traceback
        return None, None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"


def create_app(args):
    """Create Gradio application"""
    
    with gr.Blocks(title="DeCo-Diff Anomaly Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔬 DeCo-Diff Automatic Anomaly Detection System
        Automatic category recognition and anomaly detection with visualization of each processing stage.
        """)
        
        with gr.Row():
            # ========== Left Column: Input & Config ==========
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Input")
                image_input = gr.Image(label="Upload Image", type="numpy", height=300)
                
                gr.Markdown("### ⚙️ Parameters")
                threshold = gr.Slider(0.1, 0.8, value=0.30, step=0.05, label="Anomaly Threshold")
                use_gate = gr.Checkbox(label="Use Gate Fusion", value=False)
                
                run_btn = gr.Button("🚀 Run Detection", variant="primary", size="lg")
                
                with gr.Accordion("🔧 Model Configuration", open=False):
                    model_path = gr.Textbox(label="DeCo-Diff Model Path", value=args.model_path)
                    classifier_path = gr.Textbox(label="Classifier Path", value=args.classifier_path)
                    
                    with gr.Row():
                        model_size = gr.Dropdown(
                            choices=['UNet_XS', 'UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'],
                            value=args.model_size, label="Model Size"
                        )
                        vae_type = gr.Dropdown(choices=['ema', 'mse'], value='ema', label="VAE Type")
                    
                    with gr.Row():
                        center_size = gr.Slider(128, 512, value=256, step=32, label="Image Size")
                        reverse_steps = gr.Slider(1, 10, value=5, step=1, label="Reverse Steps")
                    
                    load_btn = gr.Button("🔄 Load Models", variant="secondary")
                    load_status = gr.Textbox(label="Status", interactive=False, lines=3)
            
            # ========== Right Column: Results ==========
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Classification Result")
                classification_output = gr.Image(label="Category Prediction", height=200)
                
                gr.Markdown("### 🔍 Detection Results")
                results_output = gr.Image(label="Analysis Results")
                
                result_report = gr.Markdown(label="Report")
        
        # ========== Event Bindings ==========
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
