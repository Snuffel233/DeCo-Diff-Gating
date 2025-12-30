"""
MVTec 类别分类器训练脚本

用于训练一个前置分类模型，自动识别 MVTec 数据集中的图像类别。
训练完成后可以用于推理脚本中自动确定 class_id。

用法:
    # 训练
    python train_classifier.py --data-dir ./mvtec-dataset --epochs 50
    
    # 推理测试
    python train_classifier.py --data-dir ./mvtec-dataset --mode test --checkpoint ./classifier_best.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import argparse
from glob import glob
from tqdm import tqdm
import json

# MVTec 类别映射（按用户指定顺序）
MVTEC_CLASS_MAP = {
    "capsule": 0,
    "bottle": 1,
    "grid": 2,
    "leather": 3,
    "metal_nut": 4,
    "tile": 5,
    "transistor": 6,
    "zipper": 7,
    "cable": 8,
    "carpet": 9,
    "hazelnut": 10,
    "pill": 11,
    "screw": 12,
    "toothbrush": 13,
    "wood": 14,
}

# 反向映射
ID_TO_CLASS = {v: k for k, v in MVTEC_CLASS_MAP.items()}
NUM_CLASSES = len(MVTEC_CLASS_MAP)


class MVTecClassificationDataset(Dataset):
    """MVTec 分类数据集"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: MVTec 数据集根目录
            split: 'train' 或 'test'
            transform: 图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 收集所有图像
        for class_name, class_id in MVTEC_CLASS_MAP.items():
            class_dir = os.path.join(root_dir, class_name, split)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} does not exist, skipping...")
                continue
            
            # 遍历所有子目录（good, defect types...）
            for subdir in os.listdir(class_dir):
                subdir_path = os.path.join(class_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_name in os.listdir(subdir_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(subdir_path, img_name)
                            self.samples.append((img_path, class_id))
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MVTecClassifier(nn.Module):
    """基于 ResNet 的分类器"""
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        # 使用 ResNet-18 作为骨干网络（轻量且效果好）
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 每个类别的统计
    class_correct = {i: 0 for i in range(NUM_CLASSES)}
    class_total = {i: 0 for i in range(NUM_CLASSES)}
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 统计每个类别
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    # 打印每个类别的准确率
    print("\n每个类别的准确率:")
    print("-" * 40)
    for class_id in range(NUM_CLASSES):
        class_name = ID_TO_CLASS[class_id]
        if class_total[class_id] > 0:
            acc = 100. * class_correct[class_id] / class_total[class_id]
            print(f"  {class_name:12s}: {acc:6.2f}% ({class_correct[class_id]}/{class_total[class_id]})")
    print("-" * 40)
    
    return running_loss / len(dataloader), 100. * correct / total


def predict_single_image(model, image_path, transform, device):
    """预测单张图片"""
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_id = outputs.argmax(1).item()
        confidence = probabilities[0, predicted_id].item()
    
    return predicted_id, ID_TO_CLASS[predicted_id], confidence


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.mode == 'train':
        # 加载数据
        train_dataset = MVTecClassificationDataset(args.data_dir, 'train', train_transform)
        test_dataset = MVTecClassificationDataset(args.data_dir, 'test', test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # 创建模型
        model = MVTecClassifier(num_classes=NUM_CLASSES, pretrained=True).to(device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        best_acc = 0.0
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()
            
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'class_map': MVTEC_CLASS_MAP,
                }, args.save_path)
                print(f"  -> Saved best model with accuracy {best_acc:.2f}%")
        
        print(f"\nTraining completed! Best accuracy: {best_acc:.2f}%")
        print(f"Model saved to: {args.save_path}")
    
    elif args.mode == 'test':
        # 加载模型
        model = MVTecClassifier(num_classes=NUM_CLASSES, pretrained=False).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.checkpoint}, best acc: {checkpoint['best_acc']:.2f}%")
        
        # 测试单张图片
        if args.image:
            pred_id, pred_class, confidence = predict_single_image(model, args.image, test_transform, device)
            print(f"\n预测结果:")
            print(f"  图片: {args.image}")
            print(f"  类别: {pred_class}")
            print(f"  ID: {pred_id}")
            print(f"  置信度: {confidence:.2%}")
        else:
            # 在测试集上评估
            test_dataset = MVTecClassificationDataset(args.data_dir, 'test', test_transform)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            criterion = nn.CrossEntropyLoss()
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    elif args.mode == 'predict':
        # 预测模式：输出类别 ID
        if not args.image:
            print("Error: --image is required for predict mode")
            return
        
        model = MVTecClassifier(num_classes=NUM_CLASSES, pretrained=False).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        pred_id, pred_class, confidence = predict_single_image(model, args.image, test_transform, device)
        
        # 输出 JSON 格式，方便其他脚本调用
        result = {
            "image": args.image,
            "class_name": pred_class,
            "class_id": pred_id,
            "confidence": confidence
        }
        print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVTec 类别分类器训练")
    parser.add_argument("--data-dir", type=str, default='./mvtec-dataset', help="MVTec 数据集路径")
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'predict'], default='train', help="运行模式")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--save-path", type=str, default='./classifier_mvtec.pth', help="模型保存路径")
    parser.add_argument("--checkpoint", type=str, default='./classifier_mvtec.pth', help="加载的模型路径")
    parser.add_argument("--image", type=str, default='', help="单张图片路径（用于测试或预测）")
    
    args = parser.parse_args()
    main(args)
