"""
Training script for the COVID-19 vs Pneumonia classifier.
Supports training EfficientCNN and MedicalViT models.
"""
import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset paths - use absolute path from environment or default
PROJECT_ROOT = Path(os.environ.get('PROJECT_ROOT', '/home/exorcist1770/Projects/hybrid-cnn-transformer-for-distinguishing-covid-vs-pnumonia'))
DATA_ROOT = PROJECT_ROOT / "data" / "COVID-19_Radiography_Dataset"
COVID_DIR = DATA_ROOT / "COVID" / "images"
NORMAL_DIR = DATA_ROOT / "Normal" / "images"
PNEUMONIA_DIR = DATA_ROOT / "Viral Pneumonia" / "images"

# Model output directory
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)


class XRayDataset(torch.utils.data.Dataset):
    """Custom dataset for X-ray images"""
    
    def __init__(self, image_dirs, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        # Class mapping: COVID=0, Normal=1, Pneumonia=2
        class_dirs = [
            (COVID_DIR, 0, "COVID"),
            (NORMAL_DIR, 1, "Normal"),
            (PNEUMONIA_DIR, 2, "Pneumonia")
        ]
        
        for dir_path, label, class_name in class_dirs:
            if dir_path.exists():
                for img_file in dir_path.iterdir():
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        self.image_paths.append(img_file)
                        self.labels.append(label)
        
        logger.info(f"Loaded {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224, is_training=True):
    """Get data transforms"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class EfficientCNN(nn.Module):
    """EfficientNet-B0 based CNN classifier"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            feature_dim = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


class MedicalViT(nn.Module):
    """Vision Transformer for medical imaging"""
    
    def __init__(self, num_classes=3, pretrained=True, img_size=224):
        super().__init__()
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size
        )
        
        feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled = features[:, 0]  # Use CLS token
        return self.classifier(pooled)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def train_model(model_name, epochs=5, batch_size=32, lr=1e-4, img_size=224):
    """Train a model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    dataset = XRayDataset([COVID_DIR, NORMAL_DIR, PNEUMONIA_DIR], transform=train_transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.transform = val_transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Create model
    if model_name == 'cnn':
        model = EfficientCNN(num_classes=3, pretrained=True)
    elif model_name == 'vit':
        model = MedicalViT(num_classes=3, pretrained=True, img_size=img_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = MODEL_DIR / f"{model_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model to {save_path}")
    
    logger.info(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    return model


def export_to_onnx(model_name):
    """Export trained model to ONNX format"""
    import torch.onnx
    
    device = torch.device('cpu')
    
    # Load model
    if model_name == 'cnn':
        model = EfficientCNN(num_classes=3, pretrained=False)
    elif model_name == 'vit':
        model = MedicalViT(num_classes=3, pretrained=False, img_size=224)
    
    model_path = MODEL_DIR / f"{model_name}_best.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    onnx_path = MODEL_DIR / f"{model_name}_best.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        input_names=['input_image'],
        output_names=['logits'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train COVID-19 classifier")
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'vit'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--export', action='store_true', help='Export to ONNX after training')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size
    )
    
    # Export to ONNX if requested
    if args.export:
        export_to_onnx(args.model)
