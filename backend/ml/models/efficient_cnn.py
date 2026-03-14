import torch
import torch.nn as nn
import timm

class EfficientCNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        # Use EfficientNet-B0 as backbone (5.3M parameters)
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Custom classification head
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

    def get_confidence(self, x):
        """Return softmax probabilities for confidence scoring"""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        return probs, confidence

    def get_input_spec(self):
        """Get input specification for ONNX export"""
        return {
            "input_shape": (1, 3, 224, 224),
            "input_names": ["input_image"],
            "output_names": ["logits"],
            "dynamic_axes": {
                "input_image": {0: "batch_size"},
                "logits": {0: "batch_size"}
            }
        }