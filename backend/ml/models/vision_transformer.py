import torch
import torch.nn as nn
import timm

class MedicalViT(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, img_size=224):
        super().__init__()

        # Use ViT-Small patch16 (22M parameters)
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size
        )

        feature_dim = self.backbone.num_features

        # Medical-specific head with attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Get patch embeddings
        features = self.backbone.forward_features(x)  # [B, num_patches, dim]

        # Attention pooling
        attn_weights = self.attention_pool(features)
        pooled = torch.sum(features * attn_weights, dim=1)

        return self.classifier(pooled)

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