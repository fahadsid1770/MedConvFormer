import torch
import torch.nn as nn
from .efficient_cnn import EfficientCNN
from .vision_transformer import MedicalViT

class HybridClassifier:
    def __init__(self, cnn_path, vit_path, confidence_threshold=0.85, device='cuda'):
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Load models
        self.cnn = EfficientCNN(num_classes=3)
        self.cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        self.cnn.to(device)
        self.cnn.eval()

        self.vit = MedicalViT(num_classes=3)
        self.vit.load_state_dict(torch.load(vit_path, map_location=device))
        self.vit.to(device)
        self.vit.eval()

        self.class_names = ['Normal', 'Pneumonia', 'COVID-19']

    @torch.no_grad()
    def predict(self, image_tensor):
        """
        Two-stage prediction:
        1. Fast CNN inference
        2. If confidence < threshold, use ViT for refinement
        """
        image_tensor = image_tensor.to(self.device)

        # Stage 1: CNN prediction
        cnn_probs, cnn_confidence = self.cnn.get_confidence(image_tensor)
        cnn_pred = torch.argmax(cnn_probs, dim=1)

        # Stage 2: Conditional ViT inference
        if cnn_confidence.item() >= self.confidence_threshold:
            final_probs = cnn_probs
            model_used = 'cnn'
        else:
            vit_logits = self.vit(image_tensor)
            vit_probs = torch.softmax(vit_logits, dim=1)

            # Ensemble: weighted average (CNN: 0.4, ViT: 0.6)
            final_probs = 0.4 * cnn_probs + 0.6 * vit_probs
            model_used = 'hybrid'

        final_pred = torch.argmax(final_probs, dim=1)
        final_confidence = torch.max(final_probs, dim=1)[0]

        return {
            'prediction': self.class_names[final_pred.item()],
            'confidence': final_confidence.item(),
            'probabilities': {
                name: prob.item()
                for name, prob in zip(self.class_names, final_probs[0])
            },
            'model_used': model_used,
            'cnn_confidence': cnn_confidence.item()
        }