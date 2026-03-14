import torch
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging
import copy

logger = logging.getLogger(__name__)

class ModelQuantizer:
    """Handles quantization of PyTorch models for optimized inference"""

    def __init__(self):
        pass

    def quantize_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        layers_to_quantize: Optional[list] = None
    ) -> nn.Module:
        """
        Apply dynamic quantization to the model

        Args:
            model: PyTorch model to quantize
            dtype: Quantization data type (default: qint8)
            layers_to_quantize: Specific layers to quantize (default: Linear and LSTM)

        Returns:
            Quantized model
        """
        if layers_to_quantize is None:
            layers_to_quantize = [nn.Linear, nn.LSTM, nn.GRU]

        quantized_model = quantize_dynamic(
            model,
            layers_to_quantize,
            dtype=dtype
        )

        return quantized_model

    def quantize_static(
        self,
        model: nn.Module,
        calibration_loader: torch.utils.data.DataLoader,
        device: str = "cpu"
    ) -> nn.Module:
        """
        Apply static quantization to the model

        Args:
            model: PyTorch model to quantize
            calibration_loader: DataLoader for calibration
            device: Device to run quantization on

        Returns:
            Statically quantized model
        """
        # Prepare model for quantization
        model.eval()
        model.to(device)

        # Fuse layers if possible (Conv + BN + ReLU)
        model = self._fuse_layers(model)

        # Add quantization stubs
        model = self._add_quant_stubs(model)

        # Prepare for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)

        # Calibrate with sample data
        self._calibrate(model, calibration_loader, device)

        # Convert to quantized model
        torch.quantization.convert(model, inplace=True)

        return model

    def _fuse_layers(self, model: nn.Module) -> nn.Module:
        """Fuse Conv2d + BatchNorm2d + ReLU layers for better quantization"""
        fusion_patterns = [
            [nn.Conv2d, nn.BatchNorm2d],
            [nn.Conv2d, nn.BatchNorm2d, nn.ReLU],
            [nn.Linear, nn.ReLU]
        ]

        for pattern in fusion_patterns:
            model = torch.quantization.fuse_modules(model, fusion_patterns, inplace=True)

        return model

    def _add_quant_stubs(self, model: nn.Module) -> nn.Module:
        """Add QuantStub and DeQuantStub to model for static quantization"""
        # This is a simplified version - in practice, you'd need to identify
        # the input and output points of the model
        model.quant = QuantStub()
        model.dequant = DeQuantStub()

        # Wrap forward method to include quantization
        original_forward = model.forward

        def quantized_forward(self, x):
            x = self.quant(x)
            x = original_forward(x)
            x = self.dequant(x)
            return x

        model.forward = quantized_forward.__get__(model, model.__class__)
        return model

    def _calibrate(self, model: nn.Module, calibration_loader: torch.utils.data.DataLoader, device: str):
        """Calibrate the model with sample data"""
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                data = data.to(device)
                _ = model(data)

                # Limit calibration to first few batches
                if batch_idx >= 100:
                    break

    def save_quantized_model(self, model: nn.Module, path: Union[str, Path]) -> str:
        """Save quantized model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), path)
        logger.info(f"Saved quantized model to {path}")
        return str(path)

    def load_quantized_model(self, model_class: type, path: Union[str, Path], device: str = "cpu") -> nn.Module:
        """Load quantized model"""
        model = model_class()
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def quantize_efficient_cnn(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        quantization_type: str = "dynamic",
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cpu"
    ) -> str:
        """Quantize EfficientCNN model"""
        from ..models.efficient_cnn import EfficientCNN

        # Load model
        model = EfficientCNN(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        if quantization_type == "dynamic":
            quantized_model = self.quantize_dynamic(model)
        elif quantization_type == "static":
            if calibration_loader is None:
                raise ValueError("Calibration loader required for static quantization")
            quantized_model = self.quantize_static(model, calibration_loader, device)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        # Save quantized model
        output_path = Path(output_path) / f"efficient_cnn_{quantization_type}_quantized.pth"
        return self.save_quantized_model(quantized_model, output_path)

    def quantize_vision_transformer(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        quantization_type: str = "dynamic",
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cpu"
    ) -> str:
        """Quantize MedicalViT model"""
        from ..models.vision_transformer import MedicalViT

        # Load model
        model = MedicalViT(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        if quantization_type == "dynamic":
            quantized_model = self.quantize_dynamic(model)
        elif quantization_type == "static":
            if calibration_loader is None:
                raise ValueError("Calibration loader required for static quantization")
            quantized_model = self.quantize_static(model, calibration_loader, device)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        # Save quantized model
        output_path = Path(output_path) / f"vision_transformer_{quantization_type}_quantized.pth"
        return self.save_quantized_model(quantized_model, output_path)