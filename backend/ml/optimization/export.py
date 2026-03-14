import torch
import torch.onnx
import os
from pathlib import Path
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

class ModelExporter:
    """Handles ONNX export for PyTorch models"""

    def __init__(self, opset_version: int = 14):
        self.opset_version = opset_version

    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_sample: torch.Tensor,
        output_path: Union[str, Path],
        model_name: str = "model",
        input_names: Optional[list] = None,
        output_names: Optional[list] = None,
        dynamic_axes: Optional[dict] = None,
        verbose: bool = False
    ) -> str:
        """
        Export PyTorch model to ONNX format

        Args:
            model: PyTorch model to export
            input_sample: Sample input tensor for tracing
            output_path: Directory to save ONNX model
            model_name: Name for the ONNX file
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification for variable batch sizes
            verbose: Enable verbose logging

        Returns:
            Path to the exported ONNX model
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        onnx_path = output_path / f"{model_name}.onnx"

        # Set default names if not provided
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Set dynamic axes for batch size if not provided
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }

        try:
            # Ensure model is in eval mode
            model.eval()

            # Export to ONNX
            torch.onnx.export(
                model,
                input_sample,
                str(onnx_path),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=verbose
            )

            logger.info(f"Successfully exported model to {onnx_path}")
            return str(onnx_path)

        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            raise

    def export_efficient_cnn(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        device: str = "cpu"
    ) -> str:
        """Export EfficientCNN model to ONNX"""
        from ..models.efficient_cnn import EfficientCNN

        # Load model
        model = EfficientCNN(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Create sample input
        input_sample = torch.randn(1, 3, 224, 224).to(device)

        return self.export_to_onnx(
            model,
            input_sample,
            output_path,
            model_name="efficient_cnn",
            input_names=["input_image"],
            output_names=["logits"]
        )

    def export_vision_transformer(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        device: str = "cpu"
    ) -> str:
        """Export MedicalViT model to ONNX"""
        from ..models.vision_transformer import MedicalViT

        # Load model
        model = MedicalViT(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Create sample input
        input_sample = torch.randn(1, 3, 224, 224).to(device)

        return self.export_to_onnx(
            model,
            input_sample,
            output_path,
            model_name="vision_transformer",
            input_names=["input_image"],
            output_names=["logits"]
        )

    def export_hybrid_system(
        self,
        cnn_path: Union[str, Path],
        vit_path: Union[str, Path],
        output_path: Union[str, Path],
        device: str = "cpu"
    ) -> dict:
        """
        Export hybrid system components to ONNX
        Note: The hybrid logic itself cannot be directly exported to ONNX
        as it contains conditional logic. Export individual models instead.
        """
        results = {}

        # Export individual models
        results['cnn'] = self.export_efficient_cnn(cnn_path, output_path, device)
        results['vit'] = self.export_vision_transformer(vit_path, output_path, device)

        logger.info("Exported hybrid system components to ONNX")
        return results