#!/usr/bin/env python3
"""
Setup script to initialize the app with test models.
Generates dummy PyTorch models and exports them to ONNX format for testing.
"""
import sys
import os
from pathlib import Path
import logging

# Add the current directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.models.dummy_generator import generate_test_models
from ml.optimization.export import ModelExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_test_models():
    """Generate dummy models and export to ONNX for testing"""

    # Define directories
    test_models_dir = Path("test_models")
    onnx_models_dir = Path("test_models/onnx")

    try:
        logger.info("Generating dummy PyTorch models...")
        # Generate dummy PyTorch models
        result = generate_test_models(str(test_models_dir))
        cnn_path = result['models']['cnn']
        vit_path = result['models']['vit']

        logger.info(f"Generated CNN model: {cnn_path}")
        logger.info(f"Generated ViT model: {vit_path}")

        logger.info("Exporting models to ONNX format...")
        # Export to ONNX
        exporter = ModelExporter()

        # Export CNN
        cnn_onnx_path = exporter.export_efficient_cnn(cnn_path, onnx_models_dir)
        logger.info(f"Exported CNN to ONNX: {cnn_onnx_path}")

        # Export ViT
        vit_onnx_path = exporter.export_vision_transformer(vit_path, onnx_models_dir)
        logger.info(f"Exported ViT to ONNX: {vit_onnx_path}")

        logger.info("Setup complete! Test models are ready.")
        logger.info(f"PyTorch models: {test_models_dir}")
        logger.info(f"ONNX models: {onnx_models_dir}")

        return {
            "pytorch_models": {
                "cnn": cnn_path,
                "vit": vit_path
            },
            "onnx_models": {
                "cnn": cnn_onnx_path,
                "vit": vit_onnx_path
            }
        }

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise

if __name__ == "__main__":
    setup_test_models()