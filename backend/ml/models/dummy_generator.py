import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DummyModelGenerator:
    """Generates dummy trained models for testing optimization and export functionality"""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def create_dummy_efficient_cnn(
        self,
        output_path: Union[str, Path],
        num_classes: int = 3,
        pretrained: bool = False
    ) -> str:
        """Create and save a dummy EfficientCNN model"""
        from .efficient_cnn import EfficientCNN

        # Create model
        model = EfficientCNN(num_classes=num_classes, pretrained=pretrained)
        model.to(self.device)

        # Initialize with random weights (simulating training)
        model.eval()

        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = output_path / "dummy_efficient_cnn.pth"

        # Save model
        torch.save(model.state_dict(), model_path)

        logger.info(f"Created dummy EfficientCNN model at {model_path}")
        return str(model_path)

    def create_dummy_vision_transformer(
        self,
        output_path: Union[str, Path],
        num_classes: int = 3,
        pretrained: bool = False
    ) -> str:
        """Create and save a dummy MedicalViT model"""
        from .vision_transformer import MedicalViT

        # Create model
        model = MedicalViT(num_classes=num_classes, pretrained=pretrained)
        model.to(self.device)

        # Initialize with random weights (simulating training)
        model.eval()

        # Create output directory
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        model_path = output_path / "dummy_vision_transformer.pth"

        # Save model
        torch.save(model.state_dict(), model_path)

        logger.info(f"Created dummy MedicalViT model at {model_path}")
        return str(model_path)

    def create_dummy_models(
        self,
        output_path: Union[str, Path],
        num_classes: int = 3,
        pretrained: bool = False
    ) -> Dict[str, str]:
        """Create both dummy models for hybrid system testing"""
        results = {}

        results['cnn'] = self.create_dummy_efficient_cnn(
            output_path, num_classes, pretrained
        )

        results['vit'] = self.create_dummy_vision_transformer(
            output_path, num_classes, pretrained
        )

        logger.info(f"Created dummy models for hybrid system at {output_path}")
        return results

    def create_random_data_sample(
        self,
        batch_size: int = 1,
        image_size: tuple = (224, 224),
        num_channels: int = 3
    ) -> torch.Tensor:
        """Create a random data sample for testing"""
        return torch.randn(batch_size, num_channels, *image_size)

    def create_calibration_dataset(
        self,
        num_samples: int = 100,
        image_size: tuple = (224, 224),
        num_channels: int = 3,
        num_classes: int = 3
    ) -> torch.utils.data.Dataset:
        """Create a dummy dataset for quantization calibration"""

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples, image_size, num_channels, num_classes):
                self.num_samples = num_samples
                self.image_size = image_size
                self.num_channels = num_channels
                self.num_classes = num_classes

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                # Random image
                image = torch.randn(self.num_channels, *self.image_size)
                # Random label
                label = torch.randint(0, self.num_classes, (1,)).item()
                return image, label

        return DummyDataset(num_samples, image_size, num_channels, num_classes)

    def create_calibration_loader(
        self,
        num_samples: int = 100,
        batch_size: int = 16,
        image_size: tuple = (224, 224),
        num_channels: int = 3,
        num_classes: int = 3
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for quantization calibration"""
        dataset = self.create_calibration_dataset(
            num_samples, image_size, num_channels, num_classes
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # Avoid multiprocessing issues in tests
        )

def generate_test_models(output_dir: str = "test_models") -> Dict[str, Any]:
    """Convenience function to generate all test models and data"""
    generator = DummyModelGenerator()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate models
    models = generator.create_dummy_models(output_path)

    # Generate test data
    test_input = generator.create_random_data_sample()

    # Generate calibration loader
    calibration_loader = generator.create_calibration_loader()

    return {
        "models": models,
        "test_input": test_input,
        "calibration_loader": calibration_loader,
        "output_dir": str(output_path)
    }