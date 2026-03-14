import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import os

from ml.optimization.export import ModelExporter
from ml.optimization.quantization import ModelQuantizer
from ml.optimization.benchmark import ModelBenchmarker
from ml.models.dummy_generator import DummyModelGenerator, generate_test_models

class TestModelExporter:
    """Test ONNX export functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dummy_models(self, temp_dir):
        """Generate dummy models for testing"""
        generator = DummyModelGenerator()
        return generator.create_dummy_models(temp_dir)

    @pytest.fixture
    def exporter(self):
        """Create model exporter instance"""
        return ModelExporter()

    def test_export_efficient_cnn(self, exporter, dummy_models, temp_dir):
        """Test exporting EfficientCNN to ONNX"""
        onnx_path = exporter.export_efficient_cnn(
            dummy_models['cnn'],
            temp_dir,
            device='cpu'
        )

        assert os.path.exists(onnx_path)
        assert onnx_path.endswith('.onnx')

    def test_export_vision_transformer(self, exporter, dummy_models, temp_dir):
        """Test exporting MedicalViT to ONNX"""
        onnx_path = exporter.export_vision_transformer(
            dummy_models['vit'],
            temp_dir,
            device='cpu'
        )

        assert os.path.exists(onnx_path)
        assert onnx_path.endswith('.onnx')

    def test_export_hybrid_system(self, exporter, dummy_models, temp_dir):
        """Test exporting hybrid system components to ONNX"""
        results = exporter.export_hybrid_system(
            dummy_models['cnn'],
            dummy_models['vit'],
            temp_dir,
            device='cpu'
        )

        assert 'cnn' in results
        assert 'vit' in results
        assert os.path.exists(results['cnn'])
        assert os.path.exists(results['vit'])

class TestModelQuantizer:
    """Test quantization functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dummy_models(self, temp_dir):
        """Generate dummy models for testing"""
        generator = DummyModelGenerator()
        return generator.create_dummy_models(temp_dir)

    @pytest.fixture
    def calibration_loader(self):
        """Create calibration data loader"""
        generator = DummyModelGenerator()
        return generator.create_calibration_loader(num_samples=50, batch_size=10)

    @pytest.fixture
    def quantizer(self):
        """Create model quantizer instance"""
        return ModelQuantizer()

    def test_dynamic_quantization_efficient_cnn(self, quantizer, dummy_models, temp_dir):
        """Test dynamic quantization of EfficientCNN"""
        quantized_path = quantizer.quantize_efficient_cnn(
            dummy_models['cnn'],
            temp_dir,
            quantization_type='dynamic',
            device='cpu'
        )

        assert os.path.exists(quantized_path)
        assert 'dynamic_quantized' in quantized_path

    def test_dynamic_quantization_vision_transformer(self, quantizer, dummy_models, temp_dir):
        """Test dynamic quantization of MedicalViT"""
        quantized_path = quantizer.quantize_vision_transformer(
            dummy_models['vit'],
            temp_dir,
            quantization_type='dynamic',
            device='cpu'
        )

        assert os.path.exists(quantized_path)
        assert 'dynamic_quantized' in quantized_path

    def test_static_quantization_efficient_cnn(self, quantizer, dummy_models, calibration_loader, temp_dir):
        """Test static quantization of EfficientCNN"""
        quantized_path = quantizer.quantize_efficient_cnn(
            dummy_models['cnn'],
            temp_dir,
            quantization_type='static',
            calibration_loader=calibration_loader,
            device='cpu'
        )

        assert os.path.exists(quantized_path)
        assert 'static_quantized' in quantized_path

class TestModelBenchmarker:
    """Test benchmarking functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dummy_models(self, temp_dir):
        """Generate dummy models for testing"""
        generator = DummyModelGenerator()
        return generator.create_dummy_models(temp_dir)

    @pytest.fixture
    def benchmarker(self):
        """Create model benchmarker instance"""
        return ModelBenchmarker(num_warmup_runs=2, num_benchmark_runs=5)  # Reduced for testing

    @pytest.fixture
    def test_input(self):
        """Create test input tensor"""
        return torch.randn(1, 3, 224, 224)

    def test_benchmark_pytorch_model(self, benchmarker, dummy_models, test_input):
        """Test benchmarking PyTorch model"""
        from ml.models.efficient_cnn import EfficientCNN

        # Load model
        model = EfficientCNN(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(dummy_models['cnn'], map_location='cpu'))
        model.eval()

        results = benchmarker.benchmark_pytorch_model(model, test_input, device='cpu')

        required_keys = ['mean_latency_ms', 'std_latency_ms', 'throughput', 'size_mb']
        for key in required_keys:
            assert key in results
            assert isinstance(results[key], (int, float))

    def test_get_model_size(self, benchmarker, dummy_models):
        """Test getting model file size"""
        size_info = benchmarker.get_model_size(dummy_models['cnn'])

        assert 'size_mb' in size_info
        assert 'size_bytes' in size_info
        assert size_info['size_mb'] > 0
        assert size_info['size_bytes'] > 0

    def test_compare_models(self, benchmarker):
        """Test model comparison functionality"""
        # Create mock benchmark results
        mock_results = {
            'model1': {
                'mean_latency_ms': 10.0,
                'throughput': 100.0,
                'size_mb': 50.0
            },
            'model2': {
                'mean_latency_ms': 20.0,
                'throughput': 50.0,
                'size_mb': 25.0
            }
        }

        comparison = benchmarker.compare_models(mock_results)

        assert 'fastest_model' in comparison
        assert 'highest_throughput_model' in comparison
        assert 'smallest_model' in comparison
        assert comparison['fastest_model'] == 'model1'
        assert comparison['highest_throughput_model'] == 'model1'
        assert comparison['smallest_model'] == 'model2'

class TestDummyModelGenerator:
    """Test dummy model generation"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def generator(self):
        """Create dummy model generator instance"""
        return DummyModelGenerator()

    def test_create_dummy_efficient_cnn(self, generator, temp_dir):
        """Test creating dummy EfficientCNN"""
        model_path = generator.create_dummy_efficient_cnn(temp_dir)

        assert os.path.exists(model_path)
        assert 'efficient_cnn' in model_path

        # Verify model can be loaded
        from ml.models.efficient_cnn import EfficientCNN
        model = EfficientCNN(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    def test_create_dummy_vision_transformer(self, generator, temp_dir):
        """Test creating dummy MedicalViT"""
        model_path = generator.create_dummy_vision_transformer(temp_dir)

        assert os.path.exists(model_path)
        assert 'vision_transformer' in model_path

        # Verify model can be loaded
        from ml.models.vision_transformer import MedicalViT
        model = MedicalViT(num_classes=3, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    def test_create_dummy_models(self, generator, temp_dir):
        """Test creating both dummy models"""
        models = generator.create_dummy_models(temp_dir)

        assert 'cnn' in models
        assert 'vit' in models
        assert os.path.exists(models['cnn'])
        assert os.path.exists(models['vit'])

    def test_create_random_data_sample(self, generator):
        """Test creating random data sample"""
        sample = generator.create_random_data_sample(batch_size=2)

        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (2, 3, 224, 224)

    def test_create_calibration_dataset(self, generator):
        """Test creating calibration dataset"""
        dataset = generator.create_calibration_dataset(num_samples=10)

        assert len(dataset) == 10

        # Test getting an item
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert image.shape == (3, 224, 224)
        assert 0 <= label <= 2  # num_classes = 3

    def test_generate_test_models(self, temp_dir):
        """Test the convenience function for generating test models"""
        result = generate_test_models(temp_dir)

        assert 'models' in result
        assert 'test_input' in result
        assert 'calibration_loader' in result
        assert 'output_dir' in result

        assert 'cnn' in result['models']
        assert 'vit' in result['models']
        assert isinstance(result['test_input'], torch.Tensor)
        assert hasattr(result['calibration_loader'], '__iter__')