#!/usr/bin/env python3
"""
Manual test script for optimization functionality
Run this to test the optimization features without pytest
"""
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

def test_dummy_model_generation():
    """Test dummy model generation"""
    print("Testing dummy model generation...")

    try:
        from ml.models.dummy_generator import generate_test_models

        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_test_models(temp_dir)

            assert 'models' in result
            assert 'cnn' in result['models']
            assert 'vit' in result['models']
            assert os.path.exists(result['models']['cnn'])
            assert os.path.exists(result['models']['vit'])

            print("✓ Dummy model generation works")

    except Exception as e:
        print(f"✗ Dummy model generation failed: {e}")
        return False

    return True

def test_export_functionality():
    """Test ONNX export functionality"""
    print("Testing ONNX export functionality...")

    try:
        from ml.optimization.export import ModelExporter
        from ml.models.dummy_generator import generate_test_models

        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate dummy models
            models = generate_test_models(temp_dir)

            # Test export
            exporter = ModelExporter()

            # Export CNN
            cnn_onnx = exporter.export_efficient_cnn(
                models['models']['cnn'],
                temp_dir,
                device='cpu'
            )
            assert os.path.exists(cnn_onnx)

            # Export ViT
            vit_onnx = exporter.export_vision_transformer(
                models['models']['vit'],
                temp_dir,
                device='cpu'
            )
            assert os.path.exists(vit_onnx)

            print("✓ ONNX export functionality works")

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False

    return True

def test_quantization_functionality():
    """Test quantization functionality"""
    print("Testing quantization functionality...")

    try:
        from ml.optimization.quantization import ModelQuantizer
        from ml.models.dummy_generator import generate_test_models

        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate dummy models
            models = generate_test_models(temp_dir)

            # Test quantization
            quantizer = ModelQuantizer()

            # Quantize CNN
            quantized_cnn = quantizer.quantize_efficient_cnn(
                models['models']['cnn'],
                temp_dir,
                quantization_type='dynamic',
                device='cpu'
            )
            assert os.path.exists(quantized_cnn)

            print("✓ Quantization functionality works")

    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return False

    return True

def test_benchmark_functionality():
    """Test benchmarking functionality"""
    print("Testing benchmark functionality...")

    try:
        from ml.optimization.benchmark import ModelBenchmarker
        from ml.models.dummy_generator import DummyModelGenerator

        generator = DummyModelGenerator()
        benchmarker = ModelBenchmarker(num_warmup_runs=1, num_benchmark_runs=2)
        test_input = generator.create_random_data_sample()

        # Create a simple model for testing
        import torch.nn as nn
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 3)
        )
        model.eval()

        # Test benchmarking
        results = benchmarker.benchmark_pytorch_model(model, test_input, device='cpu')

        required_keys = ['mean_latency_ms', 'throughput']
        for key in required_keys:
            assert key in results

        print("✓ Benchmark functionality works")

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False

    return True

def main():
    """Run all tests"""
    print("Running optimization functionality tests...\n")

    tests = [
        test_dummy_model_generation,
        test_export_functionality,
        test_quantization_functionality,
        test_benchmark_functionality,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())