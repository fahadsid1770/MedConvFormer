# Model Optimization and Export

This module provides comprehensive tools for optimizing and exporting PyTorch models for production deployment.

## Features

- **ONNX Export**: Convert PyTorch models to ONNX format for cross-platform inference
- **Quantization**: Apply post-training quantization to reduce model size and improve inference speed
- **Benchmarking**: Compare performance across different model formats and optimizations
- **Dummy Model Generation**: Create test models for development and testing

## Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Additional dependencies for optimization:
- `onnxruntime` - For ONNX model inference and export
- `psutil` - For system monitoring during benchmarking
- `gputil` - For GPU monitoring (optional)

## Quick Start

### 1. Generate Dummy Models for Testing

```python
from ml.models.dummy_generator import generate_test_models

# Generate test models and data
result = generate_test_models("test_models")
print(f"Models saved to: {result['output_dir']}")
```

### 2. Export Models to ONNX

```python
from ml.optimization.export import ModelExporter

exporter = ModelExporter()

# Export individual models
cnn_onnx = exporter.export_efficient_cnn("path/to/cnn_model.pth", "exports")
vit_onnx = exporter.export_vision_transformer("path/to/vit_model.pth", "exports")

# Export hybrid system components
results = exporter.export_hybrid_system("cnn_model.pth", "vit_model.pth", "exports")
```

### 3. Quantize Models

```python
from ml.optimization.quantization import ModelQuantizer

quantizer = ModelQuantizer()

# Dynamic quantization (no calibration data needed)
quantized_cnn = quantizer.quantize_efficient_cnn(
    "path/to/cnn_model.pth",
    "quantized_models",
    quantization_type="dynamic"
)

# Static quantization (requires calibration data)
from ml.models.dummy_generator import DummyModelGenerator
generator = DummyModelGenerator()
calibration_loader = generator.create_calibration_loader()

quantized_cnn_static = quantizer.quantize_efficient_cnn(
    "path/to/cnn_model.pth",
    "quantized_models",
    quantization_type="static",
    calibration_loader=calibration_loader
)
```

### 4. Benchmark Models

```python
from ml.optimization.benchmark import run_medical_model_benchmark

# Compare multiple model variants
results = run_medical_model_benchmark(
    original_cnn_path="models/cnn_original.pth",
    original_vit_path="models/vit_original.pth",
    quantized_cnn_path="models/cnn_quantized.pth",
    onnx_cnn_path="models/cnn.onnx",
    device="cpu",
    output_dir="benchmark_results"
)

print("Benchmark complete. Results saved to benchmark_results/")
```

## Command Line Interface

Use the CLI for common optimization tasks:

```bash
# Generate dummy models
python -m ml.optimization.cli generate-dummy --output-dir test_models

# Export models to ONNX
python -m ml.optimization.cli export --model-type cnn --cnn-path model.pth --output-dir exports

# Quantize models
python -m ml.optimization.cli quantize --model-type cnn --model-path model.pth --quantization-type dynamic --output-dir quantized

# Run benchmarks
python -m ml.optimization.cli benchmark --original-cnn cnn.pth --quantized-cnn cnn_quantized.pth --output-dir results
```

## API Reference

### ModelExporter

- `export_efficient_cnn(model_path, output_path, device='cpu')` - Export EfficientCNN to ONNX
- `export_vision_transformer(model_path, output_path, device='cpu')` - Export MedicalViT to ONNX
- `export_hybrid_system(cnn_path, vit_path, output_path, device='cpu')` - Export hybrid system components

### ModelQuantizer

- `quantize_efficient_cnn(model_path, output_path, quantization_type='dynamic', calibration_loader=None, device='cpu')`
- `quantize_vision_transformer(model_path, output_path, quantization_type='dynamic', calibration_loader=None, device='cpu')`

### ModelBenchmarker

- `benchmark_pytorch_model(model, input_tensor, device='cpu')` - Benchmark PyTorch model
- `benchmark_onnx_model(onnx_path, input_tensor, device='cpu')` - Benchmark ONNX model
- `compare_models(results)` - Compare benchmark results

### DummyModelGenerator

- `create_dummy_models(output_path)` - Create both CNN and ViT dummy models
- `create_calibration_loader()` - Create DataLoader for quantization calibration
- `create_random_data_sample()` - Generate random test input

## Model Compatibility

The optimization tools are designed to work with:

- **EfficientCNN**: Uses EfficientNet-B0 backbone with custom classification head
- **MedicalViT**: Uses Vision Transformer with attention pooling
- **HybridClassifier**: Ensemble of CNN and ViT models

All models include `get_input_spec()` methods for ONNX export compatibility.

## Performance Optimization Tips

1. **ONNX Export**:
   - Use `opset_version=11` for best compatibility
   - Enable `do_constant_folding=True` for optimization
   - Set appropriate `dynamic_axes` for variable batch sizes

2. **Quantization**:
   - Dynamic quantization is simpler and works well for most cases
   - Static quantization requires calibration data but can be more accurate
   - Test quantized models thoroughly as accuracy may decrease slightly

3. **Benchmarking**:
   - Use sufficient warmup runs for stable measurements
   - Run benchmarks multiple times and average results
   - Consider both latency and throughput metrics

## Testing

Run the manual test script to verify functionality:

```bash
python test_optimization_manual.py
```

Or run the full test suite:

```bash
python -m pytest tests/test_optimization.py -v
```

## Troubleshooting

**ONNX Export Issues**:
- Ensure models are in eval mode before export
- Check that all operations are ONNX-compatible
- Use appropriate input shapes and data types

**Quantization Issues**:
- Some layers may not support quantization
- Calibration data should be representative of real data
- Test accuracy after quantization

**Benchmarking Issues**:
- GPU memory monitoring requires CUDA
- System monitoring may require additional permissions
- Use consistent batch sizes across comparisons