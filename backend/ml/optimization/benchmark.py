import torch
import time
import numpy as np
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import logging
import onnxruntime as ort
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def gpu_memory_monitor():
    """Context manager to monitor GPU memory usage"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        yield
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        logger.info(f"Peak GPU memory usage: {peak_memory:.2f} MB")
    else:
        yield

class ModelBenchmarker:
    """Benchmarks model performance across different formats and optimizations"""

    def __init__(self, num_warmup_runs: int = 10, num_benchmark_runs: int = 100):
        self.num_warmup_runs = num_warmup_runs
        self.num_benchmark_runs = num_benchmark_runs

    def benchmark_pytorch_model(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Benchmark PyTorch model performance"""
        model.to(device)
        model.eval()
        input_tensor = input_tensor.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup_runs):
                _ = model(input_tensor)

        # Benchmark
        latencies = []
        with torch.no_grad(), gpu_memory_monitor():
            for _ in range(self.num_benchmark_runs):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # ms

        return {
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput": 1000 / np.mean(latencies)  # inferences per second
        }

    def benchmark_onnx_model(
        self,
        onnx_path: Union[str, Path],
        input_tensor: torch.Tensor,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """Benchmark ONNX model performance"""
        # Set up ONNX runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(str(onnx_path), providers=providers)

        # Prepare input
        input_name = session.get_inputs()[0].name
        input_data = input_tensor.numpy()

        # Warmup
        for _ in range(self.num_warmup_runs):
            _ = session.run(None, {input_name: input_data})

        # Benchmark
        latencies = []
        with gpu_memory_monitor():
            for _ in range(self.num_benchmark_runs):
                start_time = time.perf_counter()
                _ = session.run(None, {input_name: input_data})
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # ms

        return {
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput": 1000 / np.mean(latencies)  # inferences per second
        }

    def get_model_size(self, model_path: Union[str, Path]) -> Dict[str, float]:
        """Get model file size in MB"""
        path = Path(model_path)
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        return {
            "size_mb": size_mb,
            "size_bytes": size_bytes
        }

    def benchmark_comprehensive(
        self,
        models_config: Dict[str, Dict[str, Any]],
        input_tensor: torch.Tensor,
        device: str = "cpu"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Comprehensive benchmark comparing different model formats

        Args:
            models_config: Dict with model names as keys and config as values
                          Config should contain 'type' ('pytorch' or 'onnx') and 'path'
            input_tensor: Sample input tensor
            device: Device to run benchmarks on

        Returns:
            Dictionary with benchmark results for each model
        """
        results = {}

        for model_name, config in models_config.items():
            logger.info(f"Benchmarking {model_name}...")

            try:
                if config['type'] == 'pytorch':
                    # Load PyTorch model
                    model_class = config.get('model_class')
                    if model_class:
                        model = model_class()
                        model.load_state_dict(torch.load(config['path'], map_location=device))
                    else:
                        model = torch.load(config['path'], map_location=device)

                    performance = self.benchmark_pytorch_model(model, input_tensor, device)

                elif config['type'] == 'onnx':
                    performance = self.benchmark_onnx_model(config['path'], input_tensor, device)

                else:
                    logger.warning(f"Unknown model type: {config['type']}")
                    continue

                # Get model size
                size_info = self.get_model_size(config['path'])

                results[model_name] = {
                    **performance,
                    **size_info,
                    "model_type": config['type'],
                    "device": device
                }

            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {e}")
                results[model_name] = {"error": str(e)}

        return results

    def compare_models(self, benchmark_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare benchmark results and provide summary"""
        if not benchmark_results:
            return {"error": "No benchmark results to compare"}

        # Filter out failed benchmarks
        valid_results = {k: v for k, v in benchmark_results.items() if "error" not in v}

        if not valid_results:
            return {"error": "No valid benchmark results"}

        # Find best models for different metrics
        comparisons = {}

        # Latency comparison
        latencies = {k: v["mean_latency_ms"] for k, v in valid_results.items()}
        comparisons["fastest_model"] = min(latencies, key=latencies.get)
        comparisons["slowest_model"] = max(latencies, key=latencies.get)
        comparisons["latency_improvement"] = latencies[comparisons["slowest_model"]] / latencies[comparisons["fastest_model"]]

        # Throughput comparison
        throughputs = {k: v["throughput"] for k, v in valid_results.items()}
        comparisons["highest_throughput_model"] = max(throughputs, key=throughputs.get)
        comparisons["lowest_throughput_model"] = min(throughputs, key=throughputs.get)
        comparisons["throughput_improvement"] = throughputs[comparisons["highest_throughput_model"]] / throughputs[comparisons["lowest_throughput_model"]]

        # Size comparison
        sizes = {k: v["size_mb"] for k, v in valid_results.items()}
        comparisons["smallest_model"] = min(sizes, key=sizes.get)
        comparisons["largest_model"] = max(sizes, key=sizes.get)
        comparisons["size_reduction"] = sizes[comparisons["largest_model"]] / sizes[comparisons["smallest_model"]]

        return comparisons

def run_medical_model_benchmark(
    original_cnn_path: Optional[str] = None,
    original_vit_path: Optional[str] = None,
    quantized_cnn_path: Optional[str] = None,
    quantized_vit_path: Optional[str] = None,
    onnx_cnn_path: Optional[str] = None,
    onnx_vit_path: Optional[str] = None,
    device: str = "cpu",
    output_dir: str = "benchmark_results"
) -> Dict[str, Any]:
    """Run comprehensive benchmark for medical models"""
    from ..models.efficient_cnn import EfficientCNN
    from ..models.vision_transformer import MedicalViT

    benchmarker = ModelBenchmarker()
    input_tensor = torch.randn(1, 3, 224, 224)

    # Configure models to benchmark
    models_config = {}

    if original_cnn_path:
        models_config["cnn_original"] = {
            "type": "pytorch",
            "path": original_cnn_path,
            "model_class": lambda: EfficientCNN(num_classes=3, pretrained=False)
        }

    if original_vit_path:
        models_config["vit_original"] = {
            "type": "pytorch",
            "path": original_vit_path,
            "model_class": lambda: MedicalViT(num_classes=3, pretrained=False)
        }

    if quantized_cnn_path:
        models_config["cnn_quantized"] = {
            "type": "pytorch",
            "path": quantized_cnn_path,
            "model_class": lambda: EfficientCNN(num_classes=3, pretrained=False)
        }

    if quantized_vit_path:
        models_config["vit_quantized"] = {
            "type": "pytorch",
            "path": quantized_vit_path,
            "model_class": lambda: MedicalViT(num_classes=3, pretrained=False)
        }

    if onnx_cnn_path:
        models_config["cnn_onnx"] = {
            "type": "onnx",
            "path": onnx_cnn_path
        }

    if onnx_vit_path:
        models_config["vit_onnx"] = {
            "type": "onnx",
            "path": onnx_vit_path
        }

    # Run benchmarks
    results = benchmarker.benchmark_comprehensive(models_config, input_tensor, device)

    # Generate comparison
    comparison = benchmarker.compare_models(results)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import json
    with open(output_path / "benchmark_results.json", "w") as f:
        json.dump({"results": results, "comparison": comparison}, f, indent=2)

    logger.info(f"Benchmark results saved to {output_path / 'benchmark_results.json'}")

    return {
        "results": results,
        "comparison": comparison,
        "output_dir": str(output_path)
    }