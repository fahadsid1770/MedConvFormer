#!/usr/bin/env python3
"""
Command-line interface for model optimization and export utilities
"""
import argparse
import sys
from pathlib import Path
import logging

from .export import ModelExporter
from .quantization import ModelQuantizer
from .benchmark import run_medical_model_benchmark
from ..models.dummy_generator import generate_test_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(description="Model Optimization and Export CLI")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export models to ONNX')
    export_parser.add_argument('--model-type', choices=['cnn', 'vit', 'hybrid'],
                              required=True, help='Type of model to export')
    export_parser.add_argument('--cnn-path', help='Path to CNN model (required for cnn and hybrid)')
    export_parser.add_argument('--vit-path', help='Path to ViT model (required for vit and hybrid)')
    export_parser.add_argument('--output-dir', default='exports',
                              help='Output directory for ONNX models')
    export_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                              help='Device to use for export')

    # Quantize command
    quantize_parser = subparsers.add_parser('quantize', help='Quantize models')
    quantize_parser.add_argument('--model-type', choices=['cnn', 'vit'],
                                required=True, help='Type of model to quantize')
    quantize_parser.add_argument('--model-path', required=True,
                                help='Path to model to quantize')
    quantize_parser.add_argument('--quantization-type', choices=['dynamic', 'static'],
                                default='dynamic', help='Quantization type')
    quantize_parser.add_argument('--calibration-data', help='Path to calibration dataset (required for static quantization)')
    quantize_parser.add_argument('--output-dir', default='quantized',
                                help='Output directory for quantized models')
    quantize_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                                help='Device to use for quantization')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark models')
    benchmark_parser.add_argument('--original-cnn', help='Path to original CNN model')
    benchmark_parser.add_argument('--original-vit', help='Path to original ViT model')
    benchmark_parser.add_argument('--quantized-cnn', help='Path to quantized CNN model')
    benchmark_parser.add_argument('--quantized-vit', help='Path to quantized ViT model')
    benchmark_parser.add_argument('--onnx-cnn', help='Path to ONNX CNN model')
    benchmark_parser.add_argument('--onnx-vit', help='Path to ONNX ViT model')
    benchmark_parser.add_argument('--output-dir', default='benchmark_results',
                                help='Output directory for benchmark results')
    benchmark_parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                                help='Device to use for benchmarking')

    # Generate dummy models command
    dummy_parser = subparsers.add_parser('generate-dummy', help='Generate dummy models for testing')
    dummy_parser.add_argument('--output-dir', default='test_models',
                             help='Output directory for dummy models')

    return parser

def handle_export(args):
    """Handle export command"""
    exporter = ModelExporter()

    try:
        if args.model_type == 'cnn':
            if not args.cnn_path:
                logger.error("CNN path required for CNN export")
                return 1
            onnx_path = exporter.export_efficient_cnn(args.cnn_path, args.output_dir, args.device)
            logger.info(f"Exported CNN model to: {onnx_path}")

        elif args.model_type == 'vit':
            if not args.vit_path:
                logger.error("ViT path required for ViT export")
                return 1
            onnx_path = exporter.export_vision_transformer(args.vit_path, args.output_dir, args.device)
            logger.info(f"Exported ViT model to: {onnx_path}")

        elif args.model_type == 'hybrid':
            if not args.cnn_path or not args.vit_path:
                logger.error("Both CNN and ViT paths required for hybrid export")
                return 1
            results = exporter.export_hybrid_system(args.cnn_path, args.vit_path, args.output_dir, args.device)
            logger.info(f"Exported hybrid system models to: {results}")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1

    return 0

def handle_quantize(args):
    """Handle quantize command"""
    quantizer = ModelQuantizer()

    try:
        if args.model_type == 'cnn':
            quantized_path = quantizer.quantize_efficient_cnn(
                args.model_path,
                args.output_dir,
                args.quantization_type,
                device=args.device
            )
            logger.info(f"Quantized CNN model saved to: {quantized_path}")

        elif args.model_type == 'vit':
            quantized_path = quantizer.quantize_vision_transformer(
                args.model_path,
                args.output_dir,
                args.quantization_type,
                device=args.device
            )
            logger.info(f"Quantized ViT model saved to: {quantized_path}")

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return 1

    return 0

def handle_benchmark(args):
    """Handle benchmark command"""
    try:
        results = run_medical_model_benchmark(
            original_cnn_path=args.original_cnn,
            original_vit_path=args.original_vit,
            quantized_cnn_path=args.quantized_cnn,
            quantized_vit_path=args.quantized_vit,
            onnx_cnn_path=args.onnx_cnn,
            onnx_vit_path=args.onnx_vit,
            device=args.device,
            output_dir=args.output_dir
        )

        logger.info(f"Benchmark results saved to: {results['output_dir']}")

        # Print summary
        comparison = results['comparison']
        print("\n=== Benchmark Summary ===")
        print(f"Fastest model: {comparison.get('fastest_model', 'N/A')}")
        print(f"Highest throughput: {comparison.get('highest_throughput_model', 'N/A')}")
        print(f"Smallest model: {comparison.get('smallest_model', 'N/A')}")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1

    return 0

def handle_generate_dummy(args):
    """Handle generate-dummy command"""
    try:
        result = generate_test_models(args.output_dir)
        logger.info(f"Generated dummy models in: {result['output_dir']}")
        logger.info(f"CNN model: {result['models']['cnn']}")
        logger.info(f"ViT model: {result['models']['vit']}")

    except Exception as e:
        logger.error(f"Dummy model generation failed: {e}")
        return 1

    return 0

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'export':
        return handle_export(args)
    elif args.command == 'quantize':
        return handle_quantize(args)
    elif args.command == 'benchmark':
        return handle_benchmark(args)
    elif args.command == 'generate-dummy':
        return handle_generate_dummy(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1

if __name__ == '__main__':
    sys.exit(main())