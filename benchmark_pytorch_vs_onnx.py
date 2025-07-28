#!/usr/bin/env python3
"""
UAV Landing System - PyTorch vs ONNX Performance Benchmark
=========================================================

Comprehensive benchmarking script to compare performance between PyTorch and ONNX
models for UAV landing detection.

Features:
- Side-by-side PyTorch vs ONNX inference timing
- Accuracy validation (ensuring identical results)
- Memory usage monitoring
- Detailed performance analysis
- Export results with visualizations

Usage:
    python benchmark_pytorch_vs_onnx.py --stage 2
    python benchmark_pytorch_vs_onnx.py --stage 3
    python benchmark_pytorch_vs_onnx.py --all --detailed
"""

import os
import sys
import argparse
import torch
import numpy as np
import cv2
import time
import json
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️  ONNX Runtime not found. Install with: pip install onnxruntime")

# Local imports
from models.mobilenetv3_edge_model import create_edge_model
from datasets.semantic_drone_dataset import create_semantic_drone_transforms
from inference import UAVLandingInference
from onnx_inference_neurosymbolic import ONNXNeuroSymbolicInference


class PerformanceBenchmark:
    """Comprehensive performance benchmark for PyTorch vs ONNX models."""
    
    def __init__(self, input_size: Tuple[int, int] = (512, 512)):
        self.input_size = input_size
        self.results = {}
        
        # Check requirements
        if not HAS_ONNX:
            raise ImportError("ONNX Runtime required for benchmarking")
        
        print(f"🏁 Performance Benchmark Initialized")
        print(f"   Input size: {input_size}")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   ONNX Runtime version: {ort.__version__}")
    
    def load_pytorch_model(self, checkpoint_path: str) -> torch.nn.Module:
        """Load PyTorch model from checkpoint."""
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"PyTorch checkpoint not found: {checkpoint_path}")
        
        print(f"📥 Loading PyTorch model: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model
        model = create_edge_model(
            model_type='enhanced',
            num_classes=6,
            use_uncertainty=True,
            pretrained=False
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"✅ PyTorch model loaded on {device}")
        
        return model, device
    
    def load_onnx_model(self, onnx_path: str) -> ort.InferenceSession:
        """Load ONNX model session."""
        
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        print(f"📥 Loading ONNX model: {onnx_path}")
        
        # Configure session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # Use all cores
        
        # Determine providers
        available_providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        
        print(f"✅ ONNX model loaded with providers: {providers}")
        
        return session
    
    def create_test_data(self, num_samples: int = 100) -> List[np.ndarray]:
        """Create test dataset for benchmarking."""
        
        print(f"🎲 Generating {num_samples} test samples...")
        
        # Create transforms
        transform = create_semantic_drone_transforms(
            image_size=self.input_size,
            augment=False
        )
        
        test_images = []
        
        for i in range(num_samples):
            # Generate random image
            random_image = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
            test_images.append(random_image)
        
        print(f"✅ Test dataset created")
        
        return test_images
    
    def benchmark_pytorch_inference(
        self, 
        model: torch.nn.Module, 
        device: torch.device,
        test_images: List[np.ndarray],
        warmup_iterations: int = 10
    ) -> Dict:
        """Benchmark PyTorch model inference."""
        
        print(f"⏱️  Benchmarking PyTorch inference...")
        
        # Create transform
        transform = create_semantic_drone_transforms(
            image_size=self.input_size,
            augment=False
        )
        
        # Warmup
        print(f"   Warming up ({warmup_iterations} iterations)...")
        warmup_image = test_images[0]
        
        for _ in range(warmup_iterations):
            with torch.no_grad():
                # Preprocess
                if hasattr(transform, 'replay'):
                    transformed = transform(image=warmup_image)
                    tensor = transformed['image']
                else:
                    tensor = transform(warmup_image)
                
                input_tensor = tensor.unsqueeze(0).to(device)
                
                # Inference
                _ = model(input_tensor)
        
        # Synchronize GPU if available
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        print(f"   Running benchmark ({len(test_images)} samples)...")
        times = []
        memory_usage = []
        
        for i, image in enumerate(test_images):
            # Monitor memory before inference
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated(device)
            else:
                memory_before = psutil.virtual_memory().used
            
            # Time inference
            start_time = time.time()
            
            with torch.no_grad():
                # Preprocess
                if hasattr(transform, 'replay'):
                    transformed = transform(image=image)
                    tensor = transformed['image']
                else:
                    tensor = transform(image)
                
                input_tensor = tensor.unsqueeze(0).to(device)
                
                # Inference
                outputs = model(input_tensor)
            
            # Synchronize and measure time
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            # Monitor memory after inference
            if device.type == 'cuda':
                memory_after = torch.cuda.memory_allocated(device)
                memory_used = (memory_after - memory_before) / (1024 ** 2)  # MB
            else:
                memory_after = psutil.virtual_memory().used
                memory_used = (memory_after - memory_before) / (1024 ** 2)  # MB
            
            memory_usage.append(memory_used)
            
            if (i + 1) % 25 == 0:
                print(f"   Progress: {i+1}/{len(test_images)}")
        
        # Calculate statistics
        times_ms = [t * 1000 for t in times]
        
        stats = {
            'framework': 'PyTorch',
            'device': str(device),
            'num_samples': len(test_images),
            'mean_ms': float(np.mean(times_ms)),
            'std_ms': float(np.std(times_ms)),
            'min_ms': float(np.min(times_ms)),
            'max_ms': float(np.max(times_ms)),
            'median_ms': float(np.median(times_ms)),
            'p95_ms': float(np.percentile(times_ms, 95)),
            'p99_ms': float(np.percentile(times_ms, 99)),
            'fps': float(1000 / np.mean(times_ms)),
            'memory_mb_mean': float(np.mean(memory_usage)),
            'memory_mb_max': float(np.max(memory_usage)),
            'times_ms': times_ms,
            'memory_usage_mb': memory_usage
        }
        
        print(f"✅ PyTorch benchmark complete:")
        print(f"   Mean: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        print(f"   FPS: {stats['fps']:.1f}")
        print(f"   Memory: {stats['memory_mb_mean']:.1f} MB (avg)")
        
        return stats
    
    def benchmark_onnx_inference(
        self,
        session: ort.InferenceSession,
        test_images: List[np.ndarray],
        warmup_iterations: int = 10
    ) -> Dict:
        """Benchmark ONNX model inference."""
        
        print(f"⏱️  Benchmarking ONNX inference...")
        
        # Create transform
        transform = create_semantic_drone_transforms(
            image_size=self.input_size,
            augment=False
        )
        
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Warmup
        print(f"   Warming up ({warmup_iterations} iterations)...")
        warmup_image = test_images[0]
        
        for _ in range(warmup_iterations):
            # Preprocess
            if hasattr(transform, 'replay'):
                transformed = transform(image=warmup_image)
                tensor = transformed['image']
                input_array = tensor.numpy()
            else:
                tensor = transform(warmup_image)
                input_array = tensor.numpy()
            
            if len(input_array.shape) == 3:
                input_array = np.expand_dims(input_array, axis=0)
            
            input_array = input_array.astype(np.float32)
            
            # Inference
            _ = session.run(None, {input_name: input_array})
        
        # Benchmark
        print(f"   Running benchmark ({len(test_images)} samples)...")
        times = []
        memory_usage = []
        
        for i, image in enumerate(test_images):
            # Monitor memory before inference
            memory_before = psutil.virtual_memory().used
            
            # Time inference
            start_time = time.time()
            
            # Preprocess
            if hasattr(transform, 'replay'):
                transformed = transform(image=image)
                tensor = transformed['image']
                input_array = tensor.numpy()
            else:
                tensor = transform(image)
                input_array = tensor.numpy()
            
            if len(input_array.shape) == 3:
                input_array = np.expand_dims(input_array, axis=0)
            
            input_array = input_array.astype(np.float32)
            
            # Inference
            outputs = session.run(None, {input_name: input_array})
            
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            # Monitor memory after inference
            memory_after = psutil.virtual_memory().used
            memory_used = (memory_after - memory_before) / (1024 ** 2)  # MB
            memory_usage.append(memory_used)
            
            if (i + 1) % 25 == 0:
                print(f"   Progress: {i+1}/{len(test_images)}")
        
        # Calculate statistics
        times_ms = [t * 1000 for t in times]
        
        stats = {
            'framework': 'ONNX',
            'device': 'CPU/CUDA' if 'CUDAExecutionProvider' in [p.provider for p in session.get_providers()] else 'CPU',
            'num_samples': len(test_images),
            'mean_ms': float(np.mean(times_ms)),
            'std_ms': float(np.std(times_ms)),
            'min_ms': float(np.min(times_ms)),
            'max_ms': float(np.max(times_ms)),
            'median_ms': float(np.median(times_ms)),
            'p95_ms': float(np.percentile(times_ms, 95)),
            'p99_ms': float(np.percentile(times_ms, 99)),
            'fps': float(1000 / np.mean(times_ms)),
            'memory_mb_mean': float(np.mean(memory_usage)),
            'memory_mb_max': float(np.max(memory_usage)),
            'times_ms': times_ms,
            'memory_usage_mb': memory_usage
        }
        
        print(f"✅ ONNX benchmark complete:")
        print(f"   Mean: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        print(f"   FPS: {stats['fps']:.1f}")
        print(f"   Memory: {stats['memory_mb_mean']:.1f} MB (avg)")
        
        return stats
    
    def validate_accuracy(
        self,
        pytorch_model: torch.nn.Module,
        pytorch_device: torch.device,
        onnx_session: ort.InferenceSession,
        test_images: List[np.ndarray],
        tolerance: float = 1e-4
    ) -> Dict:
        """Validate that PyTorch and ONNX models produce identical results."""
        
        print(f"🔍 Validating model accuracy...")
        
        transform = create_semantic_drone_transforms(
            image_size=self.input_size,
            augment=False
        )
        
        input_name = onnx_session.get_inputs()[0].name
        
        validation_results = {
            'total_samples': 0,
            'matches': 0,
            'max_difference': 0.0,
            'mean_difference': 0.0,
            'differences': []
        }
        
        num_validation_samples = min(20, len(test_images))  # Validate subset
        
        for i, image in enumerate(test_images[:num_validation_samples]):
            # PyTorch inference
            with torch.no_grad():
                if hasattr(transform, 'replay'):
                    transformed = transform(image=image)
                    tensor = transformed['image']
                else:
                    tensor = transform(image)
                
                pytorch_input = tensor.unsqueeze(0).to(pytorch_device)
                pytorch_outputs = pytorch_model(pytorch_input)
                
                if isinstance(pytorch_outputs, dict):
                    pytorch_main = pytorch_outputs['main'].cpu().numpy()
                else:
                    pytorch_main = pytorch_outputs.cpu().numpy()
            
            # ONNX inference
            if hasattr(transform, 'replay'):
                transformed = transform(image=image)
                tensor = transformed['image']
                onnx_input = tensor.numpy()
            else:
                tensor = transform(image)
                onnx_input = tensor.numpy()
            
            if len(onnx_input.shape) == 3:
                onnx_input = np.expand_dims(onnx_input, axis=0)
            
            onnx_input = onnx_input.astype(np.float32)
            onnx_outputs = onnx_session.run(None, {input_name: onnx_input})
            onnx_main = onnx_outputs[0]
            
            # Compare outputs
            diff = np.abs(pytorch_main - onnx_main).max()
            validation_results['differences'].append(float(diff))
            validation_results['max_difference'] = max(validation_results['max_difference'], diff)
            
            if diff < tolerance:
                validation_results['matches'] += 1
            
            validation_results['total_samples'] += 1
        
        validation_results['mean_difference'] = float(np.mean(validation_results['differences']))
        validation_results['accuracy'] = validation_results['matches'] / validation_results['total_samples']
        
        print(f"✅ Accuracy validation complete:")
        print(f"   Samples: {validation_results['total_samples']}")
        print(f"   Matches: {validation_results['matches']}/{validation_results['total_samples']}")
        print(f"   Accuracy: {validation_results['accuracy']*100:.1f}%")
        print(f"   Max difference: {validation_results['max_difference']:.6f}")
        print(f"   Mean difference: {validation_results['mean_difference']:.6f}")
        
        return validation_results
    
    def compare_performance(self, pytorch_stats: Dict, onnx_stats: Dict) -> Dict:
        """Compare performance between PyTorch and ONNX."""
        
        comparison = {
            'speedup_factor': pytorch_stats['mean_ms'] / onnx_stats['mean_ms'],
            'fps_improvement': onnx_stats['fps'] / pytorch_stats['fps'],
            'memory_reduction_mb': pytorch_stats['memory_mb_mean'] - onnx_stats['memory_mb_mean'],
            'pytorch_stats': pytorch_stats,
            'onnx_stats': onnx_stats
        }
        
        print(f"📊 Performance Comparison:")
        print(f"   Speedup: {comparison['speedup_factor']:.2f}x")
        print(f"   FPS improvement: {comparison['fps_improvement']:.2f}x")
        print(f"   Memory reduction: {comparison['memory_reduction_mb']:.1f} MB")
        
        return comparison
    
    def create_visualizations(self, results: Dict, output_dir: str):
        """Create comprehensive performance visualizations."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📈 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Inference time comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar plot - mean inference time
        frameworks = ['PyTorch', 'ONNX']
        mean_times = [results['pytorch_stats']['mean_ms'], results['onnx_stats']['mean_ms']]
        std_times = [results['pytorch_stats']['std_ms'], results['onnx_stats']['std_ms']]
        
        axes[0, 0].bar(frameworks, mean_times, yerr=std_times, capsize=10, 
                       color=['#ff7f0e', '#2ca02c'], alpha=0.7)
        axes[0, 0].set_title('Mean Inference Time')
        axes[0, 0].set_ylabel('Time (ms)')
        
        # Box plot - inference time distribution
        pytorch_times = results['pytorch_stats']['times_ms']
        onnx_times = results['onnx_stats']['times_ms']
        
        axes[0, 1].boxplot([pytorch_times, onnx_times], labels=frameworks)
        axes[0, 1].set_title('Inference Time Distribution')
        axes[0, 1].set_ylabel('Time (ms)')
        
        # FPS comparison
        fps_values = [results['pytorch_stats']['fps'], results['onnx_stats']['fps']]
        axes[1, 0].bar(frameworks, fps_values, color=['#ff7f0e', '#2ca02c'], alpha=0.7)
        axes[1, 0].set_title('Frames Per Second (FPS)')
        axes[1, 0].set_ylabel('FPS')
        
        # Memory usage comparison
        memory_values = [results['pytorch_stats']['memory_mb_mean'], results['onnx_stats']['memory_mb_mean']]
        axes[1, 1].bar(frameworks, memory_values, color=['#ff7f0e', '#2ca02c'], alpha=0.7)
        axes[1, 1].set_title('Memory Usage')
        axes[1, 1].set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Time series plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pytorch = range(len(pytorch_times))
        x_onnx = range(len(onnx_times))
        
        ax.plot(x_pytorch, pytorch_times, label='PyTorch', alpha=0.7, color='#ff7f0e')
        ax.plot(x_onnx, onnx_times, label='ONNX', alpha=0.7, color='#2ca02c')
        
        ax.set_title('Inference Time Over Iterations')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'time_series.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Performance summary table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = [
            ['Metric', 'PyTorch', 'ONNX', 'Improvement'],
            ['Mean Time (ms)', f"{results['pytorch_stats']['mean_ms']:.2f}", 
             f"{results['onnx_stats']['mean_ms']:.2f}", 
             f"{results['speedup_factor']:.2f}x"],
            ['FPS', f"{results['pytorch_stats']['fps']:.1f}", 
             f"{results['onnx_stats']['fps']:.1f}", 
             f"{results['fps_improvement']:.2f}x"],
            ['Memory (MB)', f"{results['pytorch_stats']['memory_mb_mean']:.1f}", 
             f"{results['onnx_stats']['memory_mb_mean']:.1f}", 
             f"{results['memory_reduction_mb']:.1f} MB less"],
            ['P95 Time (ms)', f"{results['pytorch_stats']['p95_ms']:.2f}", 
             f"{results['onnx_stats']['p95_ms']:.2f}", ''],
            ['P99 Time (ms)', f"{results['onnx_stats']['p99_ms']:.2f}", 
             f"{results['onnx_stats']['p99_ms']:.2f}", '']
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Performance Comparison Summary', fontsize=16, weight='bold', pad=20)
        
        plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualizations saved to {output_dir}")
    
    def run_complete_benchmark(
        self,
        pytorch_checkpoint: str,
        onnx_model: str,
        num_samples: int = 100,
        output_dir: str = "benchmark_results"
    ) -> Dict:
        """Run complete benchmark comparing PyTorch and ONNX models."""
        
        print(f"\n🚀 Starting Complete Benchmark")
        print(f"   PyTorch model: {pytorch_checkpoint}")
        print(f"   ONNX model: {onnx_model}")
        print(f"   Test samples: {num_samples}")
        
        # Load models
        pytorch_model, pytorch_device = self.load_pytorch_model(pytorch_checkpoint)
        onnx_session = self.load_onnx_model(onnx_model)
        
        # Create test data
        test_images = self.create_test_data(num_samples)
        
        # Benchmark PyTorch
        pytorch_stats = self.benchmark_pytorch_inference(
            pytorch_model, pytorch_device, test_images
        )
        
        # Benchmark ONNX
        onnx_stats = self.benchmark_onnx_inference(onnx_session, test_images)
        
        # Validate accuracy
        accuracy_results = self.validate_accuracy(
            pytorch_model, pytorch_device, onnx_session, test_images
        )
        
        # Compare performance
        comparison = self.compare_performance(pytorch_stats, onnx_stats)
        
        # Combine all results
        complete_results = {
            **comparison,
            'accuracy_validation': accuracy_results,
            'benchmark_config': {
                'pytorch_checkpoint': pytorch_checkpoint,
                'onnx_model': onnx_model,
                'num_samples': num_samples,
                'input_size': self.input_size,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results_file = output_path / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            # Remove large arrays for JSON
            json_results = complete_results.copy()
            json_results['pytorch_stats'] = {k: v for k, v in pytorch_stats.items() 
                                           if k not in ['times_ms', 'memory_usage_mb']}
            json_results['onnx_stats'] = {k: v for k, v in onnx_stats.items() 
                                        if k not in ['times_ms', 'memory_usage_mb']}
            
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved: {results_file}")
        
        # Create visualizations
        self.create_visualizations(complete_results, output_dir)
        
        return complete_results


def main():
    parser = argparse.ArgumentParser(description='PyTorch vs ONNX Performance Benchmark')
    
    # Model selection
    parser.add_argument('--stage', type=int, choices=[2, 3], 
                        help='Model stage to benchmark (2 or 3)')
    parser.add_argument('--all', action='store_true', 
                        help='Benchmark all available stages')
    parser.add_argument('--pytorch_model', type=str, 
                        help='Custom PyTorch checkpoint path')
    parser.add_argument('--onnx_model', type=str, 
                        help='Custom ONNX model path')
    
    # Benchmark configuration
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height width)')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--detailed', action='store_true',
                        help='Run detailed analysis with more samples')
    
    args = parser.parse_args()
    
    print("🏁 UAV Landing System - PyTorch vs ONNX Benchmark")
    print("=================================================")
    
    if not HAS_ONNX:
        print("❌ ONNX Runtime not available. Install with: pip install onnxruntime")
        return
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark(tuple(args.input_size))
    
    # Determine models to benchmark
    models_to_test = []
    
    if args.all:
        # Test all available stages
        for stage in [2, 3]:
            pytorch_path = f"outputs/stage{stage}_best.pth"
            onnx_path = f"models/stage{stage}_uav_landing.onnx"
            
            if Path(pytorch_path).exists() and Path(onnx_path).exists():
                models_to_test.append((stage, pytorch_path, onnx_path))
            else:
                print(f"⚠️  Stage {stage} models not found - skipping")
                
    elif args.stage:
        pytorch_path = f"outputs/stage{args.stage}_best.pth"
        onnx_path = f"models/stage{args.stage}_uav_landing.onnx"
        models_to_test.append((args.stage, pytorch_path, onnx_path))
        
    elif args.pytorch_model and args.onnx_model:
        models_to_test.append(("custom", args.pytorch_model, args.onnx_model))
    
    else:
        print("❌ Please specify --stage, --all, or --pytorch_model + --onnx_model")
        return
    
    # Adjust sample count for detailed analysis
    num_samples = args.num_samples
    if args.detailed:
        num_samples = max(200, num_samples)
    
    # Run benchmarks
    all_results = {}
    
    for stage, pytorch_path, onnx_path in models_to_test:
        print(f"\n{'='*60}")
        print(f"Benchmarking Stage {stage} Models")
        
        # Check if both models exist
        if not Path(pytorch_path).exists():
            print(f"❌ PyTorch model not found: {pytorch_path}")
            continue
            
        if not Path(onnx_path).exists():
            print(f"❌ ONNX model not found: {onnx_path}")
            print(f"   Convert with: python convert_to_onnx.py --model {pytorch_path} --output {onnx_path}")
            continue
        
        # Create stage-specific output directory
        stage_output_dir = Path(args.output_dir) / f"stage{stage}"
        
        try:
            # Run benchmark
            results = benchmark.run_complete_benchmark(
                pytorch_checkpoint=pytorch_path,
                onnx_model=onnx_path,
                num_samples=num_samples,
                output_dir=str(stage_output_dir)
            )
            
            all_results[f"stage{stage}"] = results
            
            print(f"✅ Stage {stage} benchmark complete")
            
        except Exception as e:
            print(f"❌ Stage {stage} benchmark failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if all_results:
        print(f"\n🎉 Benchmark Summary")
        print(f"==================")
        
        for stage, results in all_results.items():
            print(f"\n{stage.upper()}:")
            print(f"   Speedup: {results['speedup_factor']:.2f}x")
            print(f"   FPS improvement: {results['fps_improvement']:.2f}x")
            print(f"   Accuracy: {results['accuracy_validation']['accuracy']*100:.1f}%")
            
        print(f"\n📊 Detailed results saved in: {args.output_dir}")
    else:
        print(f"❌ No benchmarks completed successfully")


if __name__ == "__main__":
    main() 