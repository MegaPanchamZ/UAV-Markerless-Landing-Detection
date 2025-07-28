#!/usr/bin/env python3
"""
UAV Landing System - PyTorch to ONNX Model Converter
===================================================

Convert trained PyTorch models to ONNX format for cross-platform deployment
and optimized inference with ONNX Runtime.

Features:
- Converts stage2 and stage3 models to ONNX format
- Validates conversion accuracy
- Optimizes ONNX models for inference
- Supports both CPU and GPU ONNX runtime

Usage:
    python convert_to_onnx.py --model outputs/stage2_best.pth --output models/stage2_uav_landing.onnx
    python convert_to_onnx.py --model outputs/stage3_best.pth --output models/stage3_uav_landing.onnx
    python convert_to_onnx.py --all  # Convert all stage models
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, Tuple, Optional
from collections import OrderedDict

# Import model creation function
from models.mobilenetv3_edge_model import create_edge_model

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️  ONNX packages not found. Install with: pip install onnx onnxruntime")


class PyTorchToONNXConverter:
    """Converts PyTorch UAV landing models to ONNX format."""
    
    def __init__(self, input_size: Tuple[int, int] = (512, 512)):
        self.input_size = input_size
        self.batch_size = 1
        
        if not HAS_ONNX:
            raise ImportError("ONNX packages not installed. Run: pip install onnx onnxruntime")
    
    def load_pytorch_model(self, model_path: str) -> nn.Module:
        """Load PyTorch model from checkpoint."""
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        print(f"📥 Loading PyTorch model: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # --- FIX 1: Load the correct 'legacy' model architecture that was used for training ---
        # The 'stage2_best.pth' checkpoint was saved from the LegacyEdgeLandingNet, not the current EdgeLandingNet.
        print("   Instantiating 'legacy' EdgeLandingNet model to match checkpoint.")
        model = create_edge_model(
            model_type='legacy', # Changed from 'standard' to 'legacy'
            num_classes=6,
            use_uncertainty=True,  # Assuming uncertainty was used in training
            pretrained=False
        )
        
        # --- FIX 2: Clean the state_dict keys to handle potential inconsistencies ---
        # This handles cases where models were saved with missing 'backbone.' prefix
        original_state_dict = checkpoint['model_state_dict']
        new_state_dict = OrderedDict()
        
        for k, v in original_state_dict.items():
            name = k
            # Rule 1: Handle DataParallel prefix removal
            if name.startswith('module.'):
                name = name[7:]  # remove `module.` prefix
            # Rule 2: Add 'backbone.' prefix to MobileNetV3's direct 'features' keys
            if name.startswith('features.'):
                # The checkpoint saved 'features.x...' but the class expects 'backbone.features.x...'
                name = 'backbone.' + name
            
            new_state_dict[name] = v

        # Load the cleaned state dictionary.
        # We use strict=True now because the model architecture and keys should match perfectly.
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        
        print(f"✅ PyTorch model loaded successfully using legacy architecture")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   mIoU: {checkpoint.get('metrics', {}).get('miou', 'N/A')}")
        
        return model, checkpoint
    
    def convert_to_onnx(
        self, 
        pytorch_model: nn.Module, 
        output_path: str,
        opset_version: int = 11,
        optimize: bool = True
    ) -> bool:
        """Convert PyTorch model to ONNX format."""
        
        print(f"🔄 Converting to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randn(
            self.batch_size, 3, self.input_size[0], self.input_size[1]
        )
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Determine output names based on the model's configuration
            output_names = ['main_output']
            if hasattr(pytorch_model, 'use_uncertainty') and pytorch_model.use_uncertainty:
                output_names.append('uncertainty_output')

            dynamic_axes = {'input': {0: 'batch_size'}}
            for name in output_names:
                dynamic_axes[name] = {0: 'batch_size'}

            # Export to ONNX
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            print(f"✅ ONNX export successful")
            
            # Verify ONNX model
            if not self._verify_onnx_model(output_path):
                return False
            
            # Optimize ONNX model
            if optimize:
                self._optimize_onnx_model(output_path)
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return False
    
    def _verify_onnx_model(self, onnx_path: str) -> bool:
        """Verify ONNX model is valid."""
        
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            print(f"✅ ONNX model verification passed")
            
            # Print model info
            print(f"   ONNX version: {onnx_model.opset_import[0].version}")
            # Ensure graph.input is not empty before accessing
            if onnx_model.graph.input:
                print(f"   Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
            print(f"   Output count: {len(onnx_model.graph.output)}")
            
            return True
            
        except Exception as e:
            print(f"❌ ONNX verification failed: {e}")
            return False
    
    def _optimize_onnx_model(self, onnx_path: str):
        """Optimize ONNX model for inference."""
        
        try:
            # Create optimized session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            optimized_path = onnx_path.replace('.onnx', '_optimized.onnx')
            sess_options.optimized_model_filepath = optimized_path
            
            # Create session (this will generate optimized model)
            _ = ort.InferenceSession(onnx_path, sess_options)
            
            # Replace original with optimized if optimization was successful
            if Path(optimized_path).exists() and Path(optimized_path).stat().st_size > 0:
                Path(onnx_path).unlink()
                Path(optimized_path).rename(onnx_path)
                print(f"✅ ONNX model optimized")
            else:
                 print(f"⚠️  ONNX optimization did not produce a valid file, keeping original.")

        except Exception as e:
            print(f"⚠️  ONNX optimization failed: {e}")
    
    def validate_conversion(
        self, 
        pytorch_model: nn.Module, 
        onnx_path: str,
        num_samples: int = 5,
        tolerance: float = 1e-4
    ) -> bool:
        """Validate ONNX conversion produces same results as PyTorch."""
        
        print(f"🔍 Validating conversion accuracy...")
        
        try:
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            ort_output_names = [output.name for output in ort_session.get_outputs()]
            
            # Test with multiple random inputs
            all_passed = True
            max_diff = 0.0
            
            for i in range(num_samples):
                # Generate random input
                test_input = torch.randn(1, 3, self.input_size[0], self.input_size[1])
                
                # PyTorch inference
                with torch.no_grad():
                    pytorch_outputs = pytorch_model(test_input)
                
                # ONNX inference
                onnx_inputs = {'input': test_input.numpy()}
                onnx_outputs_list = ort_session.run(ort_output_names, onnx_inputs)
                
                # Map ONNX outputs to a dictionary by name for easier comparison
                onnx_outputs = {name: out for name, out in zip(ort_output_names, onnx_outputs_list)}

                # Compare outputs
                pytorch_main = pytorch_outputs['main'].numpy()
                onnx_main = onnx_outputs['main_output']
                main_diff = np.abs(pytorch_main - onnx_main).max()

                max_diff = max(max_diff, main_diff)
                
                # Compare uncertainty if it exists
                if 'uncertainty_output' in onnx_outputs:
                    pytorch_uncertainty = pytorch_outputs['uncertainty'].numpy()
                    onnx_uncertainty = onnx_outputs['uncertainty_output']
                    uncertainty_diff = np.abs(pytorch_uncertainty - onnx_uncertainty).max()
                    max_diff = max(max_diff, uncertainty_diff)
                    if main_diff > tolerance or uncertainty_diff > tolerance:
                        print(f"   Sample {i+1}: FAILED (main_diff={main_diff:.6f}, uncertainty_diff={uncertainty_diff:.6f})")
                        all_passed = False
                    else:
                        print(f"   Sample {i+1}: PASSED (max_diff={max(main_diff, uncertainty_diff):.6f})")
                else:
                    if main_diff > tolerance:
                        print(f"   Sample {i+1}: FAILED (main_diff={main_diff:.6f})")
                        all_passed = False
                    else:
                        print(f"   Sample {i+1}: PASSED (max_diff={main_diff:.6f})")
            
            if all_passed:
                print(f"✅ Validation passed! Max difference: {max_diff:.6f}")
                return True
            else:
                print(f"❌ Validation failed! Max difference: {max_diff:.6f}")
                return False
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False
    
    # ... (rest of the class is unchanged) ...
    def get_model_info(self, onnx_path: str) -> Dict:
        """Get detailed information about ONNX model."""
        
        try:
            # Load model
            onnx_model = onnx.load(onnx_path)
            
            # Basic info
            info = {
                'file_size_mb': Path(onnx_path).stat().st_size / (1024 * 1024),
                'opset_version': onnx_model.opset_import[0].version,
                'inputs': [],
                'outputs': [],
                'node_count': len(onnx_model.graph.node)
            }
            
            # Input info
            for input_node in onnx_model.graph.input:
                input_info = {
                    'name': input_node.name,
                    'type': input_node.type.tensor_type.elem_type,
                    'shape': [dim.dim_value for dim in input_node.type.tensor_type.shape.dim]
                }
                info['inputs'].append(input_info)
            
            # Output info
            for output_node in onnx_model.graph.output:
                output_info = {
                    'name': output_node.name,
                    'type': output_node.type.tensor_type.elem_type,
                    'shape': [dim.dim_value for dim in output_node.type.tensor_type.shape.dim]
                }
                info['outputs'].append(output_info)
            
            return info
            
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
            return {}
    
    def benchmark_inference(self, onnx_path: str, num_iterations: int = 100) -> Dict:
        """Benchmark ONNX model inference performance."""
        
        print(f"⏱️  Benchmarking ONNX inference ({num_iterations} iterations)...")
        
        try:
            # Create session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Prepare input
            test_input = np.random.randn(1, 3, self.input_size[0], self.input_size[1]).astype(np.float32)
            onnx_inputs = {'input': test_input}
            
            # Warmup
            for _ in range(10):
                _ = ort_session.run(None, onnx_inputs)
            
            # Benchmark
            times = []
            for _ in range(num_iterations):
                start_time = time.time()
                _ = ort_session.run(None, onnx_inputs)
                times.append(time.time() - start_time)
            
            # Calculate statistics
            times_ms = [t * 1000 for t in times]
            stats = {
                'mean_ms': np.mean(times_ms),
                'std_ms': np.std(times_ms),
                'min_ms': np.min(times_ms),
                'max_ms': np.max(times_ms),
                'median_ms': np.median(times_ms),
                'fps': 1000 / np.mean(times_ms)
            }
            
            print(f"✅ Benchmark results:")
            print(f"   Mean: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
            print(f"   Range: {stats['min_ms']:.2f} - {stats['max_ms']:.2f} ms")
            print(f"   FPS: {stats['fps']:.1f}")
            
            return stats
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch UAV models to ONNX')
    
    # Input/Output
    parser.add_argument('--model', type=str, help='PyTorch model checkpoint path')
    parser.add_argument('--output', type=str, help='Output ONNX model path')
    parser.add_argument('--all', action='store_true', help='Convert all stage models')
    
    # Configuration
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height width)')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX opset version')
    parser.add_argument('--no_optimize', action='store_true',
                        help='Skip ONNX optimization')
    parser.add_argument('--no_validate', action='store_true',
                        help='Skip conversion validation')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark converted model')
    
    args = parser.parse_args()
    
    print("🔄 UAV Landing System - PyTorch to ONNX Converter")
    print("=================================================")
    
    if not HAS_ONNX:
        print("❌ ONNX packages not installed. Install with:")
        print("   pip install onnx onnxruntime")
        return
    
    # Create converter
    converter = PyTorchToONNXConverter(tuple(args.input_size))
    
    # Determine models to convert
    models_to_convert = []
    
    if args.all:
        # Convert all stage models
        stage_models = [
            ('outputs/stage2_best.pth', 'models/stage2_uav_landing.onnx'),
            ('outputs/stage3_best.pth', 'models/stage3_uav_landing.onnx')
        ]
        
        for model_path, output_path in stage_models:
            if Path(model_path).exists():
                models_to_convert.append((model_path, output_path))
            else:
                print(f"⚠️  Model not found: {model_path}")
    
    elif args.model and args.output:
        models_to_convert.append((args.model, args.output))
    
    else:
        print("❌ Please specify --model and --output, or use --all")
        return
    
    # Convert models
    for model_path, output_path in models_to_convert:
        print(f"\n{'='*60}")
        print(f"Converting: {model_path} → {output_path}")
        
        try:
            # Load PyTorch model
            pytorch_model, checkpoint = converter.load_pytorch_model(model_path)
            
            # Convert to ONNX
            success = converter.convert_to_onnx(
                pytorch_model, 
                output_path,
                args.opset_version,
                not args.no_optimize
            )
            
            if not success:
                print(f"❌ Conversion failed for {model_path}")
                continue
            
            # Validate conversion
            if not args.no_validate:
                validation_success = converter.validate_conversion(pytorch_model, output_path)
                if not validation_success:
                    print(f"⚠️  Validation failed for {output_path}")
            
            # Get model info
            model_info = converter.get_model_info(output_path)
            if model_info:
                print(f"\n📊 Model Information:")
                print(f"   File size: {model_info['file_size_mb']:.2f} MB")
                print(f"   Inputs: {len(model_info['inputs'])}")
                print(f"   Outputs: {len(model_info['outputs'])}")
                print(f"   Nodes: {model_info['node_count']}")
            
            # Benchmark if requested
            if args.benchmark:
                benchmark_stats = converter.benchmark_inference(output_path)
                
                # Save benchmark results
                if benchmark_stats:
                    benchmark_file = output_path.replace('.onnx', '_benchmark.json')
                    with open(benchmark_file, 'w') as f:
                        json.dump({
                            'model_path': output_path,
                            'input_size': args.input_size,
                            'benchmark_stats': benchmark_stats,
                            'model_info': model_info,
                            'timestamp': time.time()
                        }, f, indent=2)
                    
                    print(f"💾 Benchmark results saved: {benchmark_file}")
            
            print(f"✅ Successfully converted: {output_path}")
            
        except Exception as e:
            print(f"❌ Error converting {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎉 Conversion complete!")


if __name__ == "__main__":
    main()