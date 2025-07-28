#!/usr/bin/env python3
"""
UAV Landing System - ONNX Inference with Neuro-Symbolic Integration
==================================================================

ONNX Runtime inference engine integrated with Scallop-based neuro-symbolic reasoning
for robust UAV landing decisions.

Features:
- ONNX Runtime inference for optimized performance
- Scallop integration for logical safety reasoning
- Uncertainty quantification and confidence analysis
- Real-time performance monitoring
- Comprehensive landing zone analysis

Usage:
    python onnx_inference_neurosymbolic.py --model models/stage3_uav_landing.onnx --image test.jpg
    python onnx_inference_neurosymbolic.py --model models/stage2_uav_landing.onnx --webcam
    python onnx_inference_neurosymbolic.py --benchmark --model models/stage3_uav_landing.onnx
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime

# ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("⚠️  ONNX Runtime not found. Install with: pip install onnxruntime or pip install onnxruntime-gpu")

# Scallop for neuro-symbolic reasoning
try:
    import scallopy
    HAS_SCALLOP = True
except ImportError:
    HAS_SCALLOP = False
    print("⚠️  Scallopy not found. Install with: pip install scallopy")

# Import transforms and utilities
from datasets.semantic_drone_dataset import create_semantic_drone_transforms


class ONNXNeuroSymbolicInference:
    """ONNX Runtime inference with integrated neuro-symbolic reasoning."""
    
    CLASS_NAMES = ['other', 'facade', 'road', 'vegetation', 'vehicle', 'roof']
    CLASS_COLORS = [
        [128, 128, 128],  # other - gray
        [128, 0, 0],      # facade - dark red
        [128, 64, 128],   # road - purple
        [0, 128, 0],      # vegetation - green
        [64, 0, 128],     # vehicle - blue
        [64, 64, 0]       # roof - olive
    ]
    
    def __init__(
        self,
        onnx_model_path: str,
        input_size: Tuple[int, int] = (512, 512),
        confidence_threshold: float = 0.5,
        providers: Optional[List[str]] = None,
        use_scallop: bool = True
    ):
        """Initialize ONNX inference engine with neuro-symbolic reasoning."""
        
        if not HAS_ONNX:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime or pip install onnxruntime-gpu")
        
        self.model_path = onnx_model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.use_scallop = use_scallop and HAS_SCALLOP
        
        # Performance tracking
        self.inference_times = []
        self.processed_frames = 0
        
        print(f"🚁 ONNX Neuro-Symbolic UAV Landing System")
        print(f"   Model: {onnx_model_path}")
        print(f"   Input size: {input_size}")
        print(f"   Neuro-symbolic: {'✅ Enabled' if self.use_scallop else '❌ Disabled'}")
        
        # Load ONNX model
        self.session = self._create_onnx_session(providers)
        
        # Initialize transforms
        self.transform = create_semantic_drone_transforms(
            input_size=input_size,
            is_training=False
        )
        
        # Initialize Scallop context
        self.scallop_ctx = None
        if self.use_scallop:
            self.scallop_ctx = self._initialize_scallop()
        
        print(f"✅ Inference engine initialized")
    
    def _create_onnx_session(self, providers: Optional[List[str]] = None) -> ort.InferenceSession:
        """Create optimized ONNX Runtime session."""
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # Use all available cores
        
        # Determine providers
        if providers is None:
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        print(f"   ONNX providers: {providers}")
        
        # Create session
        session = ort.InferenceSession(
            self.model_path,
            sess_options,
            providers=providers
        )
        
        # Print model info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        
        print(f"   Input: {input_info.name} {input_info.shape} {input_info.type}")
        print(f"   Outputs: {len(output_info)} tensors")
        for i, output in enumerate(output_info):
            print(f"     {i}: {output.name} {output.shape} {output.type}")
        
        return session
    
    def _initialize_scallop(self):
        """Initialize Scallop reasoning context with UAV landing rules."""
        
        if not HAS_SCALLOP:
            return None
        
        print(f"🧠 Initializing Scallop neuro-symbolic reasoning...")
        
        # Scallop program for UAV landing safety
        scallop_program = """
          // Define safe and hazardous surfaces based on our 6-class model
          rel safe_surface = {"road", "vegetation", "roof"}
          rel hazardous_surface = {"vehicle", "facade"}
          rel uncertain_surface = {"other"}

          // Dominant surface analysis
          rel dominant_surface(s) = surface_area(s, a) and a == max(a: surface_area(_, a))

          // Assign safety scores based on dominant surface
          rel area_safety_score(0.9) = dominant_surface(s) and safe_surface(s)
          rel area_safety_score(0.1) = dominant_surface(s) and hazardous_surface(s)
          rel area_safety_score(0.5) = dominant_surface(s) and uncertain_surface(s)

          // Landing zone evaluation
          rel local_area_safe(x, y) = nearby_surface(x, y, s) and safe_surface(s)
          rel sufficient_space(x, y) = landing_space_available(x, y, space) and space > 20
          rel high_confidence_prediction(x, y) = prediction_confidence(x, y, c) and c > 0.7

          rel suitable_landing_zone(x, y) =
            local_area_safe(x, y) and
            sufficient_space(x, y) and
            high_confidence_prediction(x, y)

          // Overall mission safety assessment
          rel mission_safety_level("safe") = area_safety_score(s) and s > 0.7
          rel mission_safety_level("caution") = area_safety_score(s) and s > 0.4 and s <= 0.7
          rel mission_safety_level("danger") = area_safety_score(s) and s <= 0.4

          // Uncertainty handling
          rel prediction_reliable() = avg_uncertainty(u) and u < 0.3

          // Landing recommendations
          rel landing_recommendation("proceed_landing") = mission_safety_level("safe") and prediction_reliable()
          rel landing_recommendation("caution_landing") = mission_safety_level("caution") and prediction_reliable()
          rel landing_recommendation("abort_landing") = mission_safety_level("danger") or not prediction_reliable()
        """
        
        try:
            # Create Scallop context
            ctx = scallopy.ScallopContext()
            ctx.add_program(scallop_program)
            
            print(f"✅ Scallop context initialized")
            return ctx
            
        except Exception as e:
            print(f"❌ Scallop initialization failed: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX inference."""
        
        # Ensure RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR and convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (albumentations expects named arguments)
        transformed = self.transform(image=image)
        tensor = transformed['image']
        # Convert to numpy
        if hasattr(tensor, 'numpy'):
            input_array = tensor.numpy()
        else:
            input_array = np.array(tensor)
        
        # Add batch dimension if needed
        if len(input_array.shape) == 3:
            input_array = np.expand_dims(input_array, axis=0)
        
        return input_array.astype(np.float32)
    
    def run_onnx_inference(self, input_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run ONNX inference on preprocessed input."""
        
        start_time = time.time()
        
        # Prepare input
        input_name = self.session.get_inputs()[0].name
        onnx_inputs = {input_name: input_array}
        
        # Run inference
        onnx_outputs = self.session.run(None, onnx_inputs)
        
        # Extract outputs
        main_output = onnx_outputs[0]  # Segmentation predictions
        uncertainty_output = onnx_outputs[1] if len(onnx_outputs) > 1 else None
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.processed_frames += 1
        
        return main_output, uncertainty_output, inference_time
    
    def postprocess_outputs(
        self, 
        main_output: np.ndarray, 
        uncertainty_output: Optional[np.ndarray]
    ) -> Dict:
        """Post-process ONNX outputs into usable format."""
        
        # Apply softmax to get probabilities
        exp_outputs = np.exp(main_output - np.max(main_output, axis=1, keepdims=True))
        probabilities = exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)
        
        # Get class predictions
        predictions = np.argmax(probabilities, axis=1)
        
        # Process uncertainty if available
        confidence_map = None
        if uncertainty_output is not None:
            # Sigmoid to get uncertainty, then convert to confidence
            uncertainty_sigmoid = 1.0 / (1.0 + np.exp(-uncertainty_output))
            confidence_map = 1.0 - uncertainty_sigmoid
        
        # Remove batch dimension
        predictions = predictions[0]
        probabilities = probabilities[0]
        if confidence_map is not None:
            confidence_map = confidence_map[0, 0]  # Remove batch and channel dims
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_map': confidence_map
        }
    
    def analyze_landing_zones(self, predictions: np.ndarray, confidence_map: Optional[np.ndarray] = None) -> Dict:
        """Analyze predictions to find suitable landing zones."""
        
        # Calculate class statistics
        class_stats = {}
        total_pixels = predictions.size
        
        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            mask = predictions == class_idx
            pixel_count = np.sum(mask)
            percentage = (pixel_count / total_pixels) * 100
            
            class_stats[class_name] = {
                'pixel_count': int(pixel_count),
                'percentage': float(percentage),
                'mask': mask
            }
        
        # Find safe landing areas
        safe_classes = ['road', 'vegetation', 'roof']
        safe_mask = np.zeros_like(predictions, dtype=bool)
        
        for class_name in safe_classes:
            if class_name in class_stats:
                safe_mask |= class_stats[class_name]['mask']
        
        # Apply confidence filtering if available
        if confidence_map is not None:
            high_confidence_mask = confidence_map > self.confidence_threshold
            safe_mask &= high_confidence_mask
        
        # Find connected components for landing zones
        safe_mask_uint8 = safe_mask.astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(safe_mask_uint8)
        
        landing_zones = []
        for label in range(1, num_labels):  # Skip background (label 0)
            zone_mask = labels == label
            zone_area = np.sum(zone_mask)
            
            if zone_area > 100:  # Minimum zone size
                # Find centroid
                y_coords, x_coords = np.where(zone_mask)
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                
                landing_zones.append({
                    'centroid': (centroid_x, centroid_y),
                    'area': int(zone_area),
                    'mask': zone_mask
                })
        
        # Sort by area (largest first)
        landing_zones.sort(key=lambda x: x['area'], reverse=True)
        
        return {
            'class_stats': class_stats,
            'safe_mask': safe_mask,
            'landing_zones': landing_zones,
            'safe_percentage': float(np.sum(safe_mask) / total_pixels * 100)
        }
    
    def run_scallop_reasoning(self, analysis: Dict) -> Dict:
        """Run Scallop-based neuro-symbolic reasoning."""
        
        if not self.use_scallop or self.scallop_ctx is None:
            return {'reasoning_enabled': False}
        
        try:
            # Clear previous facts
            self.scallop_ctx.clear_facts()
            
            # Add surface area facts
            for class_name, stats in analysis['class_stats'].items():
                if stats['percentage'] > 1.0:  # Only significant areas
                    self.scallop_ctx.add_facts("surface_area", [(class_name, stats['percentage'])])
            
            # Add landing zone facts
            for i, zone in enumerate(analysis['landing_zones'][:5]):  # Top 5 zones
                x, y = zone['centroid']
                area = zone['area']
                self.scallop_ctx.add_facts("landing_space_available", [(x, y, area)])
                
                # Add nearby surface facts (simplified)
                dominant_class = None
                max_pixels = 0
                for class_name, stats in analysis['class_stats'].items():
                    if stats['percentage'] > max_pixels:
                        max_pixels = stats['percentage']
                        dominant_class = class_name
                
                if dominant_class:
                    self.scallop_ctx.add_facts("nearby_surface", [(x, y, dominant_class)])
            
            # Add confidence facts (if available)
            if 'confidence_map' in analysis and analysis['confidence_map'] is not None:
                avg_confidence = float(np.mean(analysis['confidence_map']))
                self.scallop_ctx.add_facts("avg_uncertainty", [(1.0 - avg_confidence,)])
                
                # Add per-zone confidence
                for zone in analysis['landing_zones'][:5]:
                    x, y = zone['centroid']
                    zone_confidence = float(np.mean(analysis['confidence_map'][zone['mask']]))
                    self.scallop_ctx.add_facts("prediction_confidence", [(x, y, zone_confidence)])
            
            # Add default environmental facts
            self.scallop_ctx.add_facts("current_weather", [("clear",)])
            self.scallop_ctx.add_facts("uav_capability", [("small_drone",)])
            
            # Run reasoning
            self.scallop_ctx.run()
            
            # Extract results
            reasoning_results = {
                'reasoning_enabled': True,
                'area_safety_score': list(self.scallop_ctx.relation("area_safety_score")),
                'suitable_landing_zones': list(self.scallop_ctx.relation("suitable_landing_zone")),
                'mission_safety_level': list(self.scallop_ctx.relation("mission_safety_level")),
                'landing_recommendation': list(self.scallop_ctx.relation("landing_recommendation")),
                'prediction_reliable': list(self.scallop_ctx.relation("prediction_reliable"))
            }
            
            return reasoning_results
            
        except Exception as e:
            print(f"⚠️  Scallop reasoning failed: {e}")
            return {'reasoning_enabled': False, 'error': str(e)}
    
    def predict_single(self, image: np.ndarray) -> Dict:
        """Run complete prediction pipeline on single image."""
        
        # Preprocess
        input_array = self.preprocess_image(image)
        
        # ONNX inference
        main_output, uncertainty_output, inference_time = self.run_onnx_inference(input_array)
        
        # Post-process
        processed = self.postprocess_outputs(main_output, uncertainty_output)
        
        # Analyze landing zones
        analysis = self.analyze_landing_zones(
            processed['predictions'], 
            processed['confidence_map']
        )
        
        # Neuro-symbolic reasoning
        reasoning = self.run_scallop_reasoning(analysis)
        
        # Combine results
        result = {
            'predictions': processed['predictions'],
            'probabilities': processed['probabilities'],
            'confidence_map': processed['confidence_map'],
            'analysis': analysis,
            'reasoning': reasoning,
            'inference_time': inference_time,
            'timestamp': time.time()
        }
        
        return result
    
    def visualize_results(self, image: np.ndarray, result: Dict, save_path: Optional[str] = None) -> np.ndarray:
        """Create comprehensive visualization of results."""
        
        predictions = result['predictions']
        confidence_map = result['confidence_map']
        analysis = result['analysis']
        reasoning = result['reasoning']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Segmentation prediction
        colored_pred = np.zeros((*predictions.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.CLASS_COLORS):
            colored_pred[predictions == class_idx] = color
        
        axes[0, 1].imshow(colored_pred)
        axes[0, 1].set_title("Segmentation Prediction")
        axes[0, 1].axis('off')
        
        # Confidence map
        if confidence_map is not None:
            im = axes[0, 2].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
            axes[0, 2].set_title("Confidence Map")
            axes[0, 2].axis('off')
            plt.colorbar(im, ax=axes[0, 2])
        else:
            axes[0, 2].text(0.5, 0.5, "No Confidence\nMap Available", 
                           ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].axis('off')
        
        # Safe landing areas
        safe_visualization = image.copy()
        if analysis['safe_mask'] is not None:
            safe_visualization[analysis['safe_mask']] = [0, 255, 0]  # Green overlay
        
        # Mark landing zones
        for i, zone in enumerate(analysis['landing_zones'][:3]):  # Top 3 zones
            x, y = zone['centroid']
            cv2.circle(safe_visualization, (x, y), 20, (255, 0, 0), 3)
            cv2.putText(safe_visualization, f"{i+1}", (x-5, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        axes[1, 0].imshow(safe_visualization)
        axes[1, 0].set_title("Landing Zones")
        axes[1, 0].axis('off')
        
        # Class distribution
        class_names = list(analysis['class_stats'].keys())
        percentages = [analysis['class_stats'][name]['percentage'] for name in class_names]
        
        axes[1, 1].bar(class_names, percentages)
        axes[1, 1].set_title("Class Distribution (%)")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Reasoning results
        reasoning_text = "Neuro-Symbolic Analysis:\n\n"
        
        if reasoning.get('reasoning_enabled', False):
            # Safety level
            safety_levels = reasoning.get('mission_safety_level', [])
            if safety_levels:
                safety_level = safety_levels[0][0] if safety_levels[0] else "unknown"
                reasoning_text += f"Safety Level: {safety_level}\n"
            
            # Recommendation
            recommendations = reasoning.get('landing_recommendation', [])
            if recommendations:
                recommendation = recommendations[0][0] if recommendations[0] else "unknown"
                reasoning_text += f"Recommendation: {recommendation}\n"
            
            # Landing zones
            suitable_zones = reasoning.get('suitable_landing_zones', [])
            reasoning_text += f"Suitable Zones: {len(suitable_zones)}\n"
            
            # Reliability
            reliable = reasoning.get('prediction_reliable', [])
            is_reliable = len(reliable) > 0
            reasoning_text += f"Prediction Reliable: {is_reliable}\n"
        else:
            reasoning_text += "Reasoning disabled or failed"
        
        # Add performance info
        reasoning_text += f"\nPerformance:\n"
        reasoning_text += f"Inference: {result['inference_time']*1000:.1f}ms\n"
        reasoning_text += f"Safe Area: {analysis['safe_percentage']:.1f}%"
        
        axes[1, 2].text(0.05, 0.95, reasoning_text, transform=axes[1, 2].transAxes,
                       verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved: {save_path}")
        
        # Convert to image array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img_array
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict:
        """Benchmark inference performance."""
        
        print(f"⏱️  Benchmarking ONNX inference ({num_iterations} iterations)...")
        
        # Create test input
        test_image = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(10):
            self.predict_single(test_image)
        
        # Reset timing
        self.inference_times = []
        
        # Benchmark
        start_time = time.time()
        for i in range(num_iterations):
            result = self.predict_single(test_image)
            if (i + 1) % 25 == 0:
                print(f"   Progress: {i+1}/{num_iterations}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        times_ms = [t * 1000 for t in self.inference_times]
        
        stats = {
            'total_time_s': total_time,
            'mean_ms': np.mean(times_ms),
            'std_ms': np.std(times_ms),
            'min_ms': np.min(times_ms),
            'max_ms': np.max(times_ms),
            'median_ms': np.median(times_ms),
            'fps': num_iterations / total_time,
            'throughput_fps': 1000 / np.mean(times_ms)
        }
        
        print(f"✅ Benchmark Results:")
        print(f"   Total time: {stats['total_time_s']:.2f}s")
        print(f"   Mean inference: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        print(f"   Range: {stats['min_ms']:.2f} - {stats['max_ms']:.2f} ms")
        print(f"   Throughput: {stats['throughput_fps']:.1f} FPS")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='ONNX Neuro-Symbolic UAV Landing Inference')
    
    # Model configuration
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--input_size', type=int, nargs=2, default=[512, 512],
                        help='Input image size (height width)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                        help='Confidence threshold for predictions')
    parser.add_argument('--no_scallop', action='store_true',
                        help='Disable Scallop neuro-symbolic reasoning')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--image', type=str, help='Input image file')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam input')
    input_group.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output file/directory')
    parser.add_argument('--save_results', action='store_true', help='Save detailed results as JSON')
    parser.add_argument('--display', action='store_true', help='Display results')
    
    # Performance options
    parser.add_argument('--providers', type=str, nargs='+', 
                        choices=['CPUExecutionProvider', 'CUDAExecutionProvider'],
                        help='ONNX Runtime providers')
    parser.add_argument('--benchmark_iterations', type=int, default=100,
                        help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    print("🚁 UAV Landing System - ONNX Neuro-Symbolic Inference")
    print("=====================================================")
    
    # Check ONNX model exists
    if not Path(args.model).exists():
        print(f"❌ ONNX model not found: {args.model}")
        return
    
    try:
        # Initialize inference engine
        inference_engine = ONNXNeuroSymbolicInference(
            onnx_model_path=args.model,
            input_size=tuple(args.input_size),
            confidence_threshold=args.confidence_threshold,
            providers=args.providers,
            use_scallop=not args.no_scallop
        )
        
        if args.benchmark:
            # Performance benchmark
            stats = inference_engine.benchmark_performance(args.benchmark_iterations)
            
            # Save benchmark results
            if args.output:
                benchmark_file = args.output
            else:
                model_name = Path(args.model).stem
                benchmark_file = f"{model_name}_onnx_benchmark.json"
            
            with open(benchmark_file, 'w') as f:
                json.dump({
                    'model_path': args.model,
                    'benchmark_stats': stats,
                    'configuration': {
                        'input_size': args.input_size,
                        'confidence_threshold': args.confidence_threshold,
                        'scallop_enabled': not args.no_scallop,
                        'providers': args.providers
                    },
                    'timestamp': time.time()
                }, f, indent=2)
            
            print(f"💾 Benchmark results saved: {benchmark_file}")
            
        elif args.image:
            # Single image inference
            if not Path(args.image).exists():
                print(f"❌ Image not found: {args.image}")
                return
            
            # Load and process image
            image = cv2.imread(args.image)
            if image is None:
                print(f"❌ Failed to load image: {args.image}")
                return
            
            print(f"🖼️  Processing image: {args.image}")
            
            # Run inference
            result = inference_engine.predict_single(image)
            
            # Create visualization
            output_path = args.output or f"{Path(args.image).stem}_onnx_result.jpg"
            viz_image = inference_engine.visualize_results(image, result, output_path)
            
            # Save detailed results
            if args.save_results:
                results_file = output_path.replace('.jpg', '_results.json')
                
                # Convert numpy arrays to lists for JSON serialization
                json_result = {
                    'predictions': result['predictions'].tolist(),
                    'analysis': {
                        'class_stats': {k: {**v, 'mask': None} for k, v in result['analysis']['class_stats'].items()},
                        'safe_percentage': result['analysis']['safe_percentage'],
                        'landing_zones': [{**zone, 'mask': None} for zone in result['analysis']['landing_zones']]
                    },
                    'reasoning': result['reasoning'],
                    'inference_time': result['inference_time'],
                    'timestamp': result['timestamp']
                }
                
                with open(results_file, 'w') as f:
                    json.dump(json_result, f, indent=2)
                
                print(f"💾 Detailed results saved: {results_file}")
            
            # Display if requested
            if args.display:
                cv2.imshow("ONNX Neuro-Symbolic Results", viz_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            print(f"✅ Image processing complete")
            
        elif args.webcam:
            # Real-time webcam processing
            print(f"📹 Starting webcam inference...")
            print(f"   Press 'q' to quit, 's' to save frame")
            
            cap = cv2.VideoCapture(0)
            frame_count = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Run inference
                    result = inference_engine.predict_single(frame)
                    
                    # Create visualization
                    viz_image = inference_engine.visualize_results(frame, result)
                    
                    # Resize for display
                    display_height = 600
                    aspect_ratio = viz_image.shape[1] / viz_image.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    viz_resized = cv2.resize(viz_image, (display_width, display_height))
                    
                    # Add performance info
                    fps = 1.0 / result['inference_time'] if result['inference_time'] > 0 else 0
                    cv2.putText(viz_resized, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('ONNX Neuro-Symbolic UAV Landing', viz_resized)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        save_path = f"webcam_frame_{frame_count:04d}.jpg"
                        inference_engine.visualize_results(frame, result, save_path)
                        frame_count += 1
                        
            finally:
                cap.release()
                cv2.destroyAllWindows()
            
            print(f"📹 Webcam session ended")
            
        else:
            print("❌ Please specify --image, --webcam, or --benchmark")
            return
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 