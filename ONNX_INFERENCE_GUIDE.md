# ONNX Inference & Benchmarking Guide

This guide explains how to convert PyTorch models to ONNX format and run optimized inference with neuro-symbolic reasoning for UAV landing detection.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core ONNX dependencies
pip install onnx onnxruntime

# For GPU acceleration (optional)
pip install onnxruntime-gpu

# For neuro-symbolic reasoning
pip install scallopy
```

### 2. Quick Test

Run the quick test to convert models and verify everything works:

```bash
cd uav_landing_project
python test_onnx_conversion.py
```

This will:
- ✅ Convert stage2 and stage3 models to ONNX
- ✅ Validate conversion accuracy  
- ✅ Test neuro-symbolic inference
- ✅ Run performance benchmark

## 📋 Available Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `convert_to_onnx.py` | Convert PyTorch to ONNX | Validation, optimization, benchmarking |
| `onnx_inference_neurosymbolic.py` | ONNX inference with reasoning | Real-time inference, Scallop integration |
| `benchmark_pytorch_vs_onnx.py` | Performance comparison | Detailed analysis, visualizations |
| `test_onnx_conversion.py` | Quick test and validation | End-to-end verification |

## 🔄 Model Conversion

### Convert All Models

```bash
python convert_to_onnx.py --all --benchmark
```

### Convert Specific Model

```bash
python convert_to_onnx.py \
    --model outputs/stage3_best.pth \
    --output models/stage3_uav_landing.onnx \
    --benchmark
```

### Conversion Options

- `--all`: Convert all stage models
- `--benchmark`: Run performance benchmark after conversion
- `--no_validate`: Skip accuracy validation
- `--no_optimize`: Skip ONNX optimization
- `--opset_version 11`: Specify ONNX opset version

## 🧠 ONNX Inference with Neuro-Symbolic Reasoning

### Single Image Inference

```bash
python onnx_inference_neurosymbolic.py \
    --model models/stage3_uav_landing.onnx \
    --image test_image.jpg \
    --output result.jpg \
    --save_results
```

### Real-time Webcam

```bash
python onnx_inference_neurosymbolic.py \
    --model models/stage3_uav_landing.onnx \
    --webcam \
    --display
```

### Performance Benchmark

```bash
python onnx_inference_neurosymbolic.py \
    --model models/stage3_uav_landing.onnx \
    --benchmark \
    --benchmark_iterations 200
```

### Inference Options

- `--no_scallop`: Disable neuro-symbolic reasoning
- `--confidence_threshold 0.7`: Set confidence threshold
- `--providers CUDAExecutionProvider`: Force GPU execution
- `--display`: Show real-time visualization

## 📊 Performance Benchmarking

### Compare All Models

```bash
python benchmark_pytorch_vs_onnx.py --all --detailed
```

### Compare Specific Stage

```bash
python benchmark_pytorch_vs_onnx.py --stage 3 --num_samples 200
```

### Benchmark Features

- **Accuracy Validation**: Ensures ONNX produces identical results
- **Performance Metrics**: Latency, throughput, memory usage
- **Statistical Analysis**: Mean, std, percentiles, distributions
- **Visualizations**: Charts, tables, time series plots
- **Detailed Reports**: JSON exports with all metrics

## 🎯 Expected Performance Improvements

Based on typical ONNX optimizations:

| Metric | PyTorch | ONNX | Improvement |
|--------|---------|------|-------------|
| **Inference Time** | ~45ms | ~25ms | **1.8x faster** |
| **Throughput** | ~22 FPS | ~40 FPS | **1.8x higher** |
| **Memory Usage** | ~850MB | ~450MB | **400MB less** |
| **Model Size** | ~13MB | ~13MB | Same |

*Results may vary based on hardware and model complexity*

## 🧠 Neuro-Symbolic Integration

### Scallop Rules Engine

The ONNX inference integrates with Scallop for logical reasoning:

```prolog
// Example safety rules
rel safe_surface = {"road", "vegetation", "roof"}
rel hazardous_surface = {"vehicle", "facade"}

rel suitable_landing_zone(x, y, confidence) = 
    local_area_safe(x, y) and
    sufficient_space(x, y) and
    high_confidence_prediction(x, y) and
    confidence := 0.8
```

### Reasoning Output

The system provides:
- **Safety Assessment**: Safe/Caution/Danger levels
- **Landing Recommendations**: Proceed/Caution/Abort
- **Zone Analysis**: Suitable landing locations
- **Confidence Evaluation**: Prediction reliability

## 📁 Output Structure

```
uav_landing_project/
├── models/                          # ONNX models
│   ├── stage2_uav_landing.onnx
│   ├── stage3_uav_landing.onnx
│   └── *_benchmark.json             # Individual benchmarks
├── benchmark_results/               # Comparison results
│   ├── stage2/
│   │   ├── benchmark_results.json   # Detailed metrics
│   │   ├── performance_comparison.png
│   │   ├── time_series.png
│   │   └── summary_table.png
│   └── stage3/
│       └── ... (same structure)
└── inference_outputs/               # Inference results
    ├── result_with_analysis.jpg     # Visualizations
    └── detailed_results.json        # Raw outputs
```

## 🔧 Configuration Options

### Hardware Optimization

```bash
# CPU-only inference
python onnx_inference_neurosymbolic.py \
    --model models/stage3_uav_landing.onnx \
    --providers CPUExecutionProvider

# GPU-accelerated inference  
python onnx_inference_neurosymbolic.py \
    --model models/stage3_uav_landing.onnx \
    --providers CUDAExecutionProvider CPUExecutionProvider
```

### Input Resolution

```bash
# Standard resolution (512x512)
python convert_to_onnx.py --input_size 512 512

# High resolution (1024x1024) 
python convert_to_onnx.py --input_size 1024 1024

# Custom resolution
python convert_to_onnx.py --input_size 768 768
```

## 🐛 Troubleshooting

### Common Issues

**1. ONNX Runtime not found**
```bash
pip install onnxruntime
# For GPU: pip install onnxruntime-gpu
```

**2. Scallop import error**
```bash
pip install scallopy
```

**3. Model conversion fails**
- Ensure PyTorch model checkpoint exists
- Check model architecture compatibility
- Verify sufficient disk space

**4. GPU inference not working**
- Install CUDA-compatible ONNX Runtime
- Verify CUDA installation
- Check GPU memory availability

### Performance Issues

**1. Slow inference**
- Enable ONNX optimization (`--optimize`)
- Use GPU providers
- Reduce input resolution
- Disable detailed analysis

**2. High memory usage**
- Reduce batch size
- Clear GPU cache between runs
- Use smaller input resolution

## 📈 Monitoring & Logging

### Real-time Performance

The inference engine tracks:
- Frame-by-frame inference times
- Memory usage patterns
- Confidence distributions
- Reasoning execution times

### Metrics Export

All scripts support JSON export:
```json
{
  "inference_time": 0.025,
  "fps": 40.0,
  "memory_usage_mb": 450.2,
  "confidence_map_mean": 0.85,
  "safe_area_percentage": 67.3,
  "landing_zones_found": 3,
  "reasoning_enabled": true,
  "recommendation": "proceed_landing"
}
```

## 🚁 Integration Examples

### Drone Application

```python
from onnx_inference_neurosymbolic import ONNXNeuroSymbolicInference

# Initialize inference engine
engine = ONNXNeuroSymbolicInference(
    onnx_model_path="models/stage3_uav_landing.onnx",
    use_scallop=True
)

# Real-time inference loop
while flying:
    frame = get_camera_frame()
    result = engine.predict_single(frame)
    
    # Make landing decision
    if result['reasoning']['landing_recommendation'][0][0] == "proceed_landing":
        initiate_landing()
    elif result['reasoning']['landing_recommendation'][0][0] == "abort_landing":
        abort_landing()
```

### Edge Deployment

```python
# Optimized for edge devices
engine = ONNXNeuroSymbolicInference(
    onnx_model_path="models/stage3_uav_landing.onnx",
    confidence_threshold=0.6,
    use_scallop=False,  # Disable for speed
    providers=['CPUExecutionProvider']
)
```

## 📚 Additional Resources

- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Scallop Language Guide](https://www.scallop-lang.org/)
- [UAV Landing Dataset Paper](https://link-to-paper.com)
- [Model Architecture Details](docs/ARCHITECTURE.md)

---

For questions or issues, please refer to the [main README](README.md) or create an issue in the repository. 