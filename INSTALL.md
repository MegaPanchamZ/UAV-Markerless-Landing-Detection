# Installation Guide - UAV Landing System

## Quick Start

The project dependencies have been unified into `pyproject.toml`. All previous `requirements.txt` dependencies are now properly organized.

### Basic Installation

```bash
# Install core dependencies
pip install -e .

# Or install specific extras
pip install -e ".[dev]"           # Development tools
pip install -e ".[visualization]" # Advanced plotting
pip install -e ".[deployment]"    # Production deployment
```

### Platform-Specific Installation

#### CUDA GPU Support (Linux/Windows)
```bash
pip install -e ".[cuda]"
```

#### Apple Silicon (M1/M2 Macs)
```bash
pip install -e ".[mps]"
```

#### Intel Optimization
```bash
pip install -e ".[intel]"
```

#### CPU-Only Systems
```bash
pip install -e ".[cpu-only]"
```

### Development Setup
```bash
# Full development environment
pip install -e ".[dev,test,visualization]"
```

### Neuro-Symbolic Computing (Optional)
For Scallop neuro-symbolic reasoning, install separately:
```bash
pip install scallopy
# Or from source:
pip install git+https://github.com/scallop-lang/scallop.git
```

## Testing the Installation

### Test ONNX Inference
```bash
# Quick benchmark
python onnx_inference_neurosymbolic.py --model model_pths/stage3_uav_landing.onnx --benchmark --benchmark_iterations 5

# Process an image
python onnx_inference_neurosymbolic.py --model model_pths/stage3_uav_landing.onnx --image path/to/image.jpg

# Real-time webcam
python onnx_inference_neurosymbolic.py --model model_pths/stage3_uav_landing.onnx --webcam
```

### Expected Performance
- **Inference Time**: ~20-25ms per frame
- **Throughput**: 40-50 FPS
- **Memory Usage**: ~2-4GB GPU memory

## Migration from requirements.txt

The old `requirements.txt` has been **deprecated** and replaced with `pyproject.toml`. All dependencies have been:

1. **Organized** into logical categories
2. **Version-updated** to latest compatible versions  
3. **Separated** into core vs optional dependencies
4. **Platform-optimized** with specific extras

### Key Changes
- `onnxruntime-gpu` is now the default (with CPU fallback)
- Development tools moved to `[dev]` extra
- Testing dependencies organized in `[test]` extra
- Platform-specific optimizations available as extras

## Troubleshooting

### ONNX Runtime Issues
If you see `AttributeError: module 'onnxruntime' has no attribute 'InferenceSession'`:
```bash
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime  # or onnxruntime-gpu
```

### Albumentations Transform Errors
The transforms now require named arguments:
```python
# Correct usage (already fixed in code)
transformed = transform(image=image)
```

### Missing Dependencies
If encountering import errors:
```bash
# Reinstall with all dependencies
pip install -e ".[dev,test,visualization]"
``` 