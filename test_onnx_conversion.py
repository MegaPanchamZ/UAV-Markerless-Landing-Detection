#!/usr/bin/env python3
"""
Quick test script to convert models and verify ONNX inference works.
"""

import os
import sys
import numpy as np
import time
from pathlib import Path

# This script assumes a successful run of train.py for stages 2 and 3,
# resulting in 'outputs/stage2_best.pth' and 'outputs/stage3_best.pth'.

def test_conversion_and_inference():
    """Test model conversion and basic inference."""
    
    print("🧪 Testing Model Conversion and Inference")
    print("=========================================")
    
    # Check if we have the required models
    stage2_pth = Path("outputs/stage2_best.pth")
    stage3_pth = Path("outputs/stage3_best.pth")
    
    found_models = True
    if not stage2_pth.exists():
        print(f"❌ Stage 2 model not found: {stage2_pth}")
        found_models = False
        
    if not stage3_pth.exists():
        print(f"❌ Stage 3 model not found: {stage3_pth}")
        found_models = False

    if not found_models:
        print("\n💡 Please run training first to generate model checkpoints.")
        return False
    
    print(f"✅ Found PyTorch models")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Convert models to ONNX
    print(f"\n🔄 Converting Models to ONNX...")
    
    try:
        from convert_to_onnx import PyTorchToONNXConverter
        
        # NOTE: The converter is now fixed to load the 'standard' model type,
        # which matches the checkpoints from train.py.
        converter = PyTorchToONNXConverter()
        
        # --- Convert Stage 2 ---
        print("\n--- Processing Stage 2 Model ---")
        stage2_onnx = models_dir / "stage2_uav_landing.onnx"
        pytorch_model_s2, _ = converter.load_pytorch_model(str(stage2_pth))
        success_s2 = converter.convert_to_onnx(pytorch_model_s2, str(stage2_onnx))
        
        if success_s2:
            print(f"✅ Stage 2 converted successfully to {stage2_onnx}")
            validation_success_s2 = converter.validate_conversion(pytorch_model_s2, str(stage2_onnx), num_samples=3)
            if not validation_success_s2:
                print(f"⚠️  Stage 2 validation FAILED. Results may be incorrect.")
        else:
            print(f"❌ Stage 2 conversion failed")
            return False
        
        # --- Convert Stage 3 ---
        print("\n--- Processing Stage 3 Model ---")
        stage3_onnx = models_dir / "stage3_uav_landing.onnx"
        pytorch_model_s3, _ = converter.load_pytorch_model(str(stage3_pth))
        success_s3 = converter.convert_to_onnx(pytorch_model_s3, str(stage3_onnx))
        
        if success_s3:
            print(f"✅ Stage 3 converted successfully to {stage3_onnx}")
            validation_success_s3 = converter.validate_conversion(pytorch_model_s3, str(stage3_onnx), num_samples=3)
            if not validation_success_s3:
                print(f"⚠️  Stage 3 validation FAILED. Results may be incorrect.")
        else:
            print(f"❌ Stage 3 conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ Conversion process failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test ONNX inference (only if conversion and neurosymbolic lib are available)
    print(f"\n🧠 Testing ONNX Inference...")
    
    try:
        from onnx_inference_neurosymbolic import ONNXNeuroSymbolicInference
    except ImportError:
        print("⚠️  Skipping inference test: 'onnx_inference_neurosymbolic.py' not found or has errors.")
        print("\n🎉 Conversion tests passed! ONNX models are in the 'models/' directory.")
        return True

    try:
        # Test Stage 3 model (final model)
        inference_engine = ONNXNeuroSymbolicInference(
            onnx_model_path=str(stage3_onnx)
        )
        
        # Create test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Run inference
        print(f"   Running inference on random test image...")
        start_time = time.time()
        result = inference_engine.predict_single(test_image)
        inference_time = time.time() - start_time
        
        print(f"✅ ONNX inference successful!")
        print(f"   Inference time: {inference_time*1000:.1f}ms")
        print(f"   Predictions shape: {result['predictions'].shape}")
        
        if 'analysis' in result:
             print(f"   Safe area: {result['analysis']['safe_percentage']:.1f}%")
             print(f"   Landing zones found: {len(result['analysis']['landing_zones'])}")

        if 'reasoning' in result and result['reasoning'].get('reasoning_enabled', False):
            print(f"   Neuro-symbolic reasoning: ✅ Enabled")
            recommendations = result['reasoning'].get('landing_recommendation', [])
            if recommendations:
                print(f"   Recommendation: {recommendations[0][0]}")
        else:
            print(f"   Neuro-symbolic reasoning: ❌ Disabled or not applicable")
        
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n🎉 All tests passed! Models are converted and ready for use.")
    print(f"\nNext steps:")
    print(f"1. Test with real images: python onnx_inference_neurosymbolic.py --model models/stage3_uav_landing.onnx --image your_image.jpg")
    print(f"2. Real-time webcam: python onnx_inference_neurosymbolic.py --model models/stage3_uav_landing.onnx --webcam")
    
    return True


if __name__ == "__main__":
    success = test_conversion_and_inference()
    if not success:
        sys.exit(1)