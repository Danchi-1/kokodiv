#!/usr/bin/env python3
"""
Kokodiv — MobileNetV3 INT8 PyTorch to ONNX Converter
Converts `cocoa_int8_qat_fx.pt` (PyTorch FX QAT model) to ONNX format.

Usage:
    python scripts/export_mobilenet_onnx.py --input /path/to/cocoa_int8_qat_fx.pt --output models/mobilenetv3_cocoa.onnx
"""

import argparse
import os
import sys
import torch
import torchvision.models as models

CLASS_NAMES = [
    "anthracnose",
    "black_pod",
    "cssvd",
    "frosty_pod",
    "healthy",
    "mirid",
    "monilia",
    "pod_borer",
    "witches_broom"
]

def export_onnx(input_path: str, output_path: str):
    print(f"Loading PyTorch checkpoint from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input checkpoint file '{input_path}' not found.")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Attempt 1: Load using torch.jit if it's a TorchScript archive
        try:
            print("Attempting to load as TorchScript archive via torch.jit.load...")
            model = torch.jit.load(input_path, map_location="cpu")
        except Exception as jit_err:
            print(f"JIT load failed ({jit_err}), falling back to standard torch.load(weights_only=False)...")
            model = torch.load(input_path, map_location="cpu", weights_only=False)

        if isinstance(model, dict):
            # Attempt 2: If state_dict was saved, rebuild MobileNetV3 Large with 9 classes
            print("State dict detected. Instantiating MobileNetV3-Large (9 classes)...")
            base_model = models.mobilenet_v3_large(weights=None)
            base_model.classifier[3] = torch.nn.Linear(base_model.classifier[3].in_features, 9)
            
            # Check if FX quantization was used
            try:
                import torch.ao.quantization as quantization
                from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
                
                example_inputs = (torch.randn(1, 3, 336, 336),)
                qconfig_mapping = quantization.get_default_qconfig_mapping('fbgemm')
                prepared_model = prepare_fx(base_model, qconfig_mapping, example_inputs)
                model = convert_fx(prepared_model)
                model.load_state_dict(torch.load(input_path, map_location="cpu", weights_only=False))
            except Exception as e:
                print(f"FX reconstruction fallback: {e}")
                base_model.load_state_dict(model)
                model = base_model

        model.eval()
        print("Model loaded successfully into evaluation mode.")
        
        # Create dummy input: 1 batch, 3 channels, 336x336 image
        dummy_input = torch.randn(1, 3, 336, 336, requires_grad=False)
        
        print(f"Exporting to ONNX at: {output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_tensor'],
            output_names=['confidence_vector'],
            dynamic_axes={
                'input_tensor': {0: 'batch_size'},
                'confidence_vector': {0: 'batch_size'}
            }
        )
        print(f"Success! ONNX model exported to: {output_path}")
        
    except Exception as e:
        print(f"Failed to export ONNX model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch MobileNetV3 model to ONNX.")
    parser.add_argument("--input", "-i", type=str, default="cocoa_int8_qat_fx.pt", help="Path to cocoa_int8_qat_fx.pt")
    parser.add_argument("--output", "-o", type=str, default="models/mobilenetv3_cocoa.onnx", help="Output path for .onnx file")
    args = parser.parse_args()
    
    export_onnx(args.input, args.output)
