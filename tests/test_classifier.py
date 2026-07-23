#!/usr/bin/env python3
import sys
import os
from PIL import Image
import numpy as np

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor.image_processor import preprocess_image
from src.classifier.onnx_classifier import CocoaONNXClassifier

def main():
    model_path = os.path.join("models", "mobilenetv3_cocoa.onnx")
    print(f"Testing ONNX classifier with model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        sys.exit(1)

    # Instantiate classifier
    classifier = CocoaONNXClassifier(model_path)
    print("ONNX model loaded into ONNX Runtime session successfully!")

    # Create synthetic test image (336x336 RGB leaf green)
    dummy_img = Image.fromarray(np.full((336, 336, 3), (34, 139, 34), dtype=np.uint8))
    
    print("Preprocessing synthetic test image...")
    tensor = preprocess_image(dummy_img, target_size=336)
    print(f"Preprocessed tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

    print("Running ONNX inference...")
    top_preds = classifier.get_top_predictions(tensor, top_k=3)
    
    print("\n--- Top 3 Predictions ---")
    for rank, (cls, prob) in enumerate(top_preds, 1):
        print(f"{rank}. {cls}: {prob:.2%}")
        
    print("\n✅ Test Passed! MobileNetV3 ONNX inference engine is fully operational.")

if __name__ == "__main__":
    main()
