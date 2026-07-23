import os
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple

from src.classifier.constants import DISEASE_CLASSES, DISEASE_DISPLAY_NAMES

class CocoaONNXClassifier:
    """
    Lightweight ONNX Runtime inference engine for MobileNetV3 Cocoa Disease Classifier.
    Memory footprint: ~30-50MB RAM.
    Inference latency: ~30-100ms on CPU.
    """
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at path: {model_path}")
        
        # Configure ONNX Runtime for multi-threaded CPU execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, preprocessed_tensor: np.ndarray) -> Dict[str, float]:
        """
        Runs inference on a preprocessed image tensor (1, 3, 336, 336).
        Returns a dictionary mapping disease class names to softmax probability scores.
        """
        if preprocessed_tensor.ndim == 3:
            preprocessed_tensor = np.expand_dims(preprocessed_tensor, axis=0)
            
        raw_outputs = self.session.run([self.output_name], {self.input_name: preprocessed_tensor.astype(np.float32)})[0]
        
        # Apply softmax to convert logits to probabilities
        exp_scores = np.exp(raw_outputs[0] - np.max(raw_outputs[0]))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return {cls: float(prob) for cls, prob in zip(DISEASE_CLASSES, probabilities)}

    def get_top_predictions(self, preprocessed_tensor: np.ndarray, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        Returns the top-K disease predictions sorted by confidence descending.
        """
        scores = self.predict(preprocessed_tensor)
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scores[:top_k]
