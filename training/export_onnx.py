#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

# Define MLPModel class locally to avoid import dependencies
class MLPModel(nn.Module):
    def __init__(self, input_dim=51, hidden_dims=[128, 64], output_dim=2):
        super(MLPModel, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def export_mlp_to_onnx():
    """
    Export the trained MLP model to ONNX format for Triton Inference Server
    """
    
    # Model parameters
    INPUT_DIM = 51
    HIDDEN_DIMS = [128, 64]  
    OUTPUT_DIM = 2
    PYTORCH_MODEL_PATH = 'mlp_phone_detector_best.pth'
    ONNX_MODEL_PATH = 'mlp_phone_detector.onnx'
    
    print("EXPORTING MLP MODEL TO ONNX FORMAT")
    
    # Load the trained PyTorch model
    device = torch.device('cpu') 
    model = MLPModel(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM)
    
    try:
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
        print(f"Loaded PyTorch model from {PYTORCH_MODEL_PATH}")
    except FileNotFoundError:
        print(f"Model file {PYTORCH_MODEL_PATH} not found.")
        print("Please train the model first using: python train_mlp.py")
        return False
    
    model.eval()
    
    # Create example input for ONNX export
    # Shape: [batch_size, input_features] = [1, 51]
    dummy_input = torch.randn(1, INPUT_DIM, dtype=torch.float32)
    
    print(f"   Input shape: [batch_size, {INPUT_DIM}]")
    print(f"   Output shape: [batch_size, {OUTPUT_DIM}]") 
    print(f"   Hidden layers: {HIDDEN_DIMS}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Export to ONNX
    
    try:
        torch.onnx.export(
            model,                       
            dummy_input,                 
            ONNX_MODEL_PATH,              
            export_params=True,           
            opset_version=11,             
            do_constant_folding=True,     
            input_names=['input'],        
            output_names=['output'],     
            dynamic_axes={               
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"Successfully exported to {ONNX_MODEL_PATH}")
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False
    
    # Validate ONNX model
    print("Validating ONNX model...")
    
    try:
        onnx_model = onnx.load(ONNX_MODEL_PATH)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")
        # Print input/output info
        for input_tensor in onnx_model.graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"   Input '{input_tensor.name}': {shape}")
            
        for output_tensor in onnx_model.graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                    for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"   Output '{output_tensor.name}': {shape}")
            
    except Exception as e:
        print(f" ONNX model validation warning: {e}")
    
    return True

def test_onnx_model():
    """
    Test the exported ONNX model with ONNX Runtime
    """
    
    ONNX_MODEL_PATH = 'mlp_phone_detector.onnx'
    PYTORCH_MODEL_PATH = 'mlp_phone_detector_best.pth'
    
    print("TESTING ONNX MODEL")
    
    if not Path(ONNX_MODEL_PATH).exists():
        print(f"ONNX model not found: {ONNX_MODEL_PATH}")
        return False
    
    try:
        # Load ONNX Runtime session
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        print("ONNX Runtime session created")
        
        # Get input/output names
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        print(f"   Input name: {input_name}")
        print(f"   Output name: {output_name}")
        
        # Test with multiple batch sizes
        test_batch_sizes = [1, 2, 4]
        
        for batch_size in test_batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create test input
            test_input = np.random.randn(batch_size, 51).astype(np.float32)
            
            # Run ONNX inference
            ort_inputs = {input_name: test_input}
            ort_outputs = ort_session.run([output_name], ort_inputs)
            onnx_output = ort_outputs[0]
            
            # Apply softmax to get probabilities
            def softmax(x):
                exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
            
            probabilities = softmax(onnx_output)
            
            print(f"   Input shape: {test_input.shape}")
            print(f"   Output shape: {onnx_output.shape}")
            
            # Show sample predictions
            for i in range(min(batch_size, 5)):  # Show max 2 samples
                phone_prob = probabilities[i, 1]
                prediction = "Phone" if phone_prob > 0.5 else "No Phone"
                print(f"   Sample {i}: {prediction} (confidence: {phone_prob:.4f})")
        
        print("ONNX model testing completed successfully!")
        
        # Compare with PyTorch model if available
        if Path(PYTORCH_MODEL_PATH).exists():
            print("Comparing ONNX vs PyTorch outputs...")
            compare_models()
        
        return True
        
    except Exception as e:
        print(f"ONNX model testing failed: {e}")
        return False

def compare_models():
    """
    Compare ONNX and PyTorch model outputs for accuracy verification
    """
    
    PYTORCH_MODEL_PATH = 'mlp_phone_detector_best.pth'
    ONNX_MODEL_PATH = 'mlp_phone_detector.onnx'
    
    try:
        # Load PyTorch model
        pytorch_model = MLPModel(input_dim=51, hidden_dims=[128, 64], output_dim=2)
        pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu'))
        pytorch_model.eval()
        
        # Load ONNX model
        ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        
        # Test with same input
        test_input = np.random.randn(3, 51).astype(np.float32)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_input = torch.from_numpy(test_input)
            pytorch_output = pytorch_model(pytorch_input).numpy()
        
        # ONNX inference  
        onnx_output = ort_session.run([output_name], {input_name: test_input})[0]
        
        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
        
        print(f"   Max difference: {max_diff:.8f}")
        print(f"   Mean difference: {mean_diff:.8f}")
        
        if max_diff < 1e-5:
            print("Models match within acceptable tolerance")
        elif max_diff < 1e-3:
            print(" Models have small differences (acceptable for production)")
        else:
            print("Models have significant differences - investigate!")
        
    except Exception as e:
        print(f"⚠️  Model comparison failed: {e}")

def main():
    """Main execution function"""
    
    # Export MLP model to ONNX
    if export_mlp_to_onnx():
        # Test the exported model
        if test_onnx_model():
            return True
    
    print(f"\nExport failed!")
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)