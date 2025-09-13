import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from export_onnx import MLPModel
import argparse

def evaluate_model(model_path='models/mlp_phone_detector_best.pth', 
                  keypoints_file='keypoints_data.npy', 
                  labels_file='labels.npy'):
    """
    Evaluate the trained MLP model on the dataset.
    """
    
    # Load data
    try:
        keypoints_data = np.load(keypoints_file)
        labels = np.load(labels_file)
        print(f"Loaded {len(keypoints_data)} samples from {keypoints_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have the keypoints_data.npy and labels.npy files.")
        return None
    
    # Convert to tensors
    keypoints_tensor = torch.tensor(keypoints_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset and dataloader
    dataset = TensorDataset(keypoints_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPModel(input_dim=51, hidden_dims=[128, 64], output_dim=2).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        print("Train the model first using: python train_mlp.py")
        return None
    
    model.eval()
    
    # Evaluate
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Phone class probabilities
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    print(f"Dataset Size: {len(all_targets)}")
    print(f"Phone samples: {sum(all_targets)}")
    print(f"No-phone samples: {len(all_targets) - sum(all_targets)}")
    
    print("ACCURACY METRICS:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    
    print("CONFUSION MATRIX:")
    print("Actual \\ Predicted | No-Phone | Phone")
    print("--------------------|----------|-------")
    print(f"No-Phone            | {tn:<8} | {fp:<5}")
    print(f"Phone               | {fn:<8} | {tp:<5}")
    
    print(f"True Positives (TP):  {tp} - Correctly identified phone usage")
    print(f"True Negatives (TN):  {tn} - Correctly identified no phone")
    print(f"False Positives (FP): {fp} - Incorrectly identified phone usage")
    print(f"False Negatives (FN): {fn} - Missed phone usage")
    
    # Additional metrics for comparison
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
    print(f"Sensitivity (Recall/True Positive Rate): {recall:.4f}")
    
    print("\n" + "="*50)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained MLP phone detection model")
    parser.add_argument("--model_path", type=str, default="models/mlp_phone_detector_best.pth", 
                       help="Path to trained model weights")
    parser.add_argument("--keypoints", type=str, default="training/keypoints_data.npy", 
                       help="Path to keypoints data file")
    parser.add_argument("--labels", type=str, default="training/labels.npy", 
                       help="Path to labels file")
    
    args = parser.parse_args()
    
    results = evaluate_model(args.model_path, args.keypoints, args.labels)
    
    print(f"MLP Model Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

if __name__ == "__main__":
    main()
