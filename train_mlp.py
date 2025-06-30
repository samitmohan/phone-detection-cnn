import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix
from mlp_model import MLPModel 

KEYPOINTS_FILE = 'keypoints_data.npy'
LABELS_FILE = 'labels.npy'

INPUT_DIM = 51 # 17 keypoints * (x, y, confidence)
HIDDEN_DIMS = [128, 64] # hidden layer sizes
OUTPUT_DIM = 2 # Phone / No Phone

BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50 
VALIDATION_SPLIT_RATIO = 0.2

def train_model():
    try:
        keypoints_data = np.load(KEYPOINTS_FILE)
        labels = np.load(LABELS_FILE)
    except FileNotFoundError:
        print(f"Error: Data files '{KEYPOINTS_FILE}' or '{LABELS_FILE}' not found.")
        print("Please run data_collector.py first to generate the dataset.")
        return

    if len(keypoints_data) == 0:
        print("Error: No data samples found. Exiting training.")
        return

    keypoints_tensor = torch.tensor(keypoints_data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(keypoints_tensor, labels_tensor)

    # split
    dataset_size = len(dataset)
    val_size = int(dataset_size * VALIDATION_SPLIT_RATIO)
    train_size = dataset_size - val_size

    if train_size <= 0 or val_size <= 0:
        print("Warning: Dataset too small for proper split. Using entire dataset for training/validation.")
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPModel(input_dim=INPUT_DIM, hidden_dims=HIDDEN_DIMS, output_dim=OUTPUT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting training on {device}...")
    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_losses = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_losses / len(val_dataset)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'mlp_phone_detector_best.pth')
            print("Saved best model!")

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    # evaluation
    print("Evaluation on Validation Set")
    model.load_state_dict(torch.load('mlp_phone_detector_best.pth')) 
    model.eval()
    
    final_correct = 0
    final_total = 0
    final_preds = []
    final_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            final_preds.extend(predicted.cpu().numpy())
            final_targets.extend(targets.cpu().numpy())

    # confusion matrix: binary classification: cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
    cm = confusion_matrix(final_targets, final_preds)
   
    tn, fp, fn, tp = cm.ravel()

    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    print("Actual \\ Predicted | Phone | NoPhone")
    print("--------------------|-------|--------")
    print(f"Phone               | {tp:<5} | {fn:<7}")
    print(f"NoPhone             | {fp:<5} | {tn:<7}")
    print("----------------------------------")

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")

if __name__ == '__main__':
    train_model()