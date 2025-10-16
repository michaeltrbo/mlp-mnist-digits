import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import openvino as ov
import os
import pandas as pd
import numpy as np
import cv2
import copy

# --- Device Configuration ---
train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {train_device} for training.")

# --- PARQUET LOADING AND DECODING ---
print("Loading data from Parquet file...")
df = pd.read_parquet('./data/MNIST/raw/train-00000-of-00001.parquet')
labels_np = df['label'].to_numpy()
print("Decoding image data...")
images_list = []
for row in df['image']:
    png_bytes = row['bytes']
    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    images_list.append(img)
images_np = np.array(images_list, dtype=np.float32)
images_np = images_np / 255.0
print(f"\nLoaded {len(images_np)} images and {len(labels_np)} labels from Parquet.")

# --- Create PyTorch DataLoaders with Transforms ---
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL Image to Tensor and scales to [0, 1]
    transforms.Normalize((0.1307,), (0.3081,))
])

# Convert training data to tensors
images_tensor = torch.from_numpy(images_np).unsqueeze(1)
labels_tensor = torch.from_numpy(labels_np).long()

# Apply the normalization part of the transform to the training data
images_tensor = transforms.Normalize((0.1307,), (0.3081,))(images_tensor)
print("Applied normalization to training data.")

train_dataset = TensorDataset(images_tensor, labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
print("Training data successfully loaded into DataLoader! ✅")

# Load Test Data and apply the same transform
import torchvision
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
print("Test data loaded successfully! ✅")


# --- Model Definition ---
class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

model = MNIST_MLP()
model.to(train_device)

# --- Training and Validation Loop ---
optimizer = optim.Adam(model.parameters(), lr=0.01)
epochs = 40
criterion = nn.NLLLoss()

# Variables to track the best model based on ACCURACY
best_accuracy = 0.0
best_model_state = None

print("\n--- Starting Training & Validation ---")
for e in range(epochs):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(train_device), labels.to(train_device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # --- Validation Phase (after each epoch) ---
    model.eval()
    correct_count, total_count = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(train_device), labels.to(train_device)
            log_ps = model(images)
            _, predicted_labels = torch.max(log_ps, 1)
            total_count += labels.size(0)
            correct_count += (predicted_labels == labels).sum().item()
    
    epoch_accuracy = 100 * correct_count / total_count
    epoch_loss = running_loss / len(train_loader)
    
    print(f"Epoch {e+1}/{epochs}.. Train Loss: {epoch_loss:.4f}.. Test Accuracy: {epoch_accuracy:.2f}%")

    # Check if the current model is the best one based on accuracy
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"  -> New best model saved with accuracy: {best_accuracy:.2f}%")

print("--- Finished Training! --- ✅")
print(f"Highest test accuracy achieved: {best_accuracy:.2f}%")


# --- Save the 'Last' and 'Best' Models for OpenVINO ---
save_directory = "mnist_openvino_model"
os.makedirs(save_directory, exist_ok=True)
dummy_input = torch.randn(1, 1, 28, 28)

# --- Save the 'Last' Model ---
print("\n--- Saving 'Last' Model for OpenVINO ---")
model.to("cpu")
model.eval()
ov_model_last = ov.convert_model(model, example_input=dummy_input)
model_path_last = os.path.join(save_directory, "last_model.xml")
ov.save_model(ov_model_last, model_path_last)
print(f"OpenVINO 'Last' model saved successfully to {model_path_last}! ✅")

# --- Save the 'Best' Model ---
if best_model_state:
    print("\n--- Saving 'Best' Model for OpenVINO ---")
    model.load_state_dict(best_model_state)
    model.to("cpu")
    model.eval()
    ov_model_best = ov.convert_model(model, example_input=dummy_input)
    model_path_best = os.path.join(save_directory, "best_model.xml")
    ov.save_model(ov_model_best, model_path_best)
    print(f"OpenVINO 'Best' model saved successfully to {model_path_best}! ✅")
else:
    print("\nNo best model state was saved.")

