import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import openvino as ov
import os

# Define the transformation pipeline for the images.
#   - ToTensor() converts images to PyTorch Tensors and scales pixels to [0, 1].
#   - Normalize() standardizes the pixel values for better training.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# Create a DataLoader to handle batching and shuffling.
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True   # Shuffles the data every epoch for better training
)

# Load the test dataset
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

images, labels = next(iter(train_loader))

print("Data loaded successfully! ✅")

class MNIST_MLP(nn.Module):
    def __init__(self):
        """
        Initializes the layers of the neural network.
        """
        super().__init__()
        
        # First linear layer (input -> hidden)
        self.fc1 = nn.Linear(in_features = 784, out_features = 128, bias=True)

        # Second linear layer (hidden -> output)
        self.fc2 = nn.Linear(in_features = 128, out_features = 10, bias=True)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        # We need to flatten the in put 'x'
        x = x.view(-1, 784)

        # Pass 'x' through the first fully connected layer
        x = self.fc1(x)

        # Apply the ReLU activation function
        x = F.relu(x)
        
        # Pass the result through the second fully connected layer
        x = self.fc2(x)
        
        # Apply the LogSoftmax activation function to the output
        output = F.log_softmax(x, dim=1)
        
        return output

model = MNIST_MLP()


# --- Training Loop ---

# Define the optimizer, passing in the model's parameters and a learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5 # Set the number of epochs to train for

# Define the loss function
criterion = nn.NLLLoss()

# Set the model to training mode
model.train()

for e in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(images)
        
        # Calculate the loss
        loss = criterion(output, labels)
        
        # Backward pass compute gradient of the loss
        loss.backward()
        
        # Update weights call step() on our optimizer to update the weights
        optimizer.step()
        
        # Track the loss for this epoch
        running_loss += loss.item()
    
    # Print training statistics for the epoch
    print(f"Epoch {e+1}/{epochs}.. Training loss: {running_loss/len(train_loader):.3f}")

print("\nFinished Training! ✅")


# --- Save the Model's State Dictionary ---

print("\nConverting PyTorch model to OpenVINO format...")

model.to("cpu")
model.eval()

# Create a dummy input tensor so OpenVINO understands the model's input shape
dummy_input = torch.randn(1, 1, 28, 28)

# Convert the PyTorch model directly to an OpenVINO model in memory
ov_model = ov.convert_model(model, example_input=dummy_input)

# Save the OpenVINO model to files
save_directory = "mnist_openvino_model"
os.makedirs(save_directory, exist_ok=True) # Create folder if it doesn't exist
model_path = os.path.join(save_directory, "mnist_model.xml")

print(f"Saving OpenVINO model to {model_path}...")
ov.save_model(ov_model, model_path)

print("OpenVINO model saved successfully! ✅")


# --- Evaluation Loop ---

correct_count = 0
total_count = 0

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    # Loop through the test data
    for images, labels in test_loader:
        
        # Forward pass get the model's predictions (log-probabilities)
        log_ps = model(images)
        
        # Find the predicted class by finding the index of the max log-probability
        # torch.max returns (values, indices)
        _, predicted_labels = torch.max(log_ps, 1)
        
        # Count the total number of images in this batch
        total_count += labels.size(0)
        
        # Count the number of correct predictions in this batch
        # (predicted_labels == labels) creates a tensor of True/False
        # .sum() counts the number of True values, and .item() gets it as a Python number
        correct_count += (predicted_labels == labels).sum().item()

# Calculate and print the final accuracy
accuracy = 100 * correct_count / total_count
print(f"\nModel Accuracy on the Test Set: {accuracy:.2f}% 🎯")