import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import MalariaCNN
import torch.nn as nn
import torch.optim as optim
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# Dataset
dataset = datasets.ImageFolder(root="/content/data_set/cell_images", transform=transform)

# Split dataset
total = len(dataset)
train_size = int(0.7*total)
val_size = int(0.15*total)
test_size = total - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model
model = MalariaCNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*X.size(0)
        correct += (outputs.argmax(1) == Y).sum().item()
    
    train_loss = running_loss / len(train_dataset)
    train_acc = correct / len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    with torch.no_grad():
        for X_val, Y_val in val_loader:
            X_val, Y_val = X_val.to(device), Y_val.to(device)
            outputs_val = model(X_val)
            val_loss += criterion(outputs_val, Y_val).item()*X_val.size(0)
            val_correct += (outputs_val.argmax(1) == Y_val).sum().item()
    val_loss /= len(val_dataset)
    val_acc = val_correct / len(val_dataset)
    
    print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

# Save model
os.makedirs("/content/drive/MyDrive/MalariaDetection/models", exist_ok=True)
torch.save(model.state_dict(), "/content/drive/MyDrive/MalariaDetection/models/malaria_cnn_state.pth")
