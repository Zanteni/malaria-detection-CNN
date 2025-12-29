import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MalariaCNN
import matplotlib.pyplot as plt
import random

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Dataset
dataset = datasets.ImageFolder(root="/content/data_set/cell_images", transform=transform)
test_size = int(0.15*len(dataset))
_, _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_size-test_size, test_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model
model = MalariaCNN().to(device)
model.load_state_dict(torch.load("/content/drive/MyDrive/MalariaDetection/models/malaria_cnn_state.pth"))
model.eval()

# Random predictions
classes = dataset.classes
plt.figure(figsize=(20,8))
for i in range(10):
    idx = random.randint(0,len(test_dataset)-1)
    img, label = test_dataset[idx]
    img_batch = img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_batch)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        pred_prob = probs[0][pred_class].item()
    plt.subplot(2,5,i+1)
    plt.imshow(img.permute(1,2,0)*0.5 + 0.5)
    plt.title(f"{classes[pred_class]} ({pred_prob*100:.1f}%)\nTrue: {classes[label]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
