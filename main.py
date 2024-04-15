import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = FoodDataset(root_dir='EatLens Dataset/train', transform=transform)
valid_dataset = FoodDataset(root_dir='EatLens Dataset/validation', transform=transform)
test_dataset = FoodDataset(root_dir='EatLens Dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pretrained ResNet model
pretrained_resnet = models.resnet18(pretrained=True)

# Modify the final fully connected layer to match the number of output classes in your custom model
num_ftrs = pretrained_resnet.fc.in_features
pretrained_resnet.fc = torch.nn.Linear(num_ftrs, len(train_dataset.classes))

# Initialize your custom model with the modified ResNet architecture
model = pretrained_resnet

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 3  

for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_images = 0
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += labels.size(0)
        accuracy = total_correct / total_images
        print(f'Epoch {epoch+1}, Validation Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Test the trained model
model.eval()
with torch.no_grad():
    total_correct = 0
    total_images = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_images += labels.size(0)
    accuracy = total_correct / total_images
    print(f'Test Accuracy: {accuracy:.4f}')
