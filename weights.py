import torch
import torchvision.models as models

# Download ResNet18 pretrained weights
resnet18 = models.resnet18(pretrained=True)

# Save the pretrained weights to a file
torch.save(resnet18.state_dict(), 'resnet18_weights.pth')
