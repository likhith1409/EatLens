import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

class FoodDataset:
    def __init__(self, classes):
        self.classes = classes

    def predict_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        model = models.resnet18(pretrained=True)  # Load pre-trained weights
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(self.classes))
        model.load_state_dict(torch.load('trained_model.pth'))  # Load model state dict
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            predicted_class = self.classes[predicted.item()]
            return predicted_class

if __name__ == "__main__":
    classes = [line.strip() for line in open("EatLens Dataset")]
    food_dataset = FoodDataset(classes)
    predicted_food = food_dataset.predict_image('r.jpeg')
    print('Predicted Food:', predicted_food)
