import torch,os
from torchvision import models,transforms
from PIL import Image
from torch.nn import functional as F
class MilletClassifier:
    """Classifier for millet crops."""
    def __init__(self, model_path, class_names, device):
        self.class_names = class_names
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """Loads the pre-trained DenseNet model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        num_classes = len(self.class_names)
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Millet classifier loaded from {model_path} on {self.device}.")
        return model

    def predict(self, image: Image.Image) -> str:
        """Predicts the class of a given millet image."""
        image_tensor = self.transform(image.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_idx = torch.max(probabilities, 1)[1]
        return self.class_names[predicted_idx.item()], probabilities[0][predicted_idx].item()

