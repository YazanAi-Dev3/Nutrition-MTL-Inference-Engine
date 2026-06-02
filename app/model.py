import torch
import torch.nn as nn
import torchvision.models as models


# Define the Multi-Task Learning Architecture
class NutritionMTLModel(nn.Module):
    def __init__(self, num_ingredients, pretrained=False):
        super(NutritionMTLModel, self).__init__()

        # Using convnext_tiny as the backbone feature extractor
        backbone = models.convnext_tiny(weights=None)
        feature_dim = backbone.classifier[2].in_features

        self.feature_extractor = backbone.features
        self.avgpool = backbone.avgpool
        self.flatten = nn.Flatten()

        # Task Head A: Multi-label Ingredient Classification
        self.ingredients_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Linear(512, num_ingredients)
        )

        # Task Head B: Mass and Calorie Regression
        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Linear(512, 2)
        )

    def extract_raw_features(self, x):
        """Extracts high-dimensional features for regression tasks."""
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        return self.flatten(x)

    def predict_metrics(self, raw_features):
        """Predicts mass and calories using raw (unnormalized) features."""
        return self.regression_head(raw_features)


def load_inference_model(weights_path, device, num_ingredients=555):
    """Loads the trained weights into the model architecture."""
    model = NutritionMTLModel(num_ingredients=num_ingredients, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)

    # Handle DataParallel prefix if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        clean_state_dict[name] = v

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    return model
