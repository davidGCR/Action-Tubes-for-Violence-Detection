import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_ft = models.resnet50(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.model_ft.fc = nn.Linear(num_ftrs, 2)
    
    def forward(self, x):
        x = self.model_ft(x)
        return x