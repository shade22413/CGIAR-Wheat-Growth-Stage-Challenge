import torch
import torch.nn.functional as F
from timm import create_model, list_models
from torch import nn


class ModelForward(nn.Module):
    def __init__(self, image_size, dropout=0.3, model_name='tf_efficientnet_b3_ns'):
        super().__init__()
        self.base_model = create_model(model_name, pretrained=True)
        self.base_model_output_size = self.base_model.forward_features(torch.rand((1, 3, *image_size))).shape[1]
        self.linear1 = nn.Linear(self.base_model_output_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(dropout)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.gap(x).view(x.shape[0], -1)
        x = self.dropout(F.relu(self.linear1(x)))
        x = self.dropout(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x
