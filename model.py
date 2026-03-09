import torch
import torch.nn as nn
from torchvision import models


class MultiTaskFaceNet(nn.Module):
    def __init__(self, arch='efficientnet_b0'):
        super().__init__()
        # Загружаем базу EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Голова для предсказания возраста (регрессия)
        self.age_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Голова для предсказания пола (классификация: 0-жен, 1-муж)
        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        feat = self.backbone(x)
        age = self.age_head(feat).squeeze(1)
        gender = self.gender_head(feat)
        return age, gender