from functools import reduce

import torch
from torch import nn
from torchvision.models.resnet import resnet50


class ClassificationModel(nn.Module):
    def __init__(self, n_classes, base_model_fn, freeze=False):
        super(ClassificationModel, self).__init__()

        self.base_model, self.emb_size = base_model_fn()
        self.n_classes = n_classes
        self.freeze = freeze
        if freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False

        in_features = reduce(lambda x, t: x * t, self.emb_size, 1)
        self.left_classifier = nn.Linear(in_features, n_classes, bias=True)
        self.right_classifier = nn.Linear(in_features, n_classes, bias=True)

    def forward(self, left_image, right_image):
        pass


class SDAEClassification(ClassificationModel):
    def __init__(self, n_classes, sdae_fn, freeze=False):
        super(SDAEClassification, self).__init__(n_classes, sdae_fn, freeze=freeze)

    def forward(self, left_image, right_image):
        out = self.base_model(left_image, right_image)
        left_embs, _ = out['left']
        right_embs, _ = out['right']
        left_logits = self.left_classifier(torch.flatten(left_embs[-1], 1))
        right_logits = self.right_classifier(torch.flatten(right_embs[-1], 1))
        return (left_logits + right_logits) / 2


class ResNet50Classification(ClassificationModel):
    def __init__(self, n_classes, resnet50_fn, freeze=True):
        super(ResNet50Classification, self).__init__(n_classes, resnet50_fn, freeze=freeze)

    def _extract_features(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, left_image, right_image):
        left_emb = torch.flatten(self._extract_features(left_image), 1)
        right_emb = torch.flatten(self._extract_features(right_image), 1)
        left_logits = self.left_classifier(left_emb)
        right_logits = self.right_classifier(right_emb)
        return (left_logits + right_logits) / 2
