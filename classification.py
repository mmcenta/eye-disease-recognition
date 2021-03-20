from functools import reduce

import torch
from torch import nn


class ClassificationModel(nn.Module):
    def __init__(self, n_classes, base_model_fn, freeze=False):
        super(ClassificationModel, self).__init__(**kwargs)

        self.base_model, self.emb_size = base_model_fn()
        self.n_classes = n_classes
        self.freeze = freeze

        if freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False

        in_features = reduce(lambda x, t: x * t, self.emb_size, 1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(in_features, n_classes)

    def forward(self, left_image, right_image):
        pass


class SDAEClassification(ClassificationModel):
    def __init__(self, n_classes, sdae_fn, freeze=False):
        super(SDAEClassification, self).__init__(n_classes, sdae_fn, freeze=freeze)

    def forward(self, left_image, right_image):
        out = self.base_model(left_image, right_image)
        left_embs, _ = out['left']
        right_embs, _ = out['right']
        x = (left_embs[-1] + right_embs[-1]) / 2
        x = self.flatten(x)
        return self.classifier(x)
