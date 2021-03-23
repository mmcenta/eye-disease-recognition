from functools import reduce

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class JointResNet50Model(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_size, resnet50_fn):
        super(JointResNet50Model, self).__init__()

        self.base_model, self.emb_size = resnet50_fn()
        self.n_classes = n_classes
        for p in self.base_model.parameters():
            p.requires_grad = False

        in_features = reduce(lambda x, t: x * t, self.emb_size, 1)

        self.to_emb = nn.Embedding(vocab_size, emb_size, padding_idx=0)

        self.left_align = nn.Linear(in_features, emb_size)
        self.right_align = nn.Linear(in_features, emb_size)

        self.left_align_classifier = nn.Linear(emb_size, n_classes)
        self.right_align_classifier = nn.Linear(emb_size, n_classes)
        
        self.left_classifier = nn.Linear(in_features, n_classes)
        self.right_classifier = nn.Linear(in_features, n_classes)

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

    def get_aligned_embedding(self, left_image, right_image):
        left_feats = torch.flatten(self._extract_features(left_image), 1)
        right_feats = torch.flatten(self._extract_features(right_image), 1)
        left_emb = self.left_align(left_feats)
        right_emb = self.right_align(right_feats)
        return left_emb, right_emb

    def forward(self, left_image, right_image):
        left_feats = torch.flatten(self._extract_features(left_image), 1)
        right_feats = torch.flatten(self._extract_features(right_image), 1)
        left_emb = self.left_align(left_feats)
        right_emb = self.right_align(right_feats)
        left_logits = (self.left_classifier(left_feats) + self.left_align_classifier(left_emb)) / 2
        right_logits = (self.right_classifier(right_feats) + self.right_align_classifier(right_emb)) / 2
        return {
            'logits': (left_logits + right_logits) / 2,
            'left_emb': left_emb,
            'right_emb': right_emb,
            'left_sim': F.linear(left_emb, self.to_emb.weight),
            'right_sim': F.linear(right_emb, self.to_emb.weight),
        }
