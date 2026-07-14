import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    def __init__(self, embedding_dim=192, num_classes=149, margin=0.2, scale=30):
        super().__init__()

        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))

        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

        phi = cosine * self.cos_m - sine * self.sin_m

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(1, labels.view(-1, 1), 1)

        logits = one_hot * phi + (1 - one_hot) * cosine

        logits *= self.scale

        loss = F.cross_entropy(logits, labels)

        return loss, logits
