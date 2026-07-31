from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class MarginStrategy(nn.Module, ABC):
    """
    Базовый класс для всех margin-based методов.

    На вход получает:

        cosine      : [B, C]
        labels      : [B]
        embeddings  : [B, D]

    На выходе возвращает logits
    (до применения scale).
    """

    @abstractmethod
    def forward(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class ArcMargin(MarginStrategy):
    """
    Обычный ArcFace / AAMSoftmax.
    """

    def __init__(
        self,
        margin: float = 0.2,
    ):
        super().__init__()

        self.margin = margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        cosine = cosine.clamp(
            -1 + 1e-7,
            1 - 1e-7,
        )

        sine = torch.sqrt(
            torch.clamp(
                1.0 - cosine.pow(2),
                min=1e-9,
            )
        )

        phi = cosine * self.cos_m - sine * self.sin_m

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(
            1,
            labels.view(-1, 1),
            1.0,
        )

        logits = one_hot * phi + (1.0 - one_hot) * cosine

        return logits


class AdaMargin(MarginStrategy):
    """
    Margin для AdaFace.

    Основан на норме эмбеддинга.

    Реализация соответствует общей идее статьи,
    но параметры h, t_alpha и статистики можно
    будет легко изменить.
    """

    def __init__(
        self,
        margin: float = 0.4,
        h: float = 0.333,
        t_alpha: float = 0.01,
    ):
        super().__init__()

        self.margin = margin
        self.h = h
        self.t_alpha = t_alpha

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.register_buffer(
            "batch_mean",
            torch.tensor(20.0),
        )

        self.register_buffer(
            "batch_std",
            torch.tensor(100.0),
        )

    @torch.no_grad()
    def update_statistics(
        self,
        norm: torch.Tensor,
    ):

        mean = norm.mean()

        std = norm.std()

        self.batch_mean.mul_(1 - self.t_alpha).add_(self.t_alpha * mean)

        self.batch_std.mul_(1 - self.t_alpha).add_(self.t_alpha * std)

    def forward(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        norm = embeddings.norm(
            dim=1,
            keepdim=True,
        )

        self.update_statistics(norm.detach())

        margin_scaler = (norm - self.batch_mean) / (self.batch_std + 1e-6)

        margin_scaler = (margin_scaler * self.h).clamp(
            -1,
            1,
        )

        adaptive_margin = self.margin * margin_scaler

        cosine = cosine.clamp(
            -1 + 1e-7,
            1 - 1e-7,
        )

        theta = torch.acos(cosine)

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(
            1,
            labels.view(-1, 1),
            1.0,
        )

        theta = theta + one_hot * adaptive_margin

        logits = torch.cos(theta)

        return logits


class ElasticMargin(MarginStrategy):
    """
    ElasticFace.

    Для каждого батча
    случайно изменяет margin.
    """

    def __init__(
        self,
        margin: float = 0.2,
        std: float = 0.05,
    ):
        super().__init__()

        self.margin = margin
        self.std = std

    def forward(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        margin = (
            torch.randn(
                labels.size(0),
                device=cosine.device,
            )
            * self.std
            + self.margin
        )

        theta = torch.acos(
            cosine.clamp(
                -1 + 1e-7,
                1 - 1e-7,
            )
        )

        one_hot = torch.zeros_like(cosine)

        one_hot.scatter_(
            1,
            labels.view(-1, 1),
            margin.unsqueeze(1),
        )

        theta = theta + one_hot

        return torch.cos(theta)
