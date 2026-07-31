from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineStrategy(ABC):
    """
    Strategy вычисления cosine similarity.

    На выходе всегда должен быть tensor формы

        [batch_size, num_classes]
    """

    @abstractmethod
    def __call__(
        self,
        embeddings: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class MarginStrategy(ABC):
    """
    Strategy применения margin.

    На вход получает cosine.

    На выходе возвращает logits
    (до масштабирования scale).
    """

    @abstractmethod
    def __call__(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class NegativeStrategy(ABC):
    """
    Strategy модификации отрицательных классов.

    Нужна для CurricularFace и подобных методов.

    Для обычного ArcFace ничего не делает.
    """

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        cosine: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class IdentityNegative(NegativeStrategy):
    """
    Стандартное поведение ArcFace.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        cosine: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return logits


class Classifier(nn.Module):
    """
    Универсальный margin-based classifier.

    Архитектура состоит из трех независимых стратегий:

        cosine_strategy
        margin_strategy
        negative_strategy

    Благодаря этому можно собрать

        ArcFace
        SubCenter ArcFace
        AdaFace
        CurricularFace
        ElasticFace

    практически без дублирования кода.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        num_classes: int,
        scale: float,
        cosine_strategy: CosineStrategy,
        margin_strategy: MarginStrategy,
        negative_strategy: NegativeStrategy | None = None,
        weight_shape: tuple[int, ...] | None = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale

        self.cosine_strategy = cosine_strategy
        self.margin_strategy = margin_strategy
        self.negative_strategy = (
            negative_strategy if negative_strategy is not None else IdentityNegative()
        )

        if weight_shape is None:
            weight_shape = (
                num_classes,
                embedding_dim,
            )

        self.weight = nn.Parameter(torch.empty(*weight_shape))

        nn.init.xavier_normal_(self.weight)

    def normalize_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return F.normalize(embeddings)

    def normalize_weight(
        self,
    ) -> torch.Tensor:
        return F.normalize(
            self.weight,
            dim=-1,
        )

    def compute_cosine(
        self,
        embeddings: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return self.cosine_strategy(
            embeddings,
            weight,
        )

    def apply_margin(
        self,
        cosine: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.margin_strategy(
            cosine,
            labels,
        )

    def apply_negative_strategy(
        self,
        logits: torch.Tensor,
        cosine: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.negative_strategy(
            logits,
            cosine,
            labels,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ):
        embeddings = self.normalize_embeddings(
            embeddings,
        )

        weight = self.normalize_weight()

        cosine = self.compute_cosine(
            embeddings,
            weight,
        )

        logits = self.apply_margin(
            cosine,
            labels,
        )

        logits = self.apply_negative_strategy(
            logits,
            cosine,
            labels,
        )

        logits = logits * self.scale

        loss = F.cross_entropy(
            logits,
            labels,
        )

        return loss, logits
