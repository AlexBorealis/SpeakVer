from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NegativeStrategy(nn.Module, ABC):
    """
    Базовый класс модификации логитов.

    Получает logits после применения margin,
    но до масштабирования scale.

    Может изменять отрицательные логиты,
    положительный логит или оба сразу.

    Parameters
    ----------
    logits : [B, C]
        Логиты после применения margin.

    cosine : [B, C]
        Исходные cosine similarity.

    labels : [B]
        Индексы целевых классов.

    Returns
    -------
    logits : [B, C]
    """

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        cosine: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


class IdentityNegative(NegativeStrategy):
    """
    Стандартное поведение ArcFace.

    Логиты не изменяются.
    """

    def forward(
        self,
        logits: torch.Tensor,
        cosine: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        return logits


class CurricularNegative(NegativeStrategy):
    """
    Упрощённая реализация идеи CurricularFace.

    Внимание:
    ---------
    Это НЕ полная реализация статьи.

    Полная реализация требует доступа
    к target_logit и внутренней EMA-переменной t.

    Данный класс оставлен как заготовка,
    совместимая с архитектурой.
    """

    def __init__(
        self,
        momentum: float = 0.99,
    ):
        super().__init__()

        self.momentum = momentum

        self.register_buffer(
            "t",
            torch.zeros(1),
        )

    @torch.no_grad()
    def update_t(
        self,
        target_logit: torch.Tensor,
    ):
        self.t.mul_(self.momentum).add_((1 - self.momentum) * target_logit.mean())

    def forward(
        self,
        logits: torch.Tensor,
        cosine: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        target_logit = cosine.gather(
            1,
            labels.view(-1, 1),
        )

        self.update_t(target_logit)

        hard_negative = cosine > target_logit

        logits = torch.where(
            hard_negative,
            logits * (self.t + logits),
            logits,
        )

        return logits
