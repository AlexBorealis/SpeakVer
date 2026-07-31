from __future__ import annotations

import torch
import torch.nn.functional as F

from src.model.classifier.base import CosineStrategy


class LinearCosine(CosineStrategy):
    """
    Стандартное вычисление cosine similarity.

    Weight:
        [num_classes, embedding_dim]

    Output:
        [batch_size, num_classes]
    """

    def __call__(
        self,
        embeddings: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:

        return F.linear(
            embeddings,
            weight,
        )


class SubCenterCosine(CosineStrategy):
    """
    Sub-Center ArcFace.

    Каждый класс имеет K центров.

    Weight:
        [num_classes, num_subcenters, embedding_dim]

    Output:
        [batch_size, num_classes]
    """

    def __call__(
        self,
        embeddings: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:

        # cosine:
        #
        # [B,D]
        #
        # x
        #
        # [C,K,D]
        #
        # ->
        #
        # [B,C,K]

        cosine = torch.einsum(
            "bd,ckd->bck",
            embeddings,
            weight,
        )

        # Для каждого класса выбираем
        # наиболее близкий центр.

        cosine = cosine.max(dim=2).values

        return cosine
