from __future__ import annotations

from src.model.classifier.strategies.cosine import (
    LinearCosine,
    SubCenterCosine,
)
from src.model.classifier.strategies.margin import (
    AdaMargin,
    ArcMargin,
    ElasticMargin,
)
from src.model.classifier.strategies.negative import (
    CurricularNegative,
    IdentityNegative,
)

from .base import Classifier


class ClassifierFactory:
    """
    Factory для создания различных margin-based классификаторов.

    Поддерживаемые варианты

        aam
        subcenter
        adaface
        curricular

    Благодаря композиции можно легко добавить
    новые комбинации стратегий.
    """

    @staticmethod
    def create(
        classifier: str,
        *,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
        num_subcenters: int = 3,
    ) -> Classifier:

        classifier = classifier.lower()

        if classifier == "aam":
            return Classifier(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                scale=scale,
                cosine_strategy=LinearCosine(),
                margin_strategy=ArcMargin(
                    margin=margin,
                ),
                negative_strategy=IdentityNegative(),
            )

        elif classifier == "subcenter":
            return Classifier(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                scale=scale,
                cosine_strategy=SubCenterCosine(),
                margin_strategy=ArcMargin(
                    margin=margin,
                ),
                negative_strategy=IdentityNegative(),
                weight_shape=(
                    num_classes,
                    num_subcenters,
                    embedding_dim,
                ),
            )

        elif classifier == "adaface":
            return Classifier(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                scale=scale,
                cosine_strategy=LinearCosine(),
                margin_strategy=AdaMargin(
                    margin=margin,
                ),
                negative_strategy=IdentityNegative(),
            )

        elif classifier == "curricular":
            return Classifier(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                scale=scale,
                cosine_strategy=LinearCosine(),
                margin_strategy=ArcMargin(
                    margin=margin,
                ),
                negative_strategy=CurricularNegative(),
            )

        elif classifier == "elastic":
            return Classifier(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                scale=scale,
                cosine_strategy=LinearCosine(),
                margin_strategy=ElasticMargin(
                    margin=margin,
                ),
                negative_strategy=IdentityNegative(),
            )

        raise ValueError(
            f"Unknown classifier '{classifier}'. "
            f"Available: "
            f"'aam', 'subcenter', "
            f"'adaface', 'curricular', "
            f"'elastic'."
        )
