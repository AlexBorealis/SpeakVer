import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class Metrics:
    def find_best_threshold(self, scores, labels):
        """
        Find threshold maximizing F1-score.

        Uses precision_recall_curve instead
        of brute-force threshold search.
        """

        scores = np.asarray(scores)
        labels = np.asarray(labels)

        precision, recall, thresholds = precision_recall_curve(
            labels,
            scores,
        )

        # thresholds has size N-1,
        # precision/recall has size N
        # so remove last precision/recall value

        precision = precision[:-1]
        recall = recall[:-1]

        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        best_idx = np.argmax(f1)

        best_threshold = thresholds[best_idx]

        return float(best_threshold)

    def evaluate(
        self,
        labels,
        scores,
        threshold,
    ):

        labels = np.asarray(labels)
        scores = np.asarray(scores)

        predictions = (scores >= threshold).astype(int)

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        precision = precision_score(
            labels,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            labels,
            scores,
        )

        fpr, tpr, roc_thresholds = roc_curve(
            labels,
            scores,
        )

        fnr = 1 - tpr

        eer_index = np.nanargmin(np.abs(fnr - fpr))

        eer = (fnr[eer_index] + fpr[eer_index]) / 2

        cm = confusion_matrix(
            labels,
            predictions,
        )

        cm_norm = confusion_matrix(
            labels,
            predictions,
            normalize="true",
        )

        p_target = 0.01  # Априорная вероятность целевого спикера
        c_miss = 1.0     # Штраф за пропуск истинного спикера (FRR)
        c_fa = 1.0       # Штраф за ложный допуск чужого спикера (FPR)

        # Считаем стоимость для каждой точки ROC-кривой
        dcf = c_miss * fnr * p_target + c_fa * fpr * (1.0 - p_target)
        # Дефолтная стоимость системы, которая всегда говорит "нет" или всегда "да"
        dcf_baseline = min(c_miss * p_target, c_fa * (1.0 - p_target))
        # Находим минимальную нормализованную стоимость
        min_dcf = float(np.min(dcf / dcf_baseline))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "eer": eer,
            "min_dcf": min_dcf, 
            "threshold": threshold,
            "confusion_matrix": cm,
            "confusion_matrix_normalized": cm_norm,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
        }
