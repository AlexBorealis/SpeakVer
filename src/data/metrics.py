import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class Metrics:
    def find_best_threshold(self, scores, labels):
        """
        Поиск threshold с максимальным F1
        """
        scores = np.asarray(scores)
        labels = np.asarray(labels)

        thresholds = np.linspace(scores.min(), scores.max(), 500)

        best_threshold = 0
        best_f1 = 0

        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)

            f1 = f1_score(labels, predictions, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold

    def evaluate(self, labels, scores, threshold):
        labels = np.asarray(labels)
        scores = np.asarray(scores)

        predictions = (scores >= threshold).astype(int)

        accuracy = accuracy_score(labels, predictions)

        precision = precision_score(labels, predictions, zero_division=0)

        recall = recall_score(labels, predictions, zero_division=0)

        f1 = f1_score(labels, predictions, zero_division=0)

        roc_auc = roc_auc_score(labels, scores)

        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fnr - fpr))
        eer = (fnr[eer_index] + fpr[eer_index]) / 2
        cm = confusion_matrix(labels, predictions)
        cm_norm = confusion_matrix(labels, predictions, normalize='true')

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "eer": eer,
            "threshold": threshold,
            "confusion_matrix": cm,
            "confusion_matrix_normalized": cm_norm,
            "fpr": fpr,
            "tpr": tpr,
            "roc_thresholds": roc_thresholds,
        }
