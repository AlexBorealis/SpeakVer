import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def plot_similarity(
        self, labels, scores, save_path="similarity_distribution.png", threshold=None
    ):
        scores = np.asarray(scores)
        labels = np.asarray(labels)

        positive = scores[labels == 1]
        negative = scores[labels == 0]

        plt.figure(figsize=(10, 6))

        plt.hist(
            positive,
            bins=40,
            alpha=0.6,
            density=True,
            label=f"Same speaker ({len(positive)})",
        )

        plt.hist(
            negative,
            bins=40,
            alpha=0.6,
            density=True,
            label=f"Different speaker ({len(negative)})",
        )

        if threshold is not None:
            plt.axvline(
                threshold,
                linestyle="--",
                linewidth=2,
                label=f"Threshold = {threshold:.3f}",
            )

        plt.xlabel("Cosine similarity")
        plt.ylabel("Density")
        plt.title("Speaker verification cosine similarity distribution")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_roc(self, metrics, save_path="roc_curve.png"):
        plt.figure(figsize=(6, 6))

        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            linewidth=2,
            label=f"AUC = {metrics['roc_auc']:.4f}",
        )

        plt.plot([0, 1], [0, 1], "--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_confusion_matrix(self, metrics, save_path="confusion_matrix.png"):
        cm = metrics["confusion_matrix"]

        plt.figure(figsize=(5, 5))
        plt.imshow(cm)

        plt.xticks([0, 1], ["Different", "Same"])
        plt.yticks([0, 1], ["Different", "Same"])

        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()
