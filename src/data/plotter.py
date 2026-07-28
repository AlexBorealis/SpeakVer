import matplotlib.pyplot as plt
import numpy as np


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
        cm = np.array(metrics["confusion_matrix"])
        cm_norm = np.array(metrics["confusion_matrix_normalized"])

        # Создаем холст с двумя графиками (1 строка, 2 колонки)
        _, axes = plt.subplots(1, 2, figsize=(11, 5))
        labels = ["Different", "Same"]

        # --- Левый график: Обычная матрица ошибок ---
        axes[0].imshow(cm, cmap="Blues")  # cmap добавит контраста тексту
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(labels)
        axes[0].set_yticks([0, 1])
        axes[0].set_yticklabels(labels)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Ground Truth")
        axes[0].set_title("Confusion Matrix")

        # --- Правый график: Нормализованная матрица ошибок ---
        axes[1].imshow(cm_norm, cmap="Blues")
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(labels)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(labels)
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Ground Truth")
        axes[1].set_title("Normalized Confusion Matrix")

        # Добавляем текстовые значения в ячейки обоих графиков
        for i in range(2):
            for j in range(2):
                # Текст для обычной матрицы (целые числа)
                axes[0].text(
                    j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=14
                )
                # Текст для нормализованной (проценты с 2 знаками после запятой)
                axes[1].text(
                    j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=14
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
