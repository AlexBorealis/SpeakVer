import json
import os

import numpy as np
import pandas as pd
import torch


class BaselineReport:
    def __init__(self, output_dir="baseline_report"):
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, metrics, pairs, scores, labels, plotter=None):
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        metrics_json = {}

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()

            elif isinstance(value, np.generic):
                metrics_json[key] = value.item()

            else:
                metrics_json[key] = value

        with open(
            os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metrics_json, f, indent=4, ensure_ascii=False)

        rows = []

        for pair, score, label in zip(pairs, scores, labels):
            sample1 = pair["sample1"]
            sample2 = pair["sample2"]

            rows.append(
                {
                    "speaker1": sample1["speaker_id"],
                    "speaker2": sample2["speaker_id"],
                    "label": int(label),
                    "cosine_similarity": float(score),
                }
            )

        df = pd.DataFrame(rows)

        df.to_csv(os.path.join(self.output_dir, "pairs.csv"), index=False)

        if plotter is not None:
            plotter.plot_similarity(
                labels,
                scores,
                threshold=metrics["threshold"],
                save_path=os.path.join(self.output_dir, "similarity_distribution.png"),
            )

            plotter.plot_roc(
                metrics, save_path=os.path.join(self.output_dir, "roc_curve.png")
            )

            plotter.plot_confusion_matrix(
                metrics, save_path=os.path.join(self.output_dir, "confusion_matrix.png")
            )

        positives = int(np.sum(labels))
        negatives = int(len(labels) - positives)

        with open(
            os.path.join(self.output_dir, "summary.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("ECAPA-TDNN BASELINE REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Accuracy  : {metrics['accuracy']:.4f}\n")
            f.write(f"Precision : {metrics['precision']:.4f}\n")
            f.write(f"Recall    : {metrics['recall']:.4f}\n")
            f.write(f"F1-score  : {metrics['f1']:.4f}\n")
            f.write(f"ROC-AUC   : {metrics['roc_auc']:.4f}\n")
            f.write(f"EER       : {metrics['eer']:.4f}\n")
            f.write(f"Threshold : {metrics['threshold']:.4f}\n\n")

            f.write(f"Positive pairs : {positives}\n")
            f.write(f"Negative pairs : {negatives}\n")
            f.write(f"Total pairs    : {len(labels)}\n")

        print()
        print("=" * 60)
        print("Baseline report successfully saved")
        print(f"Directory: {self.output_dir}")
        print("=" * 60)
