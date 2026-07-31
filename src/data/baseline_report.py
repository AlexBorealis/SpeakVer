import json
import os

import numpy as np
import pandas as pd
import torch


class BaselineReport:
    def __init__(self, output_dir: str = "baseline_report"):
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, metrics, pairs, scores, labels, dataset_stats=None, plotter=None):
        """
        Save complete verification report.

        Parameters
        ----------
        metrics:
            Verification metrics.

        pairs:
            Validation pairs.

        scores:
            Cosine similarity scores.

        labels:
            Pair labels.

        dataset_stats:
            Dataset statistics from SpeakerDataset.statistics()

        plotter:
            Plotter instance.
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Metrics
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

        # Pair statistics
        rows = []

        for pair, score, label in zip(pairs, scores, labels):
            sample1 = pair["sample1"]
            sample2 = pair["sample2"]

            rows.append(
                {
                    "speaker1": sample1["speaker_id"],
                    "recording1": sample1["recording"],
                    "audio_path1": sample1.get("audio_path"),

                    "speaker2": sample2["speaker_id"],
                    "recording2": sample2["recording"],
                    "audio_path2": sample2.get("audio_path"),

                    "label": int(label),
                    "cosine_similarity": float(score),
                }
            )

        df = pd.DataFrame(rows)

        df.to_csv(os.path.join(self.output_dir, "pairs.csv"), index=False)

        # Plots
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

        # Dataset statistics
        if dataset_stats is not None:
            dataset_stats_json = {}

            for key, value in dataset_stats.items():
                if isinstance(value, np.ndarray):
                    dataset_stats_json[key] = value.tolist()

                elif isinstance(value, np.generic):
                    dataset_stats_json[key] = value.item()

                elif isinstance(value, dict):
                    dataset_stats_json[key] = dict(value)

                else:
                    dataset_stats_json[key] = value

            with open(
                os.path.join(
                    self.output_dir,
                    "dataset_statistics.json",
                ),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    dataset_stats_json,
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

        # Summary
        positives = int(np.sum(labels))
        negatives = int(len(labels) - positives)

        with open(
            os.path.join(self.output_dir, "summary.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("Dataset statistics\n")
            f.write("=" * 60 + "\n")
            f.write(f"Positive pairs                  : {positives}\n")
            f.write(f"Negative pairs                  : {negatives}\n")
            f.write(f"Total pairs                     : {len(labels)}\n\n")

            if dataset_stats is not None:
                f.write(
                    f"Speakers                        : {dataset_stats['num_speakers']}\n"
                )
                f.write(
                    f"Audio files                     : {dataset_stats['num_samples']}\n"
                )
                f.write(
                    f"Min count recordings            : {dataset_stats['min_qty']}\n"
                )
                f.write(
                    f"Max count recordings            : {dataset_stats['max_qty']}\n"
                )
                f.write(
                    f"Mean count recordings           : {dataset_stats['mean_qty']:.2f}\n"
                )
                f.write(
                    f"Median count recordings         : {dataset_stats['median_qty']:.2f}\n"
                )
                f.write(
                    f"Std count recordings            : {dataset_stats['std_qty']:.2f}\n\n"
                )
                f.write(
                    f"Min duration recordings         : {dataset_stats['min_duration']:.2f}\n"
                )
                f.write(
                    f"Max duration recordings         : {dataset_stats['max_duration']:.2f}\n"
                )
                f.write(
                    f"Mean duration recordings        : {dataset_stats['mean_duration']:.2f}\n"
                )
                f.write(
                    f"Median duration recordings      : {dataset_stats['median_duration']:.2f}\n"
                )
                f.write(
                    f"Std duration recordings         : {dataset_stats['std_duration']:.2f}\n"
                )
                f.write("=" * 60 + "\n\n")

            f.write("ECAPA-TDNN report\n")
            f.write("=" * 60 + "\n")

            f.write(f"Accuracy                        : {metrics['accuracy']:.4f}\n")
            f.write(f"Precision                       : {metrics['precision']:.4f}\n")
            f.write(f"Recall                          : {metrics['recall']:.4f}\n")
            f.write(f"F1-score                        : {metrics['f1']:.4f}\n")
            f.write(f"ROC-AUC                         : {metrics['roc_auc']:.4f}\n")
            f.write(f"EER                             : {metrics['eer']:.4f}\n")
            f.write(f"MinDCF                          : {metrics['min_dcf']:.4f}\n")
            f.write(f"Threshold                       : {metrics['threshold']:.4f}\n\n")
