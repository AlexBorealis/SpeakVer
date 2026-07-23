import re


def _extract(
    text: str,
    key: str,
    default: str = "-",
    as_int: bool = False,
) -> str:
    match = re.search(
        rf"{re.escape(key)}\s*:\s*(.+)",
        text,
    )

    value = match.group(1).strip() if match else default

    if as_int:
        value = str(int(float(value)))

    return value


def parse_report(console_output: str) -> dict:
    """
    Parse report.py console output into template variables.
    """

    return {
        # Dataset
        "num_speakers": _extract(console_output, "Count speakers", as_int=True),
        "num_samples": _extract(console_output, "Count audio files", as_int=True),
        "min_qty": _extract(console_output, "Min count recordings", as_int=True),
        "max_qty": _extract(console_output, "Max count recordings", as_int=True),
        "mean_qty": _extract(console_output, "Mean count recordings"),
        "median_qty": _extract(console_output, "Median count recordings", as_int=True),
        "std_qty": _extract(console_output, "Std count recordings"),
        # Duration
        "min_duration": _extract(console_output, "Min duration recordings"),
        "max_duration": _extract(console_output, "Max duration recordings"),
        "mean_duration": _extract(console_output, "Mean duration recordings"),
        "median_duration": _extract(console_output, "Median duration recordings"),
        "std_duration": _extract(console_output, "Std duration recordings"),
        # Metrics
        "accuracy": _extract(console_output, "Accuracy"),
        "precision": _extract(console_output, "Precision"),
        "recall": _extract(console_output, "Recall"),
        "f1": _extract(console_output, "F1-score"),
        "roc_auc": _extract(console_output, "ROC AUC"),
        "eer": _extract(console_output, "EER"),
        "threshold": _extract(console_output, "Threshold"),
        # Report
        "directory": _extract(console_output, "Directory"),
    }
