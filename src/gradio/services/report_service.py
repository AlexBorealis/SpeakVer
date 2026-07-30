from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    ARCHIVES_DIR,
    DEFAULT_DEVICE,
    DEFAULT_REPORT_DATASET,
)
from src.gradio.utils.archive import make_zip
from src.gradio.utils.model_utils import resolve_checkpoint


@dataclass
class ReportResult:
    """
    Report generation result.
    """

    success: bool
    output_dir: Path | None
    archive_path: str | None
    console_output: str


class ReportService:
    """
    Service responsible for generating evaluation reports.
    """

    @staticmethod
    def _extract_report_dir(
        console_output: str,
    ) -> Path | None:
        """
        Extract generated report directory.
        """

        match = re.search(
            r"Report\s*:\s*(.+)",
            console_output,
        )

        if not match:
            return None

        return Path(match.group(1).strip())

    @staticmethod
    def generate(
        experiment: str,
        report_dir: str | None = None,
        dataset_path: str = DEFAULT_REPORT_DATASET,
        device: str = DEFAULT_DEVICE,
        negative_ratio: float | None = 10,
        balance: bool = False,
    ) -> ReportResult:
        cmd = [
            "python",
            "report.py",
            "--dataset_path",
            dataset_path,
            "--device",
            device,
            "--disable",
        ]

        if report_dir and report_dir.strip():
            cmd.extend(
                [
                    "--output_dir",
                    report_dir.strip(),
                ]
            )

        checkpoint = resolve_checkpoint(experiment)

        if checkpoint is not None:
            cmd.extend(
                [
                    "--model_path",
                    str(checkpoint),
                ]
            )

        if negative_ratio:
            cmd.extend(
                [
                    "--negative_ratio",
                    str(negative_ratio),
                ]
            )

        if balance:
            cmd.append("--balance")

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        console_output = process.stdout + process.stderr

        output_dir = ReportService._extract_report_dir(console_output)

        archive_path = None

        if process.returncode == 0 and output_dir and output_dir.exists():
            ARCHIVES_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            archive_path = make_zip(
                output_dir,
                ARCHIVES_DIR,
            )

        return ReportResult(
            success=process.returncode == 0,
            output_dir=output_dir,
            archive_path=archive_path,
            console_output=console_output,
        )
