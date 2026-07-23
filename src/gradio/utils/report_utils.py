from pathlib import Path

from src.gradio.utils.report_parser import parse_report

TEMPLATE = Path(__file__).parent.parent / "templates" / "report.md"


def render_report(console_output: str) -> str:
    """
    Render markdown report from console output.
    """
    template = TEMPLATE.read_text(encoding="utf-8")

    values = parse_report(console_output)

    return template.format(**values)
