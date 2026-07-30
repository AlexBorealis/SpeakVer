import gradio as gr
from src.config import DEFAULT_REPORT_DATASET
from src.gradio.services.report_service import ReportService
from src.gradio.utils.model_utils import get_models
from src.gradio.utils.report_utils import render_report


def generate_report(
    experiment,
    report_dir,
    dataset_path,
    device,
    negative_ratio,
    balance,
):
    """
    Generate evaluation report.
    """

    result = ReportService.generate(
        experiment=experiment,
        report_dir=report_dir,
        dataset_path=dataset_path,
        device=device,
        negative_ratio=negative_ratio,
        balance=balance,
    )

    if not result.success:
        return (
            f"```text\n{result.console_output}\n```",
            gr.update(
                visible=False,
            ),
        )

    return (
        render_report(result.console_output),
        gr.update(
            value=result.archive_path,
            visible=True,
        ),
    )


def create_report_tab():
    """
    Create report tab.
    """

    models = get_models()

    with gr.Tab("Report"):
        with gr.Row():
            report_experiment = gr.Dropdown(
                choices=["Default model"] + models,
                value="Default model",
                label="Experiment",
            )

            report_device = gr.Dropdown(
                choices=["cpu", "cuda"],
                value="cuda",
                label="Device",
            )

        report_directory = gr.Textbox(
            label="Report directory",
            placeholder="Leave empty to generate report_YYYYMMDD_HHMMSS",
        )

        report_dataset = gr.Textbox(
            value=DEFAULT_REPORT_DATASET,
            label="Dataset path",
        )

        report_negative_ratio = gr.Slider(
            minimum=0.01,
            value=10,
            step=0.1,
            label="How much times negative pairs bigger than positive",
        )

        report_balance = gr.Checkbox(
            value=False,
            label="Balance positive/negative pairs",
        )

        generate_button = gr.Button(
            "Generate report",
            variant="primary",
        )

        report_console = gr.Markdown(
            label="Report summary",
        )

        download_report = gr.DownloadButton(
            label="Download report (.zip)",
            visible=False,
        )

        generate_button.click(
            fn=generate_report,
            inputs=[
                report_experiment,
                report_directory,
                report_dataset,
                report_device,
                report_negative_ratio,
                report_balance,
            ],
            outputs=[
                report_console,
                download_report,
            ],
        )
