import gradio as gr
from src.gradio.ui.report_tab import create_report_tab
from src.gradio.ui.verification_tab import create_verification_tab


def create_app() -> gr.Blocks:
    """
    Create Gradio application.
    """

    with gr.Blocks(
        title="Speaker Verification",
    ) as demo:
        gr.Markdown("# Speaker Verification")

        with gr.Tabs():
            create_verification_tab()
            create_report_tab()

    return demo