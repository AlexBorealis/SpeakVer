import gradio as gr
from src.config import DEFAULT_THRESHOLD
from src.gradio.services.model_service import ModelService
from src.gradio.state import state
from src.gradio.utils.model_utils import (
    get_checkpoints,
    get_models,
)


def update_checkpoints(experiment):
    """
    Update checkpoint list when experiment changes.
    """
    if experiment == "Default model":
        return gr.update(
            choices=["Default model"],
            value="Default model",
        )

    checkpoints = get_checkpoints(experiment)

    return gr.update(
        choices=["Default model"] + checkpoints,
        value="Default model",
    )


def load_model(experiment, checkpoint):
    """
    Load selected model.
    """
    status = ModelService.load_model(
        experiment,
        checkpoint,
    )

    return (
        status,
        gr.update(value=state.verifier.threshold),
    )


def compare(audio1, audio2, threshold):
    """
    Compare two audio files.
    """
    try:
        result = ModelService.compare(
            audio1,
            audio2,
            threshold,
        )
    except ValueError as e:
        return (
            str(e),
            "",
            "",
        )

    prediction = "SAME" if result["same"] else "DIFFERENT"

    return (
        prediction,
        f"{result['confidence']:.4f}",
        f"{result['score']:.4f}",
    )


def create_verification_tab():
    """
    Create verification tab.
    """
    models = get_models()

    with gr.Tab("Verification"):
        with gr.Row():
            experiment = gr.Dropdown(
                choices=["Default model"] + models,
                value="Default model",
                label="Experiment",
            )

            checkpoint = gr.Dropdown(
                choices=["Default model"],
                value="Default model",
                label="Checkpoint",
            )

        model_status = gr.Textbox(
            label="Model status",
            value="Using default SpeechBrain ECAPA-TDNN",
            interactive=False,
        )

        threshold = gr.Slider(
            minimum=-1,
            maximum=1,
            value=DEFAULT_THRESHOLD,
            step=0.01,
            label="Cosine similarity threshold",
        )

        experiment.change(
            fn=update_checkpoints,
            inputs=experiment,
            outputs=checkpoint,
        )

        checkpoint.change(
            fn=load_model,
            inputs=[
                experiment,
                checkpoint,
            ],
            outputs=[
                model_status,
                threshold,
            ],
        )

        with gr.Row():
            audio1 = gr.Audio(
                type="filepath",
                label="Audio 1",
            )

            audio2 = gr.Audio(
                type="filepath",
                label="Audio 2",
            )

        compare_button = gr.Button(
            "Compare",
            variant="primary",
        )

        prediction = gr.Textbox(
            label="Prediction",
            interactive=False,
        )

        confidence = gr.Textbox(
            label="Confidence",
            interactive=False,
        )

        cosine = gr.Textbox(
            label="Cosine similarity",
            interactive=False,
        )

        compare_button.click(
            fn=compare,
            inputs=[
                audio1,
                audio2,
                threshold,
            ],
            outputs=[
                prediction,
                confidence,
                cosine,
            ],
        )
