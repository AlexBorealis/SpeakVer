import os
import subprocess

import gradio as gr

from src.speaker_verifier.speaker_verifier import SpeakerVerifier

# ============================================================
# Paths
# ============================================================
RUNS_DIR = "runs"


# ============================================================
# Models utils
# ============================================================
def get_models():
    if not os.path.exists(RUNS_DIR):
        return []

    return sorted(
        d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))
    )


def get_checkpoints(exp_name):
    if not exp_name or exp_name == "Default model":
        return []

    weights_dir = os.path.join(
        RUNS_DIR,
        exp_name,
        "weights",
    )

    if not os.path.exists(weights_dir):
        return []

    return sorted(f for f in os.listdir(weights_dir) if f.endswith(".pt"))


def resolve_checkpoint(exp, checkpoint=None):
    """
    Возвращает абсолютный путь до чекпоинта.
    None -> использовать предобученную SpeechBrain модель.
    """
    if exp is None or exp == "" or exp == "Default model":
        return None

    if checkpoint and checkpoint != "Default model":
        path = os.path.join(
            RUNS_DIR,
            exp,
            "weights",
            checkpoint,
        )
    else:
        path = os.path.join(
            RUNS_DIR,
            exp,
            "weights",
            "best.pt",
        )

    if os.path.exists(path):
        return path

    return None


# ============================================================
# Global verifier
# ============================================================
current_checkpoint = None

verifier = SpeakerVerifier(
    checkpoint=None,
    threshold=0.15,
    device="cpu",
)


# ============================================================
# Model loading
# ============================================================
def load_model(exp, checkpoint):
    global verifier
    global current_checkpoint

    path = resolve_checkpoint(
        exp,
        checkpoint,
    )

    verifier = SpeakerVerifier(
        checkpoint=path,
        threshold=0.15,
        device="cpu",
    )

    current_checkpoint = path

    if path is None:
        status = "Using default SpeechBrain ECAPA-TDNN"
    else:
        status = f"Loaded checkpoint:\n{path}"

    return (
        status,
        gr.update(
            minimum=-1,
            maximum=1,
            value=0.15,
            step=0.01,
        ),
    )


# ============================================================
# Dynamic checkpoint list
# ============================================================
def update_checkpoints(exp):
    if exp is None or exp == "" or exp == "Default model":
        return gr.update(
            choices=["Default model"],
            value="Default model",
        )

    checkpoints = get_checkpoints(exp)

    return gr.update(
        choices=["Default model"] + checkpoints,
        value="Default model",
    )


# ============================================================
# Verification
# ============================================================
def compare(audio1, audio2, threshold):
    global verifier

    if audio1 is None or audio2 is None:
        return (
            "Please upload two audio files.",
            "",
            "",
        )

    verifier.threshold = threshold

    result = verifier.compare_audio(
        audio1,
        audio2,
    )

    prediction = "SAME" if result["same"] else "DIFFERENT"

    return (
        prediction,
        f"{result['confidence']:.4f}",
        f"{result['score']:.4f}",
    )


# ============================================================
# Report
# ============================================================
def run_report(
    model,
    report_dir,
    dataset_path,
    balance,
):
    cmd = [
        "python",
        "report.py",
        "--dataset_path",
        dataset_path,
    ]

    checkpoint = resolve_checkpoint(model)

    if checkpoint is not None:
        cmd.extend(
            [
                "--model_path",
                checkpoint,
            ]
        )

    if report_dir and report_dir.strip():
        cmd.extend(
            [
                "--output_dir",
                report_dir.strip(),
            ]
        )

    if not balance:
        cmd.append("--no-balance")

    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    return process.stdout + process.stderr


# ============================================================
# UI
# ============================================================
models = get_models()

with gr.Blocks(title="Speaker Verification") as demo:
    gr.Markdown("# Speaker Verification")

    with gr.Tabs():
        # ====================================================
        # Verification
        # ====================================================
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
            )

            threshold = gr.Slider(
                minimum=-1,
                maximum=1,
                value=0.15,
                step=0.01,
                label="Cosine similarity threshold",
            )

            experiment.change(
                update_checkpoints,
                inputs=experiment,
                outputs=checkpoint,
            )

            checkpoint.change(
                load_model,
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

            compare_btn = gr.Button(
                "Compare",
                variant="primary",
            )

            prediction = gr.Textbox(label="Prediction")
            confidence = gr.Textbox(label="Confidence")
            cosine = gr.Textbox(label="Cosine similarity")

            compare_btn.click(
                compare,
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

        # ====================================================
        # Report
        # ====================================================
        with gr.Tab("Report"):
            report_exp = gr.Dropdown(
                choices=["Default model"] + models,
                value="Default model",
                label="Experiment",
            )

            report_dir = gr.Textbox(
                label="Report directory name",
                placeholder="Leave empty for report_YYYYMMDD_HHMMSS",
            )

            report_dataset = gr.Textbox(
                value="speaker_dataset/test",
                label="Dataset path",
            )

            report_balance = gr.Checkbox(
                label="Balance positive/negative pairs",
                value=True,
            )

            report_btn = gr.Button(
                "Generate report",
                variant="primary",
            )

            report_log = gr.Textbox(
                label="Console output",
                lines=30,
            )

            report_btn.click(
                run_report,
                inputs=[
                    report_exp,
                    report_dir,
                    report_dataset,
                    report_balance,
                ],
                outputs=report_log,
            )

demo.launch()
