import datetime
import os
import subprocess

import gradio as gr

from src.speaker_verifier.speaker_verifier import SpeakerVerifier

# ============================================================
# Paths
# ============================================================
RUNS_DIR = "runs/speaker_train"


def get_experiments():
    """
    Все папки экспериментов
    """
    if not os.path.exists(RUNS_DIR):
        return []

    return sorted(
        [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    )


def get_checkpoints(exp_name):
    """
    Получение чекпоинтов выбранного эксперимента
    """
    if exp_name is None:
        return []

    weights_dir = os.path.join(RUNS_DIR, exp_name, "weights")

    if not os.path.exists(weights_dir):
        return []

    return sorted([f for f in os.listdir(weights_dir) if f.endswith(".pt")])


# ============================================================
# Global verifier
# ============================================================
current_checkpoint = None

# Default model
verifier = SpeakerVerifier(checkpoint=None, threshold=0.15, device="cpu")


# ============================================================
# Model loading
# ============================================================
def load_model(exp, checkpoint):
    global verifier
    global current_checkpoint

    # Default model
    if exp is None or exp == "" or checkpoint is None or checkpoint == "":
        verifier = SpeakerVerifier(checkpoint=None, threshold=0.15, device="cpu")
        current_checkpoint = None
        return (
            "Using default model: EmbeddingExtractor()",
            gr.update(minimum=-1, maximum=1, value=0.15, step=0.01),
        )

    path = os.path.join(RUNS_DIR, exp, "weights", checkpoint)

    verifier = SpeakerVerifier(checkpoint=path, threshold=0.15, device="cpu")

    current_checkpoint = path

    return (f"Loaded:\n{path}", gr.update(minimum=-1, maximum=1, value=0.15, step=0.01))


# ============================================================
# Dynamic dropdown callbacks
# ============================================================
def update_checkpoints(exp):
    checkpoints = get_checkpoints(exp)

    return gr.update(choices=checkpoints, value=checkpoints[0] if checkpoints else None)


# ============================================================
# Verification
# ============================================================
def compare(audio1, audio2, threshold):
    global verifier
    if verifier is None:
        verifier = SpeakerVerifier(checkpoint=None, threshold=threshold, device="cpu")
    if audio1 is None or audio2 is None:
        return ("Please upload two audio files.", "", "")

    verifier.set_threshold(threshold)

    result = verifier.compare_audio(audio1, audio2)

    prediction = "SAME" if result["same"] else "DIFFERENT"

    return (prediction, f"{result['confidence']:.4f}", f"{result['score']:.4f}")


# ============================================================
# Report
# ============================================================
def run_report(exp, report_dir, dataset_path):
    # Создание папки отчета
    if report_dir is None or report_dir.strip() == "":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = os.path.join("reports", f"report_{timestamp}")
    else:
        output_dir = os.path.join("reports", report_dir.strip())

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python",
        "report.py",
        "--output_dir",
        output_dir,
        "--dataset_path",
        dataset_path,
    ]
    # Только если выбран эксперимент
    if exp is not None and exp.strip() != "":
        cmd.extend(["--experiment", exp])

    process = subprocess.run(cmd, capture_output=True, text=True)

    return f"Report directory:\n{output_dir}\n\n" + process.stdout + process.stderr


# ============================================================
# UI
# ============================================================
experiments = get_experiments()

with gr.Blocks(title="Speaker Verification") as demo:
    gr.Markdown("# Speaker Verification")

    with gr.Tabs():
        # ====================================================
        # Verification
        # ====================================================
        with gr.Tab("Verification"):
            with gr.Row():
                experiment = gr.Dropdown(
                    choices=experiments,
                    label="Training experiment",
                    value=None if experiments else None,
                )

                checkpoint = gr.Dropdown(label="Checkpoint", choices=[])

            model_status = gr.Textbox(label="Model status")

            threshold = gr.Slider(
                minimum=-1,
                maximum=1,
                value=0.15,
                step=0.01,
                label="Cosine similarity threshold",
            )

            experiment.change(
                update_checkpoints, inputs=[experiment], outputs=[checkpoint]
            )

            checkpoint.change(
                load_model,
                inputs=[experiment, checkpoint],
                outputs=[model_status, threshold],
            )

            with gr.Row():
                audio1 = gr.Audio(type="filepath", label="Audio 1")

                audio2 = gr.Audio(type="filepath", label="Audio 2")

            compare_btn = gr.Button("Compare", variant="primary")

            prediction = gr.Textbox(label="Prediction")

            confidence = gr.Textbox(label="Confidence")

            cosine = gr.Textbox(label="Cosine similarity")

            compare_btn.click(
                compare,
                inputs=[audio1, audio2, threshold],
                outputs=[prediction, confidence, cosine],
            )

        # ====================================================
        # Report
        # ====================================================
        with gr.Tab("Report"):
            report_exp = gr.Dropdown(
                choices=experiments,
                label="Report experiment",
                value=None if experiments else None,
            )

            report_dir = gr.Textbox(
                label="Report directory name",
                placeholder=("Leave empty for automatic name: report_YYYYMMDD_HHMMSS"),
                value="",
            )

            report_dataset = gr.Textbox(
                label="Test dataset path", value="speaker_dataset/test"
            )

            report_btn = gr.Button("Generate report", variant="primary")

            report_log = gr.Textbox(label="Console output", lines=30)

            report_btn.click(
                run_report,
                inputs=[report_exp, report_dir, report_dataset],
                outputs=[report_log],
            )

demo.launch()
