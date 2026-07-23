import argparse

from huggingface_hub import login
from tqdm import tqdm

from datasets import load_dataset
from src.config import HF_TOKEN
from src.data.audio_preprocessor import AudioPreprocessor

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="Download VibraVox Dataset")

parser.add_argument(
    "--config_path",
    type=str,
    default="config/envs/.env",
    help="Path to dataset",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    default="speaker_dataset",
    help="Path to dataset",
)

parser.add_argument(
    "--n",
    type=str,
    default=100,
    help="Count files",
)

args = parser.parse_args()

# ============================================================
# Environments
# ============================================================
login(token=HF_TOKEN)

# ============================================================
# Downloading Data
# ============================================================
dataset = load_dataset(
    "Cnam-LMSSC/vibravox", "speech_clean", split="train", streaming=True
)

main_list = []
for sample in tqdm(dataset.take(args.n), total=args.n):
    main_list.append(sample)

preprocessor = AudioPreprocessor()

# ============================================================
# Saving Data
# ============================================================
preprocessor.save_samples(main_list, output_dir=args.dataset_path)
