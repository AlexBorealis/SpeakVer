import os

from dotenv import load_dotenv
from huggingface_hub import login

from datasets import load_dataset
from src.data.audio_preprocessor import AudioPreprocessor

load_dotenv(
    dotenv_path="/home/borealis/Documents/work/projects/work_tests/SpeakVer/config/envs/.env"
)
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

n = 100

dataset = load_dataset(
    "Cnam-LMSSC/vibravox", "speech_clean", split="train", streaming=True
)
mini_dataset = dataset.take(n)

main_list = list(mini_dataset)

preprocessor = AudioPreprocessor()

preprocessor.save_samples(main_list, "debug_audio")
