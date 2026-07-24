import argparse
import os
import sys
import traceback
from datetime import datetime

from huggingface_hub import login
from tqdm import tqdm

from datasets import load_dataset
from src.config import HF_TOKEN
from src.data.audio_preprocessor import AudioPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download VibraVox dataset from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            Download first 3000 samples:
                python -m download_data --n 3000

            Download in background:
                nohup bash -c 'echo "PID: $$"; exec python -u -m download_data --n 3000' > download_data.log 2>&1 &

            Monitor progress:
                tail -f download_data.log
        """,
    )

    parser.add_argument(
        "--config_path",
        type=str,
        default="config/envs/.env",
        help="Path to environment configuration.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="speaker_dataset",
        help="Directory where dataset will be saved.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split of the data to load.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of samples to download.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of samples processed before writing to disk.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print(f"Started : {datetime.now()}")
    print(f"PID     : {os.getpid()}")
    print(f"Samples : {args.n}")
    print(f"Batch   : {args.batch_size}")
    print("=" * 80)

    login(token=HF_TOKEN)

    dataset = load_dataset(
        "Cnam-LMSSC/vibravox",
        "speech_clean",
        split=args.split,
        streaming=True,
    )

    preprocessor = AudioPreprocessor()

    batch = []
    saved = 0
    batch_idx = 1

    progress = tqdm(
        dataset.take(args.n),
        total=args.n,
        unit="sample",
    )

    for sample in progress:
        batch.append(sample)

        if len(batch) >= args.batch_size:
            print(
                f"[{datetime.now()}] "
                f"Saving batch #{batch_idx} ({len(batch)} samples)..."
            )

            preprocessor.save_samples(
                batch,
                output_dir=args.dataset_path,
            )

            saved += len(batch)
            batch.clear()
            batch_idx += 1

    if batch:
        print(f"[{datetime.now()}] Saving final batch ({len(batch)} samples)...")

        preprocessor.save_samples(
            batch,
            output_dir=args.dataset_path,
        )

        saved += len(batch)

    print("=" * 80)
    print("SUCCESS")
    print(f"Saved samples : {saved}")
    print(f"Finished      : {datetime.now()}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)

    except Exception:
        print("=" * 80)
        print("ERROR")
        print(traceback.format_exc())
        print("=" * 80)

        sys.exit(1)
