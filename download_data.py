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


# ============================================================
# Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download VibraVox dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            nohup bash -c '
            echo "PID: $$"
            exec python -u -m download_data \
                --dataset_path="speaker_dataset_new" \
                --batch_size=500 \
                --configs speech_clean speech_noisy \
                --splits train validation test \
                --n 5000 1500 500
            ' > download_data.log 2>&1 &
        """
    )

    parser.add_argument(
        "--dataset_path",
        default="speaker_dataset",
        type=str,
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        default=["speech_clean"],
        choices=[
            "speech_clean",
            "speech_noisy",
        ],
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "train",
            "validation",
            "test",
        ],
        choices=[
            "train",
            "validation",
            "test",
        ],
    )

    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[100],
        help=(
            "Number of samples.\n"
            "Example:\n"
            "--splits train validation test\n"
            "--n 5000 1500 500"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():

    args = parse_args()

    if len(args.n) == 1:
        split_sizes = {split: args.n[0] for split in args.splits}

    elif len(args.n) == len(args.splits):
        split_sizes = dict(
            zip(
                args.splits,
                args.n,
            )
        )

    else:
        raise ValueError(
            "Number of values in --n must be either 1 or equal to number of splits."
        )

    print("=" * 80)
    print(f"Started      : {datetime.now()}")
    print(f"PID          : {os.getpid()}")
    print(f"Configs      : {args.configs}")
    print(f"Splits       : {args.splits}")
    print(f"Samples      : {split_sizes}")
    print(f"Batch size   : {args.batch_size}")
    print("=" * 80)

    login(token=HF_TOKEN)

    preprocessor = AudioPreprocessor()

    total_saved = 0
    total_failed = 0

    for config in args.configs:
        for split in args.splits:
            requested = split_sizes[split]

            print("\n" + "=" * 80)
            print(f"Dataset  : {config}")
            print(f"Split    : {split}")
            print(f"Requested: {requested}")
            print("=" * 80)

            dataset = load_dataset(
                path="Cnam-LMSSC/vibravox",
                name=config,
                split=split,
                streaming=True,
            )

            output_dir = os.path.join(
                args.dataset_path,
                split,
                config,
            )

            batch = []

            saved = 0
            failed = 0

            batch_idx = 1

            progress = tqdm(
                dataset.take(requested),
                total=requested,
                desc=f"{split}/{config}",
                unit="sample",
            )

            for sample in progress:
                batch.append(sample)

                if len(batch) < args.batch_size:
                    continue

                print(f"[{datetime.now()}] Saving batch #{batch_idx}")

                batch_saved, batch_failed = preprocessor.save_samples(
                    batch,
                    output_dir=output_dir,
                )

                saved += batch_saved
                failed += batch_failed

                total_saved += batch_saved
                total_failed += batch_failed

                batch.clear()
                batch_idx += 1

                progress.set_postfix(
                    saved=saved,
                    failed=failed,
                )

            if batch:
                print(f"[{datetime.now()}] Saving final batch")

                batch_saved, batch_failed = preprocessor.save_samples(
                    batch,
                    output_dir=output_dir,
                )

                saved += batch_saved
                failed += batch_failed

                total_saved += batch_saved
                total_failed += batch_failed

            if saved < requested:
                print(f"WARNING: requested {requested}, saved only {saved} samples.")

            print("-" * 80)
            print(f"Finished : {split}/{config}")
            print(f"Saved    : {saved}")
            print(f"Failed   : {failed}")
            print("-" * 80)

    print("\n" + "=" * 80)
    print("SUCCESS")
    print(f"Total saved  : {total_saved}")
    print(f"Total failed : {total_failed}")
    print(f"Finished     : {datetime.now()}")
    print("=" * 80)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    try:
        login(token=HF_TOKEN)
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
