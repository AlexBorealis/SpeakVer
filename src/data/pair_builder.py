import pickle
import random
from itertools import combinations
from pathlib import Path


class PairBuilder:
    def __init__(
        self,
        random_seed: int = 42,
        balance: bool = False,
        negative_ratio: float | None = None,
        max_positive_pairs: int | None = None,
        max_negative_pairs: int | None = None,
        cache_filename: str | None = None,
        disable: bool = True,
    ):
        """
        Builds speaker verification pairs.

        Disk cache:
            - First run:
                generate pairs -> save .pkl

            - Next runs:
                load pairs from .pkl

        RAM cache is handled by Trainer.

        Parameters
        ----------
        balance : bool
            Balance positive and negative pairs 1:1.

        negative_ratio : int | float | None
            Number of negative pairs relative to positive pairs.

            Example:
                positive = 1000
                negative_ratio = 10

                negative = 10000

        max_positive_pairs : int | None
            Maximum positive pairs.

        max_negative_pairs : int | None
            Maximum negative pairs.

        cache_filename : str | Path | None
            Filename to pickle cache.

        disable : bool
            Disable statistics output.
        """

        self.rng = random.Random(random_seed)
        self.random_seed = random_seed

        self.balance = balance
        self.negative_ratio = negative_ratio

        self.max_positive_pairs = max_positive_pairs
        self.max_negative_pairs = max_negative_pairs

        self.cache_filename = (
            Path(cache_filename) if cache_filename is not None else None
        )

        self.disable = disable

    # ==========================================================
    # Cache
    # ==========================================================
    def _load_cache(self):
        if self.cache_filename is None or not self.cache_filename.exists():
            return None

        print(f"Loading cached pairs: {self.cache_filename}")

        with open(
            self.cache_filename,
            "rb",
        ) as f:
            return pickle.load(f)

    def _save_cache(self, pairs):
        if self.cache_filename is None:
            return

        self.cache_filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            self.cache_filename,
            "wb",
        ) as f:
            pickle.dump(
                pairs,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print(f"Saved pairs cache: {self.cache_filename}")

    def build(self, dataset):
        # Load cache
        cached_pairs = self._load_cache()

        if cached_pairs is not None:
            return cached_pairs

        positive_pairs = []
        negative_pairs = []

        for sample1, sample2 in combinations(dataset, 2):
            pair = {
                "sample1": sample1,
                "sample2": sample2,
                "label": int(sample1["speaker_id"] == sample2["speaker_id"]),
            }

            if pair["label"]:
                positive_pairs.append(pair)
            else:
                negative_pairs.append(pair)

        # Shuffle
        self.rng.shuffle(positive_pairs)
        self.rng.shuffle(negative_pairs)

        # Manual limits
        if self.max_positive_pairs is not None:
            positive_pairs = positive_pairs[: self.max_positive_pairs]

        if self.max_negative_pairs is not None:
            negative_pairs = negative_pairs[: self.max_negative_pairs]

        # Balance
        if self.balance:
            n = min(len(positive_pairs), len(negative_pairs))
            positive_pairs = positive_pairs[:n]
            negative_pairs = negative_pairs[:n]

        # Negative ratio
        elif self.negative_ratio is not None:
            max_negatives = int(len(positive_pairs) * self.negative_ratio)

            if len(negative_pairs) > max_negatives:
                negative_pairs = negative_pairs[:max_negatives]

        pairs = positive_pairs + negative_pairs
        self.rng.shuffle(pairs)

        # Save cache
        self._save_cache(pairs)

        if not self.disable:
            ratio = (
                f"1:{len(negative_pairs) / len(positive_pairs):.2f}"
                if len(positive_pairs)
                else "-"
            )

            print("=" * 60)
            print()
            print("Pair Dataset")
            print("=" * 60)
            print(f"Positive pairs                        : {len(positive_pairs)}")
            print(f"Negative pairs                        : {len(negative_pairs)}")
            print(f"Positive/Negative ratio               : {ratio}")
            print(f"Total pairs                           : {len(pairs)}")
            print()

        return pairs
