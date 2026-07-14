from collections import defaultdict
from itertools import combinations
import random


class PairBuilder:
    def __init__(
        self,
        random_seed=42,
        balance=True,
        max_positive_pairs=None,
        max_negative_pairs=None,
    ):

        self.random_seed = random_seed
        self.balance = balance

        self.max_positive_pairs = max_positive_pairs
        self.max_negative_pairs = max_negative_pairs

        random.seed(random_seed)

    def build(self, dataset):
        speakers = defaultdict(list)

        for sample in dataset:
            speakers[sample["speaker_id"]].append(sample)

        ################################################
        # Positive pairs
        ################################################

        positive_pairs = []

        for samples in speakers.values():
            for s1, s2 in combinations(samples, 2):
                positive_pairs.append({"sample1": s1, "sample2": s2, "label": 1})

        ################################################
        # Negative pairs
        ################################################

        negative_pairs = []
        speaker_ids = list(speakers.keys())

        for spk1, spk2 in combinations(speaker_ids, 2):
            s1 = random.choice(speakers[spk1])
            s2 = random.choice(speakers[spk2])

            negative_pairs.append({"sample1": s1, "sample2": s2, "label": 0})

        ################################################
        # Ограничение количества
        ################################################

        if self.max_positive_pairs:
            positive_pairs = random.sample(
                positive_pairs, min(self.max_positive_pairs, len(positive_pairs))
            )

        if self.max_negative_pairs:
            negative_pairs = random.sample(
                negative_pairs, min(self.max_negative_pairs, len(negative_pairs))
            )

        ################################################
        # Баланс
        ################################################

        if self.balance:
            n = min(len(positive_pairs), len(negative_pairs))

            positive_pairs = random.sample(positive_pairs, n)

            negative_pairs = random.sample(negative_pairs, n)

        pairs = positive_pairs + negative_pairs

        random.shuffle(pairs)

        print(f"Positive pairs: {len(positive_pairs)}")

        print(f"Negative pairs: {len(negative_pairs)}")

        print(f"Total pairs: {len(pairs)}")

        return pairs
