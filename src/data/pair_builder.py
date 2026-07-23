import random
from itertools import combinations


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
        # Все аудиозаписи
        samples = list(dataset)
        
        # Все уникальные пары
        positive_pairs = []
        negative_pairs = []

        for sample1, sample2 in combinations(samples, 2):
            pair = {
                "sample1": sample1,
                "sample2": sample2,
                "label": int(sample1["speaker_id"] == sample2["speaker_id"]),
            }

            if pair["label"] == 1:
                positive_pairs.append(pair)
            else:
                negative_pairs.append(pair)
                
        # Перемешивание
        random.shuffle(positive_pairs)
        random.shuffle(negative_pairs)
        
        # Ограничение количества
        if self.max_positive_pairs is not None:
            positive_pairs = positive_pairs[
                : min(self.max_positive_pairs, len(positive_pairs))
            ]

        if self.max_negative_pairs is not None:
            negative_pairs = negative_pairs[
                : min(self.max_negative_pairs, len(negative_pairs))
            ]
            
        # Балансировка
        if self.balance:
            n = min(len(positive_pairs), len(negative_pairs))
            positive_pairs = positive_pairs[:n]
            negative_pairs = negative_pairs[:n]
            
        # Итоговый датасет
        pairs = positive_pairs + negative_pairs
        random.shuffle(pairs)

        print("=" * 60)
        print()
                
        print("Pair Dataset")
        print("=" * 60)
        print(f"Positive pairs                        : {len(positive_pairs)}")
        print(f"Negative pairs                        : {len(negative_pairs)}")
        print(f"Total pairs                           : {len(pairs)}")
        print()

        return pairs