import math
from typing import Iterator

import numpy as np
from torch.utils.data import BatchSampler


class DynamicBatchSampler(BatchSampler):
    """
    Dynamic batching by padded batch size.

    Batch constraint:

        max_length_in_batch * batch_size <= max_batch_length

    Parameters
    ----------
    lengths:
        Lengths of preprocessed audio samples.

    max_batch_length:
        Maximum amount of samples after padding.

    max_batch_size:
        Hard limit for number of samples in batch.

    min_batch_size:
        Minimum allowed batch size.
        Needed because ECAPA contains BatchNorm.

    bucket_size:
        Sorting window.

    shuffle:
        Shuffle bucket order only.

    drop_last:
        Drop batches smaller than min_batch_size.
    """

    def __init__(
        self,
        lengths: list[int] | np.ndarray,
        max_batch_length: int,
        max_batch_size: int = 64,
        min_batch_size: int = 8,
        bucket_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):

        self.lengths = np.asarray(lengths)

        self.max_batch_length = max_batch_length
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

        self.bucket_size = bucket_size

        self.shuffle = shuffle
        self.drop_last = drop_last

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

    def __iter__(self) -> Iterator[list[int]]:
        indices = np.lexsort(
            (
                self.rng.random(len(self.lengths)),
                self.lengths,
            )
        )

        buckets = [
            indices[i : i + self.bucket_size]
            for i in range(
                0,
                len(indices),
                self.bucket_size,
            )
        ]

        if self.shuffle:
            self.rng.shuffle(buckets)

        batches = []

        for bucket in buckets:
            batch = []
            batch_max_length = 0

            for idx in bucket:
                length = int(self.lengths[idx])

                candidate_max = max(batch_max_length, length)
                candidate_size = len(batch) + 1

                overflow = (
                    candidate_size > self.max_batch_size
                    or candidate_max * candidate_size > self.max_batch_length
                )

                if batch and overflow:
                    if len(batch) >= self.min_batch_size:
                        batches.append(batch)

                    batch = []
                    batch_max_length = 0

                    candidate_max = length

                batch.append(int(idx))
                batch_max_length = candidate_max

            if batch:
                if len(batch) >= self.min_batch_size:
                    batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        yield from batches

    def __len__(self):
        avg_length = float(np.mean(self.lengths))

        estimated_batch = min(
            self.max_batch_size,
            max(
                self.min_batch_size,
                int(self.max_batch_length / avg_length),
            ),
        )

        return math.ceil(len(self.lengths) / estimated_batch)
