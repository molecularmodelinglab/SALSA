"""
Custom samplers for SALSA training.
"""

from typing import Iterator, List
import numpy as np
from torch.utils.data import Sampler, RandomSampler


class RandomFamSampler(Sampler[List[int]]):
    """
    Sampler that keeps molecule families together in batches.

    This is critical for contrastive learning where we need
    anchor-mutant pairs in the same batch.

    Parameters
    ----------
    anc_ids : list
        List of anchor IDs for each sample in dataset.
        e.g. [0,0,0,0,0,0,6,6,6,6,6,6,12,12,12,12,12,12,...]
    batch_size : int
        Desired batch size (will be adjusted to fit whole families)
    drop_last : bool
        Whether to drop incomplete final batch

    Attributes
    ----------
    fam_size : int
        Number of molecules per family
    fams_per_batch : int
        Number of families per batch
    true_batch_size : int
        Actual batch size after adjustment
    """

    def __init__(self, anc_ids: list, batch_size: int, drop_last: bool = True):
        super().__init__()
        self.anc_ids = anc_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.len_ds = len(self.anc_ids)

        self.anc_id_set = list(set(anc_ids))
        self.anc_sampler = RandomSampler(self.anc_id_set)

        # Calculate family size from first anchor
        self.fam_size = np.count_nonzero(
            np.asarray(self.anc_ids) == self.anc_ids[0]
        )
        self.fams_per_batch = self.batch_size // self.fam_size
        self.true_batch_size = self.fams_per_batch * self.fam_size

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches of indices keeping families together.

        Yields
        ------
        batch : List[int]
            List of sample indices for one batch
        """
        batch = []

        for index in self.anc_sampler:
            anc_id = self.anc_id_set[index]

            # Get all indices for this family
            fam_ids = [anc_id + i for i in range(self.fam_size)]
            batch.extend(fam_ids)

            if len(batch) == self.true_batch_size:
                yield batch
                batch = []

        # Handle remaining samples
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """
        Return number of batches.

        Returns
        -------
        n_batches : int
            Number of batches in one epoch
        """
        n_families = len(self.anc_id_set)
        if self.drop_last:
            return n_families // self.fams_per_batch
        else:
            return (n_families + self.fams_per_batch - 1) // self.fams_per_batch
