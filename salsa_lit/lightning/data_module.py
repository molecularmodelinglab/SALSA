"""
PyTorch Lightning DataModule for SALSA.
"""

from typing import Optional, Union, List
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader

from salsa_lit.data.tokenizer import SMILESTokenizer
from salsa_lit.data.datasets import TrainingDataset, InferenceDataset
from salsa_lit.data.samplers import RandomFamSampler


class SalsaDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for SALSA training.

    Handles dataset creation and dataloader setup with proper
    family-based sampling.

    Parameters
    ----------
    train_data : str or pd.DataFrame
        Path to training data CSV or DataFrame
    batch_size : int
        Batch size for training
    num_workers : int
        Number of dataloader workers
    val_data : str or pd.DataFrame, optional
        Validation data (if None, no validation)
    n_samp : int, optional
        Number of anchor families to sample (for debugging)

    Attributes
    ----------
    train_dataset : TrainingDataset
        Training dataset
    val_dataset : TrainingDataset
        Validation dataset (if provided)
    fam_size : int
        Family size from training data
    tokenizer : SMILESTokenizer
        Tokenizer instance
    """

    def __init__(
        self,
        train_data: Union[str, pd.DataFrame],
        batch_size: int = 50,
        num_workers: int = 4,
        val_data: Union[str, pd.DataFrame] = None,
        n_samp: int = None
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_samp = n_samp

        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.fam_size = None
        self._fams_per_batch = None

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training/validation.

        Parameters
        ----------
        stage : str, optional
            'fit', 'validate', 'test', or 'predict'
        """
        self.tokenizer = SMILESTokenizer()

        if stage == 'fit' or stage is None:
            self.train_dataset = TrainingDataset(
                self.train_data,
                self.tokenizer,
                n_samp=self.n_samp
            )
            self.fam_size = self.train_dataset.fam_size

            # Compute adjusted batch size
            self._adjusted_batch_size = self.batch_size - (self.batch_size % self.fam_size)
            self._fams_per_batch = self._adjusted_batch_size // self.fam_size

            if self.val_data is not None:
                self.val_dataset = TrainingDataset(
                    self.val_data,
                    self.tokenizer
                )

    def _get_sharded_anc_ids(self, dataset: TrainingDataset) -> list:
        """
        Return anc_ids for this rank's shard of families.

        In DDP each rank receives a disjoint slice of the anchor family set
        so that every family is processed by exactly one GPU per epoch.
        In single-GPU mode the full list is returned unchanged.
        """
        anc_id_set = dataset.anc_id_set
        fam_size = dataset.fam_size

        if self.trainer is not None and self.trainer.world_size > 1:
            rank = self.trainer.global_rank
            world_size = self.trainer.world_size
            rank_anc_id_set = anc_id_set[rank::world_size]
        else:
            rank_anc_id_set = anc_id_set

        # Reconstruct the flat anc_ids list for this rank's families
        return [aid for aid in rank_anc_id_set for _ in range(fam_size)]

    def train_dataloader(self) -> DataLoader:
        """
        Create training dataloader with family sampler.

        Returns
        -------
        loader : DataLoader
            Training dataloader
        """
        sampler = RandomFamSampler(
            self._get_sharded_anc_ids(self.train_dataset),
            batch_size=self._adjusted_batch_size,
            drop_last=True
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """
        Create validation dataloader.

        Returns
        -------
        loader : DataLoader or None
            Validation dataloader if val_data provided
        """
        if self.val_dataset is None:
            return None

        sampler = RandomFamSampler(
            self._get_sharded_anc_ids(self.val_dataset),
            batch_size=self._adjusted_batch_size,
            drop_last=True
        )

        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

    @property
    def fams_per_batch(self) -> int:
        """
        Number of families per batch.

        Returns
        -------
        n : int
            Number of families per batch
        """
        if self._fams_per_batch is None:
            if self.fam_size is None:
                raise ValueError("DataModule not set up yet. Call setup() first.")
            adjusted_batch = self.batch_size - (self.batch_size % self.fam_size)
            return adjusted_batch // self.fam_size
        return self._fams_per_batch
