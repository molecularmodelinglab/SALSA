"""
PyTorch Lightning module for SALSA training.
"""

from typing import Any, Dict

import torch
import lightning as L

from salsa_lit.config import ModelConfig, TrainingConfig
from salsa_lit.models.salsa_net import SalsaNet
from salsa_lit.models.losses import SupConLoss, cross_entropy_loss, aligniform_loss


class SalsaLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for SALSA training.

    Wraps SalsaNet and handles:
    - Loss computation with multiple loss functions
    - Optimizer configuration
    - Training/validation step logic
    - Metric logging

    Parameters
    ----------
    model_config : ModelConfig
        Configuration for SalsaNet architecture
    training_config : TrainingConfig
        Configuration for training procedure
    fam_size : int
        Family size (anchor + mutants)

    Attributes
    ----------
    model : SalsaNet
        The underlying SALSA model
    losses : list
        Active loss functions
    fam_size : int
        Family size for reshaping
    """

    def __init__(
        self,
        model_config: ModelConfig = None,
        training_config: TrainingConfig = None,
        fam_size: int = None
    ):
        super().__init__()

        if isinstance(model_config, dict):
            model_config = ModelConfig.from_dict(model_config)
        if isinstance(training_config, dict):
            training_config = TrainingConfig.from_dict(training_config)

        # Use defaults if not provided
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.fam_size = fam_size

        # Save hyperparameters for checkpointing
        self.save_hyperparameters({
            'model_config': self.model_config.to_dict(),
            'training_config': self.training_config.to_dict(),
            'fam_size': fam_size
        })

        # Create model
        self.model = SalsaNet(self.model_config, fam_size=fam_size)

        # Loss functions
        self.losses = self.training_config.losses
        self.supcon_loss = SupConLoss() if 'SupCon' in self.losses else None

    def forward(self, **batch) -> tuple:
        """
        Forward pass through SalsaNet.

        Parameters
        ----------
        **batch : dict
            Batch dictionary with seq, pad_mask, avg_mask, out_mask

        Returns
        -------
        tuple
            Model outputs (salsa_latent, inter_latent, dec_out) in training mode
            or (salsa_latent, inter_latent) in eval mode
        """
        return self.model(**batch)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Single training step.

        Parameters
        ----------
        batch : dict
            Batch from dataloader with keys: seq, pad_mask, avg_mask, out_mask
        batch_idx : int
            Index of current batch

        Returns
        -------
        loss : torch.Tensor
            Total loss for this step
        """
        salsa_latent, inter_latent, dec_out = self.model(**batch)

        loss_dict = {}

        if 'Recon' in self.losses:
            loss_dict['Recon'] = cross_entropy_loss(dec_out, **batch)

        if 'SupCon' in self.losses:
            fams_per_batch = inter_latent.shape[0]
            fam_ids = torch.arange(fams_per_batch, device=self.device)
            loss_dict['SupCon'] = self.supcon_loss(features=inter_latent, labels=fam_ids)

        if 'Aligniform' in self.losses:
            loss_dict['Aligniform'] = aligniform_loss(inter_latent)

        total_loss = sum(loss_dict.values())

        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train/{name}', value, on_step=True, on_epoch=True, 
                     prog_bar=True, sync_dist=True)
        self.log('train/total_loss', total_loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Batch from dataloader
        batch_idx : int
            Index of current batch

        Returns
        -------
        loss : torch.Tensor
            Total validation loss
        """
        salsa_latent, inter_latent, dec_out = self.model(**batch)

        loss_dict = {}

        if 'Recon' in self.losses:
            loss_dict['Recon'] = cross_entropy_loss(dec_out, **batch)

        if 'SupCon' in self.losses:
            fams_per_batch = inter_latent.shape[0]
            fam_ids = torch.arange(fams_per_batch, device=self.device)
            loss_dict['SupCon'] = self.supcon_loss(features=inter_latent, labels=fam_ids)

        if 'Aligniform' in self.losses:
            loss_dict['Aligniform'] = aligniform_loss(inter_latent)

        total_loss = sum(loss_dict.values())

        # Log validation losses
        for name, value in loss_dict.items():
            self.log(f'val/{name}', value, on_epoch=True, sync_dist=True)
        self.log('val/total_loss', total_loss, on_epoch=True, sync_dist=True)

        return total_loss

    def configure_optimizers(self):
        """
        Configure optimizer.

        Returns
        -------
        optimizer : torch.optim.Optimizer
            Adam optimizer with configured learning rate
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.lr
        )
        return optimizer

    def embed(self, seq: torch.Tensor, pad_mask: torch.Tensor = None,
              avg_mask: torch.Tensor = None, out_mask: torch.Tensor = None):
        """
        Inference method - encode SMILES to latent vectors.

        Parameters
        ----------
        seq : torch.Tensor
            Input token indices
        pad_mask : torch.Tensor, optional
            Padding mask
        avg_mask : torch.Tensor, optional
            Averaging mask
        out_mask : torch.Tensor, optional
            Output mask

        Returns
        -------
        salsa_latent : torch.Tensor
            SALSA latent vectors
        inter_latent : torch.Tensor
            Intermediate latents
        """
        self.eval()
        with torch.no_grad():
            return self.model.embed(seq, pad_mask, avg_mask, out_mask)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, **kwargs):
        """
        Load module from checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file

        Returns
        -------
        module : SalsaLightningModule
            Loaded module
        """
        return super().load_from_checkpoint(checkpoint_path, **kwargs)
