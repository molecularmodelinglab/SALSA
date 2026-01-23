"""
High-level training API for SALSA.
"""

import os
import pickle
from datetime import datetime
from typing import Union, List

import yaml
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from salsa_lit.config import ModelConfig, TrainingConfig
from salsa_lit.lightning.salsa_module import SalsaLightningModule
from salsa_lit.lightning.data_module import SalsaDataModule
from salsa_lit.data.datasets import TrainingDataset


def cook_salsa(
    train_csv: Union[str, pd.DataFrame, TrainingDataset],
    save_dir: str = None,
    val_csv: Union[str, pd.DataFrame] = None,
    model_config: ModelConfig = None,
    training_config: TrainingConfig = None,
    n_samp: int = None,
):
    """
    Train a SALSA model using PyTorch Lightning.

    This is the main entry point for training. It sets up the Lightning
    Trainer with multi-GPU support and handles checkpointing.

    Parameters
    ----------
    train_csv : str or pd.DataFrame or TrainingDataset
        Training data (mutated dataset)
    save_dir : str, optional
        Directory to save model and logs.
        Defaults to SALSA_<timestamp> in current directory.
    val_csv : str or pd.DataFrame, optional
        Validation data
    model_config : ModelConfig, optional
        Model architecture config (uses defaults if None)
    training_config : TrainingConfig, optional
        Training config (uses defaults if None)
    n_samp : int, optional
        Number of anchor families to sample (for debugging)

    Returns
    -------
    model : SalsaLightningModule
        Trained model
    trainer : L.Trainer
        Lightning trainer (for inspection)

    Example
    -------
    >>> from salsa_lit import cook_salsa, ModelConfig, TrainingConfig
    >>> model, trainer = cook_salsa(
    ...     "training_data.csv",
    ...     save_dir="./my_model",
    ...     model_config=ModelConfig(dim_salsa=64),
    ...     training_config=TrainingConfig(n_epochs=10, devices=4)
    ... )
    """
    # Set defaults
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()

    # Set up save directory
    if save_dir is None:
        save_dir = os.path.abspath(
            os.path.join(os.getcwd(), f"SALSA_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        )
    os.makedirs(save_dir, exist_ok=True)

    # Save configs
    all_config = {
        **model_config.to_dict(),
        **training_config.to_dict()
    }
    with open(os.path.join(save_dir, 'params.yml'), 'w') as f:
        yaml.dump(all_config, f)

    # Set up data module
    data_module = SalsaDataModule(
        train_data=train_csv,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        val_data=val_csv,
        n_samp=n_samp
    )

    # Need to call setup to get fam_size
    data_module.setup('fit')
    fam_size = data_module.fam_size

    print(f"Training SALSA model")
    print(f"  - Family size: {fam_size}")
    print(f"  - Batch size: {training_config.batch_size} -> {data_module._adjusted_batch_size} (adjusted)")
    print(f"  - Families per batch: {data_module.fams_per_batch}")
    print(f"  - Training samples: {len(data_module.train_dataset)}")
    print(f"  - Epochs: {training_config.n_epochs}")
    print(f"  - Save directory: {save_dir}")

    # Create Lightning module
    lightning_module = SalsaLightningModule(
        model_config=model_config,
        training_config=training_config,
        fam_size=fam_size
    )

    # Set up callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='model_{epoch:02d}_{step:06d}',
        save_top_k=-1,  # Save all checkpoints
        every_n_train_steps=training_config.save_step,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Create trainer
    trainer = L.Trainer(
        default_root_dir=save_dir,
        max_epochs=training_config.n_epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        precision=training_config.precision,
        gradient_clip_val=training_config.gradient_clip_val,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=50,
    )

    # Train
    trainer.fit(lightning_module, data_module)

    # Save final model weights in original format for compatibility
    final_weights_path = os.path.join(save_dir, 'model_trained_model.pt')
    import torch
    torch.save(lightning_module.model.state_dict(), final_weights_path)
    print(f"Saved final model to {final_weights_path}")

    return lightning_module, trainer
