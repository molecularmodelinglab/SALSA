"""
SALSA - Semantically-Aware Latent Space Autoencoder

A transformer-based model for molecular embeddings using contrastive learning.
"""

__version__ = '2.0.0'

# Config
from .config import ModelConfig, TrainingConfig, DataConfig, SuperMutantConfig

# Models
from .models.salsa_net import SalsaNet
from .lightning.salsa_module import SalsaLightningModule
from .lightning.data_module import SalsaDataModule

# Data
from .data.tokenizer import SMILESTokenizer
from .data.datasets import InferenceDataset, TrainingDataset
from .data.samplers import RandomFamSampler

# API
from .api.train import cook_salsa
from .api.inference import serve_salsa, load_model
from .api.mutate import mutate_dataset
from .api.supermutate import supermutate_dataset, prepare_supermutant_data

__all__ = [
    # Config
    'ModelConfig', 'TrainingConfig', 'DataConfig', 'SuperMutantConfig',
    # Models
    'SalsaNet', 'SalsaLightningModule', 'SalsaDataModule',
    # Data
    'SMILESTokenizer', 'InferenceDataset', 'TrainingDataset', 'RandomFamSampler',
    # API
    'cook_salsa', 'serve_salsa', 'load_model', 'mutate_dataset',
    'supermutate_dataset', 'prepare_supermutant_data',
]
