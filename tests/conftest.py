"""Shared fixtures for the salsa_lit test suite."""

import glob
import os

import pytest
import torch

from salsa_lit.config import ModelConfig
from salsa_lit.constants import MAX_SMI_LEN
from salsa_lit.data.tokenizer import SMILESTokenizer
from salsa_lit.models.salsa_net import SalsaNet


MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
)


@pytest.fixture(scope='session')
def tokenizer():
    """A default SMILES tokenizer."""
    return SMILESTokenizer()


@pytest.fixture(scope='session')
def small_model():
    """A tiny deterministic SalsaNet in eval mode."""
    torch.manual_seed(0)
    model = SalsaNet(ModelConfig(dim_emb=32, dim_salsa=8, n_heads=2, L_enc=2,
                                 L_dec=1, dim_ff=64, dropout=0.0))
    model.eval()
    return model


@pytest.fixture
def short_smiles():
    """Valid, uniformly short SMILES."""
    return ['CCO', 'CCCO', 'CC(=O)O', 'c1ccccc1', 'CCN', 'CCCCO']


@pytest.fixture
def mixed_smiles():
    """Valid SMILES with widely varying lengths."""
    return [
        'CCO',
        'COC(=O)c1ccc(C(NC(=O)OCc2ccccc2)C(C)C)cc1',
        'CC',
        'c1ccccc1',
        'CCOc1ccc(cc1)C(=O)NC2CCN(CC2)Cc3ccccc3',
        'O',
        'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
        'CCN',
    ]


@pytest.fixture
def max_len_smiles():
    """A valid SMILES of exactly MAX_SMI_LEN characters."""
    return 'C' * MAX_SMI_LEN


@pytest.fixture
def trained_model_path():
    """Path to a trained checkpoint, preferring a weights-only copy."""
    paths = sorted(glob.glob(os.path.join(MODELS_DIR, '*.ckpt')))
    paths.sort(key=lambda p: not p.endswith('_weights.ckpt'))
    if not paths:
        pytest.skip(f'no trained checkpoint in {MODELS_DIR}')
    return paths[0]
