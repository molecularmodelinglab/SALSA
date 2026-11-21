"""
Model configuration for SalsaNet.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """
    Configuration for SalsaNet model architecture.

    Parameters
    ----------
    dim_emb : int
        The latent dimension of the embedder
    dim_salsa : int
        The dimension of the latent space between embedder and decoder.
        This is also the dimension of the SALSA descriptors.
    n_heads : int
        Number of attention heads in transformer layers
    L_enc : int
        Number of encoder layers
    L_dec : int
        Number of decoder layers
    dim_ff : int
        Dimension of the feedforward network in transformer layers
    dropout : float
        Dropout probability
    pool_func : Literal['avg', 'max', 'set']
        Pooling function to aggregate encoded vectors
    actv : Literal['relu', 'gelu']
        Activation function in transformer layers
    layer_norm_eps : float
        Layer normalization epsilon
    proj_head : bool
        Whether to use projection head (currently not recommended)
    """
    dim_emb: int = 512
    dim_salsa: int = 16
    n_heads: int = 8
    L_enc: int = 8
    L_dec: int = 8
    dim_ff: int = 2048
    dropout: float = 0.1
    pool_func: Literal['avg', 'max', 'set'] = 'avg'
    actv: Literal['relu', 'gelu'] = 'relu'
    layer_norm_eps: float = 1e-5
    proj_head: bool = False

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'dim_emb': self.dim_emb,
            'dim_salsa': self.dim_salsa,
            'n_heads': self.n_heads,
            'L_enc': self.L_enc,
            'L_dec': self.L_dec,
            'dim_ff': self.dim_ff,
            'dropout': self.dropout,
            'pool_func': self.pool_func,
            'actv': self.actv,
            'layer_norm_eps': self.layer_norm_eps,
            'proj_head': self.proj_head,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
