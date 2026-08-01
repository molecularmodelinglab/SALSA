"""
Training configuration for SALSA.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union


@dataclass
class TrainingConfig:
    """
    Configuration for training procedure.

    Parameters
    ----------
    batch_size : int
        Size of mini-batches (will be adjusted to fit family size)
    lr : float
        Learning rate for Adam optimizer
    n_epochs : int
        Number of training epochs
    losses : List[Literal['Recon', 'SupCon', 'Aligniform']]
        Which loss functions to use
    save_step : int
        Save checkpoint every N steps
    accelerator : str
        PyTorch Lightning accelerator ('cpu', 'gpu', 'auto')
    devices : int or List[int] or str
        Number of devices or specific device IDs
    precision : Literal['32', '16-mixed', 'bf16-mixed']
        Training precision. Defaults to 'bf16-mixed' for speed.
    gradient_clip_val : Optional[float]
        Gradient clipping value
    accumulate_grad_batches : int
        Gradient accumulation steps
    num_workers : int
        Number of dataloader workers
    """
    batch_size: int = 50
    lr: float = 0.00001
    n_epochs: int = 1
    losses: List[str] = field(default_factory=lambda: ['Recon', 'SupCon'])
    save_step: int = 10000
    accelerator: str = 'auto'
    devices: Union[int, List[int], str] = 1
    precision: str = 'bf16-mixed'
    gradient_clip_val: Optional[float] = None
    accumulate_grad_batches: int = 1
    num_workers: int = 4

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'batch_size': self.batch_size,
            'lr': self.lr,
            'n_epochs': self.n_epochs,
            'losses': self.losses,
            'save_step': self.save_step,
            'accelerator': self.accelerator,
            'devices': self.devices,
            'precision': self.precision,
            'gradient_clip_val': self.gradient_clip_val,
            'accumulate_grad_batches': self.accumulate_grad_batches,
            'num_workers': self.num_workers,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
