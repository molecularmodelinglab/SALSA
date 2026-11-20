"""
Data configuration for SALSA.
"""

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class DataConfig:
    """
    Configuration for data processing and mutation.

    Parameters
    ----------
    n_muts : int
        Number of mutants per anchor
    mahala_thresh : float
        Mahalanobis distance threshold for property filtering.
        Use None for no filtering, or a float value for filtering.
    actions : List[Literal['Add', 'Replace', 'Remove']]
        Mutation action types to use
    weight : Literal['uniform', 'chembl']
        Weighting scheme for mutation sampling
    """
    n_muts: int = 5
    mahala_thresh: float = 6.0
    actions: List[str] = field(default_factory=lambda: ['Add', 'Replace', 'Remove'])
    weight: Literal['uniform', 'chembl'] = 'chembl'

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'n_muts': self.n_muts,
            'mahala_thresh': self.mahala_thresh,
            'actions': self.actions,
            'weight': self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SuperMutantConfig:
    """
    Configuration for supermutant construction (evaluation data).

    Supermutants are chains of successive single edits from an anchor, so
    the compound at depth n is n graph edits away. There is no n_muts here:
    exactly one mutation is drawn per step, by construction.

    Parameters
    ----------
    depth : int
        Maximum GED depth. Each surviving anchor yields one compound at each
        depth 1..depth.
    n_branches : int
        Independent chains grown per anchor.
    mahala_thresh : float
        Mahalanobis distance threshold, applied at every step against that
        step's own source compound. Use None for no filtering.
    actions : List[Literal['Add', 'Replace', 'Remove']]
        Mutation action types to use.
    weight : Literal['uniform', 'chembl']
        Weighting scheme for mutation sampling.
    """
    depth: int = 5
    n_branches: int = 1
    mahala_thresh: float = 6.0
    actions: List[str] = field(default_factory=lambda: ['Add', 'Replace', 'Remove'])
    weight: Literal['uniform', 'chembl'] = 'chembl'

    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'depth': self.depth,
            'n_branches': self.n_branches,
            'mahala_thresh': self.mahala_thresh,
            'actions': self.actions,
            'weight': self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict):
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
