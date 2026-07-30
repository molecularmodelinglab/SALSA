"""
Constants for SALSA model.

This file contains values that are unlikely to change and are shared across
the codebase. For configurable parameters, see the config module.
"""

import os
import numpy as np
import pandas as pd


# ============================================================================
# Tokenization Constants
# ============================================================================

# Base vocabulary for SMILES tokenization
VOCAB = '#%()+-0123456789<=>BCFHILNOPRSX[]cnosp$'

# Special tokens
S_TOKEN = '<'   # start token
E_TOKEN = '>'   # end token
P_TOKEN = 'X'   # pad token
UNK_TOKEN = '$' # unknown token

# Sorted token string (computed from vocab + special tokens)
_tokens = list(set(VOCAB + S_TOKEN + E_TOKEN + P_TOKEN + UNK_TOKEN))
TOKENS = ''.join(list(np.sort(_tokens)))

# Number of tokens in vocabulary
N_TOKENS = len(TOKENS)  # 39


# ============================================================================
# Sequence Length Constants
# ============================================================================

MAX_ANC_LEN = 120   # Maximum anchor SMILES length for filtering
MAX_MUT_LEN = 150   # Maximum mutant SMILES length
MAX_SMI_LEN = 150   # Maximum SMILES string length (after tokenization)
MAX_VEC_LEN = 152   # Maximum vector length (max_smi_len + 2 for start/end)
MAX_POS_LEN = 5000  # Maximum positional encoding length


# ============================================================================
# Mutation Constants
# ============================================================================

# Valid mutation actions
ACTIONS = ['Add', 'Replace', 'Remove']

# Default number of mutants per anchor
N_MUTS = 5

# Default Mahalanobis distance threshold
MAHALA_THRESH = 6.0

# Default weighting scheme
WEIGHT = 'chembl'

# Valid atom types for mutations
ATOMS = {'C', 'O', 'N', 'S', 'Br', 'I', 'F', 'Cl', 'P', 'B'}

# Atom counts from ChEMBL database (for weighted sampling)
ATOM_TO_CNT = {
    "C": 3806500,
    "O": 571807,
    "N": 562208,
    "S": 73267,
    "Br": 8730,
    "I": 996,
    "F": 66513,
    "Cl": 41958,
    "P": 3018,
    "B": 1
}

# Inverse counts for removal weighting
ATOM_TO_CNT_INV = {atom: 1 / cnt for atom, cnt in ATOM_TO_CNT.items()}


# ============================================================================
# Loss Function Names
# ============================================================================

NAMED_LOSSES = ['Recon', 'SupCon']


# ============================================================================
# Data File Paths (relative to package)
# ============================================================================

_PACK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Path to inverse covariance matrix for Mahalanobis distance
INV_COV_CHEMBL_PATH = os.path.join(_PACK_DIR, 'data/config/pseudo_chembl_inv_cov_20230510.txt')

# Path to property names CSV
PROP_NAMES_PATH = os.path.join(_PACK_DIR, 'data/config/selected_props_20230510.csv')


def load_inv_cov_chembl():
    """Load the inverse covariance matrix for Mahalanobis distance calculation."""
    if os.path.exists(INV_COV_CHEMBL_PATH):
        return np.loadtxt(INV_COV_CHEMBL_PATH)
    else:
        return None


def load_prop_names():
    """Load the property names for Mahalanobis distance calculation."""
    if os.path.exists(PROP_NAMES_PATH):
        prop_csv = pd.read_csv(PROP_NAMES_PATH)
        return prop_csv['prop_name'].values.tolist()
    else:
        return []


# Lazy-loaded globals (loaded on first access to avoid import-time side effects)
_INV_COV_CHEMBL = None
_PROP_NAMES = None


def get_inv_cov_chembl():
    """Get inverse covariance matrix, loading if necessary."""
    global _INV_COV_CHEMBL
    if _INV_COV_CHEMBL is None:
        _INV_COV_CHEMBL = load_inv_cov_chembl()
    return _INV_COV_CHEMBL


def get_prop_names():
    """Get property names list, loading if necessary."""
    global _PROP_NAMES
    if _PROP_NAMES is None:
        _PROP_NAMES = load_prop_names()
    return _PROP_NAMES
