"""
PyTorch Datasets for SALSA.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset

from salsa_lit.utils.io import read_ds
from salsa_lit.utils.chemistry import get_cansmiles
from salsa_lit.constants import MAX_SMI_LEN
from salsa_lit.data.tokenizer import SMILESTokenizer


# Accepted spellings of the identifier column, in order of preference.
ID_COLUMNS = ('ID', 'Id', 'id')


class InferenceDataset(Dataset):
    """
    Dataset for running inference on a set of SMILES.

    Parameters
    ----------
    ds : str or pd.DataFrame
        Dataframe of SMILES or path to a parquet or CSV file.
        Must have a 'SMILES' column.
    tokenizer : SMILESTokenizer, optional
        Tokenizer for SMILES conversion. Created if not provided.
    filter_ : bool
        Whether to filter invalid/long SMILES
    canonicalize : bool
        Whether to canonicalize the SMILES. Pass False when the caller has
        already done it, so the work is not repeated.

    Attributes
    ----------
    dataframe : pd.DataFrame
        Processed dataframe with SMILES
    tokenizer : SMILESTokenizer
        Tokenizer instance
    """

    def __init__(self, ds, tokenizer: SMILESTokenizer = None, filter_: bool = False,
                 canonicalize: bool = True):
        super().__init__()

        self.tokenizer = tokenizer or SMILESTokenizer()

        df_in = read_ds(ds)

        # Reset index and track original positions
        df = df_in.reset_index(drop=True, inplace=False)
        df['Origin_id'] = df.index

        # Normalise the identifier column's spelling, if one was supplied
        if 'ID' not in df.columns:
            for alias in ID_COLUMNS[1:]:
                if alias in df.columns:
                    df = df.rename(columns={alias: 'ID'})
                    break

        # Canonicalize SMILES
        if canonicalize:
            df['SMILES'] = df["SMILES"].apply(lambda x: get_cansmiles(x))

        # Remove empty SMILES
        df = df[df['SMILES'].str.len() > 0]

        # Filter if requested
        if filter_:
            df = df.drop_duplicates('SMILES', keep='first')
            df = df[df["SMILES"].apply(lambda x: '.' not in x)].copy()
            df = df[df["SMILES"].apply(lambda x: len(x) <= MAX_SMI_LEN)].copy()
            df = df.reset_index(drop=True)

        # Inject ID column if missing
        if 'ID' not in df.columns:
            df['ID'] = df['Origin_id']

        # Reorder columns
        other_cols = [x for x in df.columns.values if x not in ['Origin_id', 'SMILES', 'ID']]
        cols = ['ID', 'Origin_id', 'SMILES'] + other_cols

        self.dataframe = df[cols].copy()
        self._smiles = self.dataframe["SMILES"].tolist()
        self._ids = self.dataframe["ID"].tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Return the requested item in SALSA ready format.

        Parameters
        ----------
        idx : int
            Index of the sample

        Returns
        -------
        data : dict
            Dictionary with keys:
            - 'seq': torch.Tensor of token indices
            - 'smiles': str, the SMILES string
            - 'id': the supplied identifier, or the input row position
            - 'pad_mask': padding mask
            - 'avg_mask': averaging mask
            - 'out_mask': output mask
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smi = self._smiles[idx]
        vec = self.tokenizer.encode(smi)
        pad_mask, avg_mask, out_mask = self.tokenizer.get_masks(vec)

        return {
            'seq': vec,
            'smiles': smi,
            'id': self._ids[idx],
            'pad_mask': pad_mask,
            'avg_mask': avg_mask,
            'out_mask': out_mask
        }


class TrainingDataset(Dataset):
    """
    Dataset for training SALSA with mutated molecule families.

    The dataset must follow the SALSA mutation format with columns:
    - SMILES: molecule SMILES string
    - Is_anc: boolean indicating if molecule is an anchor
    - Anc_id: unique identifier for the anchor family

    Parameters
    ----------
    ds : str or pd.DataFrame
        Mutated dataset path or DataFrame
    tokenizer : SMILESTokenizer, optional
        Tokenizer for SMILES conversion. Created if not provided.
    n_samp : int, optional
        Number of anchor families to sample (for debugging)

    Attributes
    ----------
    fam_size : int
        Number of molecules per family (anchor + mutants)
    anc_ids : list
        List of anchor IDs for all samples
    anc_id_set : list
        Unique anchor IDs

    Notes
    -----
    The processed DataFrame is not kept as an attribute. Its columns are
    flattened into lists/arrays at construction so that __getitem__ avoids
    per-sample pandas row indexing, and so that dataloader workers do not
    each inherit a copy of a 42M-row frame.
    """

    def __init__(self, ds, tokenizer: SMILESTokenizer = None, n_samp: int = None):
        super().__init__()

        self.tokenizer = tokenizer or SMILESTokenizer()
        df, self.fam_size = self._process_data(ds, n_samp=n_samp)

        self.anc_ids = df["Anc_id"].tolist()
        self.anc_id_set = df["Anc_id"].unique().tolist()

        self._ids = df["ID"].to_numpy()
        self._smiles = df["SMILES"].tolist()
        self._is_anc = df["Is_anc"].to_numpy()
        self._anc_id = df["Anc_id"].to_numpy()
        self._fam_id = df["Fam_id"].to_numpy()

    def _process_data(self, ds, n_samp=None):
        """
        Read and process SALSA training dataset.

        Parameters
        ----------
        ds : str or pd.DataFrame
            Training data
        n_samp : int, optional
            Number of anchor families to sample

        Returns
        -------
        df : pd.DataFrame
            Processed dataframe
        fam_size : int
            Family size (anchor + mutants)
        """
        df = read_ds(ds)

        # Handle column name variations
        df.rename({'Anc_idx': 'Anc_id'}, axis=1, errors='ignore', inplace=True)
        df.rename({'Smiles': 'SMILES'}, axis=1, errors='ignore', inplace=True)

        # Get family size from first anchor
        first_anc_id = df["Anc_id"].values[0]
        fam_size = len(df[df["Anc_id"] == first_anc_id])

        # Filter out families with SMILES too long
        big_mutset_ids = set(df[df["SMILES"].str.len() > MAX_SMI_LEN]["Anc_id"])
        df = df[~(df["Anc_id"].isin(big_mutset_ids))].copy()

        # Handle sampling if requested
        if n_samp:
            anc_ids = df["Anc_id"].unique().tolist()
            anc_ids_samp = random.sample(anc_ids, min(n_samp, len(anc_ids)))
            df = df[df["Anc_id"].isin(anc_ids_samp)]

        df.reset_index(inplace=True, drop=True)

        # Create new ID column
        df['ID'] = df.index

        # Recompute anchor IDs and family IDs to be contiguous
        num_ancs = np.sum(df["Is_anc"].astype(bool))
        tot_mols = num_ancs * fam_size

        # Create new anchor IDs: [0,0,0,0,0,0,6,6,6,6,6,6,12,12,12,12,12,12,...]
        new_anc_ids = [i for i in range(0, tot_mols, fam_size) for _ in range(fam_size)]
        df['Anc_id'] = new_anc_ids

        # Create family IDs: [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,...]
        fam_ids = [i for i in range(num_ancs) for _ in range(fam_size)]
        df['Fam_id'] = fam_ids

        df = df[['ID', 'SMILES', 'Is_anc', 'Anc_id', 'Fam_id']]

        return df, fam_size

    def __len__(self):
        return len(self._smiles)

    def __getitem__(self, idx):
        """
        Return the requested item in SALSA ready format.

        Parameters
        ----------
        idx : int
            Index of the sample

        Returns
        -------
        data : dict
            Dictionary with keys:
            - 'seq': torch.Tensor of token indices
            - 'smiles': str, the SMILES string
            - 'is_anc': bool, whether this is an anchor
            - 'anc_id': int, anchor ID
            - 'fam_id': int, family ID
            - 'pad_mask': padding mask
            - 'avg_mask': averaging mask
            - 'out_mask': output mask
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smi = self._smiles[idx]
        vec = self.tokenizer.encode(smi)
        pad_mask, avg_mask, out_mask = self.tokenizer.get_masks(vec)

        return {
            'seq': vec,
            'smiles': smi,
            'is_anc': self._is_anc[idx],
            'anc_id': self._anc_id[idx],
            'fam_id': self._fam_id[idx],
            'pad_mask': pad_mask,
            'avg_mask': avg_mask,
            'out_mask': out_mask,
        }
