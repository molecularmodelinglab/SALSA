from typing import Iterator, List

import numpy as np
import random
import pandas as pd

import torch
from torch.utils.data import Dataset, RandomSampler, Sampler

from salsa.utils import get_cansmiles, read_ds
from salsa.constants import VOCAB, S_TOKEN, E_TOKEN, P_TOKEN, UNK_TOKEN, TOKENS, MAX_SMI_LEN


# Inference dataset #
class InferenceDataset(Dataset):
    """
    Dataset for a set of chemicals that need to be run through the model, but not used for training

    Parameters
    ----------
    ds: str or pd.DataFrame object
        dataframe of smiles to conduct inference on
    filter_: bool = False
        whether to filter the dataset (remove canonicalize SMILES and remove SMILES that are too big)

    Attributes
    ----------
    dataframe: pd.DataFrame
        the dataframe containing the SMILES to pe predicted on
    """

    def __init__(self, ds, filter_=False):
        super().__init__()

        df_in = read_ds(ds)

        # used to see what was filtered.
        df = df_in.reset_index(drop=True, inplace=False)
        df['Origin_id'] = df.index

        df['SMILES'] = df["SMILES"].apply(lambda x: get_cansmiles(x))
        df = df[df['SMILES'].str.len() > 0]  # remove empty SMILES

        # filter is wanted
        if filter_:
            df.drop_duplicates('SMILES', keep='first', inplace=True)
            df = df[df["SMILES"].apply(lambda x: '.' not in x)].copy()
            df = df["SMILES"].apply(lambda x: len(x) <= MAX_SMI_LEN)
            df.reset_index(drop=True, inplace=True)

        # inject Id column if missing
        df['ID'] = df.index

        # move columns to front
        other_cols = [x for x in df.columns.values if x not in ['Origin_id', 'SMILES', 'ID']]
        cols = ['ID', 'Origin_id', 'SMILES'] + other_cols

        self.dataframe = df[cols].copy()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Return the requested item in SALSA ready format from the dataset at index `idx`

        Parameters
        ----------
        idx: (list, Tensor, ndarray) of ints
            the numerical index of the data that need to be returned

        Returns
        -------
        data: dict
            data return as dict with the following keys:
            `seq`: torch.Tensor embedded SMILE into vector for the SALSA model to take as input
            `smiles`: str the SMILES string
            `pad_mask`: torch.Tensor boolean mask to mask out padding tokens
            `avg_mask`: torch.Tensor boolean mask to mask out all padding, start and end token
            `out_mask`: torch.Tensor boolean mask to mask out start and end token
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        smi = self.dataframe.iloc[idx]["SMILES"]
        vec = _get_vec(_replace_unks(_do_BrCl_singles(smi)))
        masks = _get_masks_from_vec(vec)

        return {
            'seq': vec,
            'smiles': smi,
            'pad_mask': masks[0],
            'avg_mask': masks[1],
            'out_mask': masks[2]
        }


# Training dataset #
class TrainingDataset(Dataset):
    """
    Dataset object used for training SALSA model with a mutated SALSA dataset

    Parameters
    ----------
    ds: str, pd.DataFrame
        the dataset to be used for training, either path or loaded DataFrame
    n_samp: int = None
        randomly sample only a portion of the dataset to use for training (as whole number of samples)
        if left as None, will use the entire dataset for training

    Attributes
    ----------
    dataframe: pd.DataFrame
        the training data as a dataframe object
    fam_size: int
        the number of mutants each anchor molecule has
    anc_ids: list[int]
        the list of anchor ids for all elements in the dataset
    anc_id_set: list[int]
        the unique list of anchor ids for all elements in the dataset

    Notes
    -----
    The dataset used here must be one created by the SALSA data mutation pipeline otherwise it will not work
    """
    def __init__(self, ds, n_samp=None):
        super().__init__()

        self.dataframe, self.fam_size = self.process_data(ds, n_samp=n_samp)
        self.anc_ids = self.dataframe["Anc_id"].tolist()
        self.anc_id_set = self.dataframe["Anc_id"].unique().tolist()

    @staticmethod
    def process_data(ds, n_samp=None):
        """
        Reads in and processes SALSA training dataset
        Parameters
        ----------
        ds: str, pd.DataFrame
            the dataset to be used for training, either path or loaded DataFrame
        n_samp: int = None
            randomly sample only a portion of the dataset to use for training (as whole number of samples)
            if left as None, will use the entire dataset for training

        Returns
        -------
        df: pd.DataFrame
            processed data as a dataframe object
        fam_size: int
            the number of mutants per anchor

        Notes
        -----
        The dataset used here must be one created by the SALSA data mutation pipeline otherwise it will not work
        for training and throw some KeyError when being processed
        """
        df = read_ds(ds)

        df.rename({'Anc_idx': 'Anc_id'}, axis=1, errors='ignore', inplace=True)

        # get fam size
        first_anc_id = df["Anc_id"].values[0]
        fam_size = len(df[df["Anc_id"] == first_anc_id])

        # filter out smiles with length more than max
        big_mutset_ids = set(df[df["SMILES"].str.len() > MAX_SMI_LEN]["Anc_id"])
        df = df[~(df["Anc_id"].isin(big_mutset_ids))].copy()

        # handle sampling if needed
        if n_samp:
            anc_ids = df["Anc_id"].unique().tolist()
            anc_ids_samp = random.sample(anc_ids, n_samp)
            df = df[df["Anc_id"].isin(anc_ids_samp)]

        df.reset_index(inplace=True, drop=True)

        # make new id col
        df['ID'] = df.index

        # Redo anc_ids and fam_ids
        num_ancs = np.sum(df["Is_anc"].astype(bool))
        tot_mols = num_ancs * fam_size

        # [0,0,0,0,0,0,6,6,6,6,6,6,12,12,12,12,12,12,...]
        new_anc_ids = [i for i in range(0, tot_mols, fam_size) for _ in range(fam_size)]
        df['Anc_id'] = new_anc_ids

        # [0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,...]
        fam_ids = [i for i in range(num_ancs) for _ in range(fam_size)]
        df['Fam_id'] = fam_ids

        df = df[['ID', 'SMILES', 'Is_anc', 'Anc_id', 'Fam_id']]

        return df, fam_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Return the requested item in SALSA ready format from the dataset at index `idx`

        Parameters
        ----------
        idx: (list, Tensor, ndarray) of ints
            the numerical index of the data that need to be returned

        Returns
        -------
        data: dict
            data return as dict with the following keys:
            `seq`: torch.Tensor embedded SMILE into vector for the SALSA model to take as input
            `smiles`: str the SMILES string
            `is_anc`: bool indicating if compound is an anchor compound
            `anc_id`: int the anchor Id of the compound
            `fam_id`: int the family Id of the compound
            `pad_mask`: torch.Tensor boolean mask to mask out padding tokens
            `avg_mask`: torch.Tensor boolean mask to mask out all padding, start and end token
            `out_mask`: torch.Tensor boolean mask to mask out start and end token
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        _id, smi, is_anc, anc_id, fam_id = self.dataframe.iloc[idx].values
        vec = convert_smi_to_vec(smi)
        masks = _get_masks_from_vec(vec)

        return {
            'seq': vec,
            'smiles': smi,
            'is_anc': is_anc,
            'anc_id': anc_id,
            'fam_id': fam_id,
            'pad_mask': masks[0],
            'avg_mask': masks[1],
            'out_mask': masks[2],
        }


# SMILES vectorization #
def _remove_extra_tokens(smi):
    """
    Removes the padding, start and end token from a SMILES string

    Parameters
    ----------
    smi: str
        SMILES string to process

    Returns
    -------
    smi: str
        SMILES string with padding, start and end tokens removed
    """
    smi = smi.replace(P_TOKEN, '')
    smi = smi.replace(S_TOKEN, '')
    smi = smi.replace(E_TOKEN, '')
    return smi


def _undo_BrCl_singles(smi):
    """
    Convert the Br and Cl tokens back into two-letter format

    Parameters
    ----------
    smi: str
        SMILES string to process

    Returns
    -------
    smi: str
        SMILES string with converted Br and Cl tokens

    """
    smi = smi.replace('R', 'Br')
    smi = smi.replace('L', 'Cl')
    return smi


def _do_BrCl_singles(smi):
    """
    Convert the Br and Cl tokens into one-letter format

    Parameters
    ----------
    smi: str
        SMILES string to process

    Returns
    -------
    smi: str
        SMILES string with converted Br and Cl tokens

    Notes
    -----
    R is for Br and L is for Cl
    """
    smi = smi.replace('Br', 'R')
    smi = smi.replace('Cl', 'L')
    return smi


def _replace_unks(smi):
    """
    Replace unknown tokens with the `UNK_TOKEN` for `constants.py`

    Parameters
    ----------
    smi: str
        SMILES string to process

    Returns
    -------
    smi: str
        SMILES string with unknown tokens converted
    """
    smi = ''.join([char if char in VOCAB else UNK_TOKEN for char in smi])
    return smi


def _get_vec(smi):
    """
    Converts a SMILES into a vector representation for input in the Transformer encoder

    Parameters
    ----------
    smi: str
        SMILES string to vectorize

    Returns
    -------
    vec: torch.Tensor
        vectorized SMILES string

    """
    padding = ''.join([P_TOKEN for _ in range(MAX_SMI_LEN - len(smi))])
    smi = S_TOKEN + smi + E_TOKEN + padding
    vec = torch.zeros(len(smi)).long()
    for i in range(len(smi)):
        vec[i] = TOKENS.index(smi[i])
    return vec


def _get_masks_from_vec(vec):
    """
    Generated the required pads for Transformer function

    Parameters
    ----------
    vec: torch.Tensor
        vectorized SMILES string

    Returns
    -------
    pad_mask: torch.Tensor
        boolean mask to mask out padding tokens
    avg_mask: torch.Tensor
        boolean mask to mask out all padding, start and end token
    out_mask: torch.Tensor
        boolean mask to mask out start and end token

    """
    p_idx = TOKENS.index(P_TOKEN)
    s_idx = TOKENS.index(S_TOKEN)
    e_idx = TOKENS.index(E_TOKEN)

    # pad mask: masks pad tokens
    pad_mask = (vec == p_idx)

    # avg mask: masks pad,s,e tokens
    avg_mask = ((vec == p_idx) | (vec == e_idx) | (vec == s_idx)).float()
    avg_mask = torch.ones_like(avg_mask) - avg_mask

    # sup (superfluous) mask: masks s,e tokens
    # called "out_mask" in other places in the code
    sup_mask = torch.ones(len(TOKENS))
    idx = torch.tensor([s_idx, p_idx])
    sup_mask = torch.zeros_like(sup_mask).scatter_(0, idx, sup_mask)
    sup_mask = sup_mask.unsqueeze(0)
    return pad_mask, avg_mask, sup_mask


def convert_vec_to_smi(vec, snip=False):
    """
    Convert one (or many) vectorized SMILES string back into a SMILES string(s)

    Parameters
    ----------
    vec: torch.Tensor
        the vectorized SMILES string
    snip: bool = False
        remove the start and end tokens

    Returns
    -------
    smiles: list[str]
        all the SMILES strings decoded from the passed tensor

    Notes
    -----
    Can function on 2D Tensor (Tensor with multiple SMILES embedded, one per row)
    """

    smi_arr = np.array(list(TOKENS))[vec.cpu().detach().numpy()]
    smi_list = [''.join(arr) for arr in smi_arr]
    smi_list = [_undo_BrCl_singles(smi) for smi in smi_list]
    if snip:
        smi_list = [_remove_extra_tokens(smi) for smi in smi_list]
    return smi_list


def convert_smi_to_vec(smi):
    """
    Function to be used for converting a generic unprocessed SMILES string to vectorized format

    Parameters
    ----------
    smi: str
        SMILES string to convert

    Returns
    -------
    vec: torch.Tensor
        the vectorized SMILES string

    Notes
    -----
    This function will handle Br/Cl and UNK tokens for you, so when you need to tokenize a SMILES, use this function
    """
    vec = _get_vec(_replace_unks(_do_BrCl_singles(smi)))
    return vec


# Samplers #
# inspired from pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html#BatchSampler
class RandomFamSampler(Sampler[List[int]]):
    """
    Sampler to randomly sample family sets from data and assign them to the same batch.
    This is needed for the contrastive part of learning to take place

    Should be built after TrainingDataset generation

    Parameters
    ----------
    anc_ids: list[ints]
        The list of all the anchor ids for each element in that dataset
        e.g. [0,0,0,0,0,0,6,6,6,6,6,6,12,12,12, ...]
    batch_size: int
        the size of the mini-batch used for training
    drop_last: bool = True
        if True, the sampler will drop the last batch if its size is less than `batch_size`

    Attributes
    ----------
    anc_ids: list[int]
        the list of all the anchor ids for each element in the dataset
    batch_size: int
        the size of the mini-batch used for training
    drop_last: bool
        whether to drop the last batch if its size is less
    len_ds: int
        number of elements in the dataset
    anc_id_set: list[int]
        list of unique anchor ids in dataset
    anc_sampler: torch.utils.data.sampler.Sampler
        a random sampler to sample anchor ids
    fam_size: int
        the number of mutants each anchor has
    fams_per_batch: int
        the number of families per mini-batch (number of anchors in batch)
    true_batch_size: int
        the number of raw elements in a mini-batch (anchors and mutants)
    """
    def __init__(self, anc_ids: list, batch_size: int, drop_last: bool):
        super().__init__()
        self.anc_ids = anc_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.len_ds = len(self.anc_ids)

        self.anc_id_set = list(set(anc_ids))
        self.anc_sampler = RandomSampler(self.anc_id_set)

        self.fam_size = np.count_nonzero(np.asarray(self.anc_ids) == self.anc_ids[0])
        self.fams_per_batch = self.batch_size // self.fam_size
        self.true_batch_size = self.fams_per_batch * self.fam_size

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        i = 0

        for index in self.anc_sampler:
            anc_id = self.anc_id_set[index]

            fam_ids = [anc_id + i for i in range(self.fam_size)]
            batch.extend(fam_ids)
            i += 1
            if len(batch) == self.true_batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.len_ds // self.true_batch_size
        else:
            return (self.len_ds + self.true_batch_size - 1) // self.true_batch_size
