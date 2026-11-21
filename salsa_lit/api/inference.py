"""
High-level inference API for SALSA.
"""

from typing import Union, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from salsa_lit.config import ModelConfig
from salsa_lit.constants import MAX_SMI_LEN
from salsa_lit.models.salsa_net import SalsaNet
from salsa_lit.lightning.salsa_module import SalsaLightningModule
from salsa_lit.data.tokenizer import SMILESTokenizer
from salsa_lit.data.datasets import InferenceDataset
from salsa_lit.utils.chemistry import get_cansmiles
from salsa_lit.utils.io import read_ds, write_embeddings


def trim_collate(items: List[dict]) -> dict:
    """
    Collate a batch and drop the trailing padding shared by every member.

    Parameters
    ----------
    items : list of dict
        Samples from InferenceDataset

    Returns
    -------
    batch : dict
        Collated batch with 'seq', 'pad_mask' and 'avg_mask' trimmed to the
        longest unpadded sequence in the batch. 'out_mask' is per-token and
        so is left as-is.
    """
    batch = default_collate(items)

    keep = int((~batch['pad_mask']).sum(dim=1).max())
    batch['seq'] = batch['seq'][:, :keep]
    batch['pad_mask'] = batch['pad_mask'][:, :keep]
    batch['avg_mask'] = batch['avg_mask'][:, :keep]

    return batch


def load_model(path: str, map_location: str = None) -> SalsaNet:
    """
    Load a trained SALSA model from checkpoint.

    Parameters
    ----------
    path : str
        Path to model directory or checkpoint file.
        If directory, looks for model_trained_model.pt and params.yml
    map_location : str, optional
        Device to map model to (e.g., 'cpu', 'cuda:0')

    Returns
    -------
    model : SalsaNet
        Loaded model in eval mode
    """
    model = SalsaNet.from_pretrained(path, map_location=map_location)
    model.eval()
    return model


def serve_salsa(
    smiles: Union[str, pd.DataFrame, List[str]],
    model: Union[str, SalsaNet, SalsaLightningModule],
    output_file: str = None,
    batch_size: int = 50,
    device: str = None,
    filter_input: bool = False,
    fmt: str = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate SALSA embeddings for a set of SMILES.

    Parameters
    ----------
    smiles : str or pd.DataFrame or List[str]
        Input SMILES. Can be:
        - Path to a parquet or CSV file with a 'SMILES' column
        - DataFrame with 'SMILES' column
        - List of SMILES strings
    model : str or SalsaNet or SalsaLightningModule
        Trained model. Can be:
        - Path to model directory or checkpoint
        - SalsaNet model object
        - SalsaLightningModule object
    output_file : str, optional
        Path to save embeddings to. If None, results are only returned.
        A '.csv' extension writes CSV, otherwise parquet is used.
    batch_size : int
        Batch size for inference
    device : str, optional
        Device to use ('cpu', 'cuda', 'cuda:0', etc.)
        If None, uses CUDA if available.
    filter_input : bool
        Screen out inputs the model cannot embed (multi-fragment SMILES, or
        SMILES longer than MAX_SMI_LEN).
    fmt : str, optional
        Output format ('parquet' or 'csv') when `output_file` has no
        recognised extension. Defaults to parquet.

    Returns
    -------
    embeddings : np.ndarray
        SALSA embeddings, shape (n_molecules, dim_salsa)
    smiles_list : List[str]
        Corresponding SMILES strings

    Example
    -------
    >>> from salsa_lit import serve_salsa
    >>> embeddings, smiles = serve_salsa(
    ...     ["CCO", "CC(=O)O", "c1ccccc1"],
    ...     model="./my_model/"
    ... )
    >>> print(embeddings.shape)
    (3, 16)
    """
    # Set up device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load model if path provided
    if isinstance(model, str):
        model = load_model(model, map_location=str(device))
    elif isinstance(model, SalsaLightningModule):
        model = model.model

    model.to(device)
    model.eval()

    # Prepare data
    if isinstance(smiles, list):
        df = pd.DataFrame({'SMILES': smiles})
    elif isinstance(smiles, str):
        df = read_ds(smiles, columns=['SMILES'])
    else:
        df = smiles

    # Filter input if requested
    if filter_input:
        df = df.copy()
        df['SMILES'] = df['SMILES'].apply(get_cansmiles)
        unusable = (df['SMILES'].str.contains('.', regex=False)
                    | (df['SMILES'].str.len() > MAX_SMI_LEN)
                    | (df['SMILES'].str.len() == 0))
        if unusable.any():
            print(f"Dropping {unusable.sum()} multi-fragment, over-length or "
                  f"unparseable SMILES")
            df = df[~unusable]

    # Create dataset and loader
    tokenizer = SMILESTokenizer()
    dataset = InferenceDataset(df, tokenizer=tokenizer, filter_=False,
                               canonicalize=not filter_input)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=False, collate_fn=trim_collate)

    # Run inference
    all_embeddings = []
    all_smiles = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating embeddings"):
            # Move tensors to device
            seq = batch['seq'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            avg_mask = batch['avg_mask'].to(device)
            out_mask = batch['out_mask'].to(device)

            # Get embeddings
            salsa_latent, _ = model.embed(seq, pad_mask, avg_mask, out_mask)

            all_embeddings.append(salsa_latent.cpu().numpy())
            all_smiles.extend(batch['smiles'])

    embeddings = np.vstack(all_embeddings)

    # Save if output file specified
    if output_file is not None:
        write_embeddings(output_file, all_smiles, embeddings, fmt=fmt)
        print(f"Saved embeddings to {output_file}")

    return embeddings, all_smiles


def embed_smiles(
    smiles_list: List[str],
    model: Union[str, SalsaNet],
    device: str = None,
) -> np.ndarray:
    """
    Simple function to embed a list of SMILES strings.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to embed
    model : str or SalsaNet
        Trained model or path to model
    device : str, optional
        Device to use

    Returns
    -------
    embeddings : np.ndarray
        Embeddings of shape (n_smiles, dim_salsa)
    """
    embeddings, _ = serve_salsa(smiles_list, model, device=device)
    return embeddings
