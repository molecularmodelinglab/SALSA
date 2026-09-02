"""
High-level inference API for SALSA.
"""

import os
from typing import Union, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from salsa_lit.config import ModelConfig
from salsa_lit.constants import MAX_SMI_LEN
from salsa_lit.models.salsa_net import SalsaNet
from salsa_lit.lightning.salsa_module import SalsaLightningModule
from salsa_lit.data.tokenizer import SMILESTokenizer
from salsa_lit.data.datasets import InferenceDataset
from salsa_lit.utils.chemistry import get_cansmiles


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
) -> Tuple[np.ndarray, List[str]]:
    """
    Generate SALSA embeddings for a set of SMILES.

    Parameters
    ----------
    smiles : str or pd.DataFrame or List[str]
        Input SMILES. Can be:
        - Path to CSV/PARQUET file with 'SMILES' column
        - DataFrame with 'SMILES' column
        - List of SMILES strings
    model : str or SalsaNet or SalsaLightningModule
        Trained model. Can be:
        - Path to model directory or checkpoint
        - SalsaNet model object
        - SalsaLightningModule object
    output_file : str, optional
        Path to save embeddings CSV. If None, results are only returned.
    batch_size : int
        Batch size for inference
    device : str, optional
        Device to use ('cpu', 'cuda', 'cuda:0', etc.)
        If None, uses CUDA if available.
    filter_input : bool
        Screen out inputs the model cannot embed (multi-fragment SMILES, or
        SMILES longer than MAX_SMI_LEN).

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
    (3, 32)
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
        if smiles.endswith('.parquet'):
            df = pd.read_parquet(smiles, columns=['SMILES'])
        else:
            df = pd.read_csv(smiles, usecols=['SMILES'])
    else:
        df = smiles

    # Filter input if requested
    if filter_input:
        df = df.copy()
        df['SMILES'] = df['SMILES'].apply(get_cansmiles)
        unusable = (df['SMILES'].str.contains('.', regex=False)
                    | (df['SMILES'].str.len() > MAX_SMI_LEN))
        if unusable.any():
            print(f"Dropping {unusable.sum()} multi-fragment or over-length SMILES")
            df = df[~unusable]

    # Create dataset and loader
    tokenizer = SMILESTokenizer()
    dataset = InferenceDataset(df, tokenizer=tokenizer, filter_=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

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
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, 'w') as f:
            # Write header
            dim_salsa = embeddings.shape[1]
            header = 'SMILES,' + ','.join([f'dim_{i}' for i in range(dim_salsa)])
            f.write(header + '\n')

            # Write data
            for i, smi in enumerate(all_smiles):
                emb_str = ','.join(map(str, embeddings[i]))
                f.write(f"{smi},{emb_str}\n")

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
