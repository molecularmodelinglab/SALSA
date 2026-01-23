"""
High-level mutation API for SALSA dataset creation.
"""

import os
from typing import Union, List, Tuple
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from salsa_lit.config import DataConfig
from salsa_lit.utils.io import read_ds, timestamp
from salsa_lit.data.mutation import mutate_data, filter_source_ds


def mutate_dataset(
    smiles_file: Union[str, pd.DataFrame],
    output_file: str = None,
    data_config: DataConfig = None,
    n_samp: int = -1,
    use_tqdm: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Create a SALSA training dataset by mutating molecules.

    Parameters
    ----------
    smiles_file : str or pd.DataFrame
        Input SMILES data. Must have 'SMILES' and 'ID' columns.
    output_file : str, optional
        Output CSV path. If None, generates timestamped name.
    data_config : DataConfig, optional
        Data configuration with mutation parameters.
    n_samp : int
        Number of molecules to sample (-1 for all)
    use_tqdm : bool
        Show progress bar

    Returns
    -------
    df : pd.DataFrame
        Mutated dataset
    output_path : str
        Path to saved file

    Example
    -------
    >>> from salsa_lit import mutate_dataset, DataConfig
    >>> df, path = mutate_dataset(
    ...     "molecules.csv",
    ...     output_file="training_data.csv",
    ...     data_config=DataConfig(n_muts=10, mahala_thresh=5.0)
    ... )
    """
    # Use defaults
    data_config = data_config or DataConfig()

    # Call the core mutation function
    df = mutate_data(
        ds=smiles_file,
        out_name=output_file,
        out_dir=None if output_file else os.getcwd(),
        n_samp=n_samp,
        num_muts=data_config.n_muts,
        actions=data_config.actions,
        mahala_thresh=data_config.mahala_thresh,
        weight=data_config.weight,
        use_tqdm=use_tqdm
    )

    # Determine output path
    if output_file is None:
        output_file = f'mutants_{timestamp()}_{n_samp}-ancs_{data_config.n_muts}-muts.csv'

    return df, output_file


def prepare_training_data(
    smiles_file: Union[str, pd.DataFrame],
    output_dir: str = None,
    data_config: DataConfig = None,
    n_samp: int = -1,
    filter_data: bool = True,
    max_len: int = 100,
    max_atoms: int = 35,
) -> Tuple[pd.DataFrame, str]:
    """
    Full pipeline to prepare training data from raw SMILES.

    This function:
    1. Loads and filters the source SMILES
    2. Generates mutations for each molecule
    3. Saves the result

    Parameters
    ----------
    smiles_file : str or pd.DataFrame
        Input SMILES data
    output_dir : str, optional
        Output directory
    data_config : DataConfig, optional
        Configuration for mutation
    n_samp : int
        Number of molecules to sample
    filter_data : bool
        Whether to filter source data first
    max_len : int
        Maximum SMILES length for filtering
    max_atoms : int
        Maximum number of atoms for filtering

    Returns
    -------
    df : pd.DataFrame
        Mutated dataset
    output_path : str
        Path to saved file
    """
    data_config = data_config or DataConfig()

    # Load data
    df = read_ds(smiles_file)

    # Filter if requested
    if filter_data:
        df = filter_source_ds(df, max_len=max_len, max_atoms=max_atoms)
        print(f"Filtered to {len(df)} molecules")

    # Set up output
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    ts = timestamp()
    output_name = f'mutants_{ts}_{len(df) if n_samp < 0 else n_samp}-ancs_{data_config.n_muts}-muts_{data_config.mahala_thresh}-mthresh_{data_config.weight}-weight.csv'
    output_path = os.path.join(output_dir, output_name)

    # Run mutation
    result_df = mutate_data(
        ds=df,
        out_name=output_name,
        out_dir=output_dir,
        n_samp=n_samp,
        num_muts=data_config.n_muts,
        actions=data_config.actions,
        mahala_thresh=data_config.mahala_thresh,
        weight=data_config.weight,
        use_tqdm=True
    )

    print(f"Created training dataset with {len(result_df)} samples")
    print(f"Saved to {output_path}")

    return result_df, output_path
