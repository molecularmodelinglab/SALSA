"""
High-level supermutant API for SALSA evaluation datasets.
"""

import os
from typing import Tuple, Union

import pandas as pd

from salsa_lit.config import SuperMutantConfig
from salsa_lit.data.mutation import filter_source_ds
from salsa_lit.data.supermutation import supermutate_data
from salsa_lit.utils.io import DEFAULT_FORMAT, read_ds, timestamp, with_ext


def supermutate_dataset(
    smiles_file: Union[str, pd.DataFrame],
    output_file: str = None,
    config: SuperMutantConfig = None,
    n_samp: int = -1,
    n_jobs: int = 1,
    use_tqdm: bool = True,
    fmt: str = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Create a SALSA supermutant dataset for evaluation.

    Parameters
    ----------
    smiles_file : str or pd.DataFrame
        Input SMILES data. Must have a 'SMILES' column; 'ID' is used if given.
    output_file : str, optional
        Output path. If None, generates a timestamped name in cwd. A '.csv'
        extension writes CSV, otherwise parquet is used.
    config : SuperMutantConfig, optional
        Supermutant configuration (uses defaults if None).
    n_samp : int
        Number of anchors to sample (-1 for all).
    n_jobs : int
        Worker processes to use.
    use_tqdm : bool
        Show progress bar.
    fmt : str, optional
        Output format ('parquet' or 'csv') used when the output name is
        generated or has no recognised extension. Defaults to parquet.

    Returns
    -------
    df : pd.DataFrame
        Supermutant dataset.
    output_path : str
        Path to the saved file.

    Example
    -------
    >>> from salsa_lit import supermutate_dataset, SuperMutantConfig
    >>> df, path = supermutate_dataset(
    ...     "anchors.csv",
    ...     output_file="supermutants.csv",
    ...     config=SuperMutantConfig(depth=5),
    ...     n_samp=10000,
    ... )
    """
    config = config or SuperMutantConfig()

    # Determine output path up front, so the path returned is the path written
    if output_file is None:
        n_ancs = n_samp if n_samp > 0 else 'all'
        acts = ''.join(a[0] for a in config.actions)
        output_file = os.path.join(
            os.getcwd(),
            with_ext(f'supermutants_{timestamp()}_{n_ancs}-ancs_'
                     f'{config.depth}-depth_{config.n_branches}-branches_'
                     f'{config.mahala_thresh}-mthresh_{config.weight}-weight_'
                     f'{acts}-acts', fmt or DEFAULT_FORMAT)
        )
    out_dir, out_name = os.path.split(output_file)

    df = supermutate_data(
        ds=smiles_file,
        out_name=out_name,
        out_dir=out_dir or os.getcwd(),
        n_samp=n_samp,
        depth=config.depth,
        n_branches=config.n_branches,
        actions=config.actions,
        mahala_thresh=config.mahala_thresh,
        weight=config.weight,
        n_jobs=n_jobs,
        use_tqdm=use_tqdm,
        fmt=fmt,
    )

    return df, output_file


def prepare_supermutant_data(
    smiles_file: Union[str, pd.DataFrame],
    output_dir: str = None,
    config: SuperMutantConfig = None,
    n_samp: int = -1,
    filter_data: bool = True,
    max_len: int = 100,
    max_atoms: int = 35,
    n_jobs: int = 1,
    fmt: str = None,
) -> Tuple[pd.DataFrame, str]:
    """
    Full pipeline to prepare supermutant evaluation data from raw SMILES.

    This function:
    1. Loads and filters the source SMILES
    2. Grows a supermutant chain per anchor
    3. Saves the result

    Parameters
    ----------
    smiles_file : str or pd.DataFrame
        Input SMILES data.
    output_dir : str, optional
        Output directory.
    config : SuperMutantConfig, optional
        Configuration for supermutant construction.
    n_samp : int
        Number of anchors to sample.
    filter_data : bool
        Whether to filter source data first.
    max_len : int
        Maximum SMILES length for filtering.
    max_atoms : int
        Maximum number of atoms for filtering.
    n_jobs : int
        Worker processes to use.
    fmt : str, optional
        Output format ('parquet' or 'csv'). Defaults to parquet.

    Returns
    -------
    df : pd.DataFrame
        Supermutant dataset.
    output_path : str
        Path to the saved file.
    """
    config = config or SuperMutantConfig()

    df = read_ds(smiles_file)

    if filter_data:
        df = filter_source_ds(df, max_len=max_len, max_atoms=max_atoms)
        print(f"Filtered to {len(df)} molecules")

    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    n_ancs = len(df) if n_samp < 0 else min(n_samp, len(df))
    acts = ''.join(a[0] for a in config.actions)
    output_name = with_ext(
        f'supermutants_{timestamp()}_{n_ancs}-ancs_'
        f'{config.depth}-depth_{config.n_branches}-branches_'
        f'{config.mahala_thresh}-mthresh_{config.weight}-weight_'
        f'{acts}-acts',
        fmt or DEFAULT_FORMAT
    )
    output_path = os.path.join(output_dir, output_name)

    result_df = supermutate_data(
        ds=df,
        out_name=output_name,
        out_dir=output_dir,
        n_samp=n_samp,
        depth=config.depth,
        n_branches=config.n_branches,
        actions=config.actions,
        mahala_thresh=config.mahala_thresh,
        weight=config.weight,
        n_jobs=n_jobs,
        use_tqdm=True,
        fmt=fmt,
    )

    return result_df, output_path
