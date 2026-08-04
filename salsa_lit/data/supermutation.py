"""
Supermutant construction for SALSA evaluation.

A "supermutant" is a chain of successive single-edit mutations away from an
anchor molecule:

    anchor -> g~(1) -> g~(2) -> ... -> g~(depth)
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from salsa_lit.constants import (
    ACTIONS, MAHALA_THRESH, MAX_MUT_LEN, WEIGHT,
)
from salsa_lit.data.mutation import _get_muts_from_anc, _is_real_mol
from salsa_lit.utils.io import read_ds, timestamp


SUPERMUT_COLS = [
    'ID', 'Anc_id', 'Branch_id', 'Src_id', 'Depth', 'SMILES', 'Is_anc',
    'Action', 'At_atom', 'To_atom', 'Prop_dist',
]


def supermutate_data(
    ds: Union[str, pd.DataFrame],
    out_name: str = None,
    out_dir: str = None,
    n_samp: int = -1,
    depth: int = 5,
    n_branches: int = 1,
    actions: List[str] = ACTIONS,
    mahala_thresh: float = MAHALA_THRESH,
    weight: str = WEIGHT,
    n_jobs: int = 1,
    use_tqdm: bool = False,
) -> pd.DataFrame:
    """
    Build a supermutant dataset from a set of anchor molecules.

    Parameters
    ----------
    ds : pd.DataFrame or str
        DataFrame (or path to CSV) with a 'SMILES' column. An 'ID' column is
        used if present, otherwise anchors are numbered from 0.
    out_name : str, optional
        Output filename. Defaults to a timestamped name.
    out_dir : str, optional
        Output directory. Defaults to cwd.
    n_samp : int
        Number of anchors to sample (-1 for all).
    depth : int
        Maximum GED depth. Each anchor yields one compound per depth 1..depth.
    n_branches : int
        Independent chains to grow per anchor. Each gets its own Branch_id.
    actions : list of str
        Mutation actions: ['Add', 'Replace', 'Remove'].
    mahala_thresh : float
        Max Mahalanobis property distance for an accepted mutation. Applied
        at every step, always against that step's own source compound.
    weight : str
        Weighting scheme: 'uniform' or 'chembl'.
    n_jobs : int
        Worker processes. 1 runs in the calling process.
    use_tqdm : bool
        Show a progress bar.

    Returns
    -------
    df : pd.DataFrame
        Supermutant dataset with columns SUPERMUT_COLS. Anchors have
        Is_anc=True and Depth=0.

    Notes
    -----
    Anchors whose chain dies before reaching `depth` are dropped entirely, so
    the returned frame usually covers fewer anchors than were supplied. The
    survival rate falls as `depth` and the strictness of `mahala_thresh` rise.
    """
    df = read_ds(ds)

    if 'SMILES' not in df.columns:
        df = df.rename({'Smiles': 'SMILES'}, axis=1, errors='ignore')
    if 'SMILES' not in df.columns:
        raise ValueError("Input data must have a 'SMILES' column.")

    if n_samp > 0 and n_samp < len(df):
        df = df.sample(n_samp)
    df = df.reset_index(drop=True)

    if 'ID' not in df.columns:
        df['ID'] = df.index

    # Independent of filter_source_ds, since callers may pass raw data or use
    # --no-filter. An unparseable or atomless anchor cannot start a chain.
    anchors = [(i, s) for i, s in zip(df['ID'].tolist(), df['SMILES'].tolist())
               if _is_real_mol(s)]
    n_bad = len(df) - len(anchors)
    if n_bad:
        print(f"Skipped {n_bad} anchors that are not valid molecules")

    if out_name is None:
        acts = ''.join(a[0] for a in actions)
        out_name = (f'supermutants_{timestamp()}_{len(anchors)}-ancs_'
                    f'{depth}-depth_{n_branches}-branches_{mahala_thresh}-mthresh_'
                    f'{weight}-weight_{acts}-acts.csv')
    if out_dir is None:
        out_dir = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, out_name)

    kwargs = dict(depth=depth, n_branches=n_branches, actions=actions,
                  mahala_thresh=mahala_thresh, weight=weight)

    branches: List[List[dict]] = []
    if n_jobs == 1:
        for anc_id, anc_smi in tqdm(anchors, disable=not use_tqdm,
                                    desc='supermutating'):
            branches.extend(_branches_for_anchor(anc_id, anc_smi, **kwargs))
    else:
        chunks = _chunk(anchors, n_jobs * 16)
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_branches_for_chunk, c, kwargs) for c in chunks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            disable=not use_tqdm,
                            desc=f'supermutating ({n_jobs} jobs)'):
                branches.extend(fut.result())

    df_out = _assemble(branches)
    df_out.to_csv(out_file, index=False, na_rep='None')

    n_anc = df_out['Anc_id'].nunique() if len(df_out) else 0
    print(f"Supermutants: {len(df_out)} compounds from {n_anc} anchors "
          f"({len(branches)} complete branches of depth {depth})")
    if len(anchors):
        print(f"  branch survival: {len(branches)}/{len(anchors) * n_branches} "
              f"({100 * len(branches) / (len(anchors) * n_branches):.1f}%)")
    print(f"  saved to {out_file}")

    return df_out


def consolidate_supmut_chunks(work_path: str, out_path: str = None) -> str:
    """
    Merge supermutant CSV chunks into one dataset, renumbering IDs.

    Parameters
    ----------
    work_path : str
        Directory of supermutant CSVs to merge.
    out_path : str, optional
        Output path for the merged dataset.

    Returns
    -------
    out_path : str
        Path to the merged dataset.
    """
    import glob

    if out_path is None:
        out_path = os.path.join(os.getcwd(),
                                f'supermutants_merged_{timestamp()}.csv')

    csv_files = sorted(glob.glob(os.path.join(work_path, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {work_path}")

    dfs = []
    anc_offset = 0
    for fname in csv_files:
        sub = pd.read_csv(fname)
        # Anc_id is only unique within a chunk, so shift each chunk into its
        # own range before concatenating.
        sub['Anc_id'] = sub['Anc_id'] + anc_offset
        anc_offset = sub['Anc_id'].max() + 1
        dfs.append(sub)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df = df.sort_values(by=['Anc_id', 'Branch_id', 'Depth'])
    df = df.reset_index(drop=True)
    df['ID'] = df.index

    df.to_csv(out_path, index=False, na_rep='None')
    return out_path


def _chunk(items: List, n: int) -> List[List]:
    """Split items into at most n roughly equal contiguous chunks."""
    if n <= 1 or len(items) <= 1:
        return [items]
    size = max(1, len(items) // n)
    return [items[i:i + size] for i in range(0, len(items), size)]


def _branches_for_chunk(anchors: List[Tuple], kwargs: dict) -> List[List[dict]]:
    """Process pool entry point: grow branches for a chunk of anchors."""
    out = []
    for anc_id, anc_smi in anchors:
        out.extend(_branches_for_anchor(anc_id, anc_smi, **kwargs))
    return out


def _branches_for_anchor(
    anc_id,
    anc_smi: str,
    depth: int,
    n_branches: int,
    actions: List[str],
    mahala_thresh: float,
    weight: str,
) -> List[List[dict]]:
    """Grow up to n_branches complete chains from one anchor."""
    out = []
    for branch_id in range(n_branches):
        branch = _grow_branch(anc_id, anc_smi, branch_id, depth,
                              actions, mahala_thresh, weight)
        if branch is not None:
            out.append(branch)
    return out


def _grow_branch(
    anc_id,
    anc_smi: str,
    branch_id: int,
    depth: int,
    actions: List[str],
    mahala_thresh: float,
    weight: str,
) -> Optional[List[dict]]:
    """
    Grow one chain of successive single edits from an anchor.

    Returns
    -------
    branch : list of dict or None
        depth + 1 rows (anchor at Depth 0 through Depth `depth`), or None if
        the chain died or revisited a SMILES.
    """
    branch = [{
        'Anc_id': anc_id, 'Branch_id': branch_id, 'Depth': 0,
        'SMILES': anc_smi, 'Is_anc': True,
        'Action': 'None', 'At_atom': -1, 'To_atom': 'None', 'Prop_dist': -1,
    }]
    seen = {anc_smi}
    src_smi = anc_smi

    for step in range(1, depth + 1):
        mut = _get_single_mut(src_smi, actions, mahala_thresh, weight)
        if mut is None:
            return None

        mut_smi, action, at_atom, to_atom, prop_dist = mut

        if mut_smi in seen:
            return None

        branch.append({
            'Anc_id': anc_id, 'Branch_id': branch_id, 'Depth': step,
            'SMILES': mut_smi, 'Is_anc': False,
            'Action': action, 'At_atom': at_atom, 'To_atom': to_atom,
            'Prop_dist': prop_dist,
        })
        seen.add(mut_smi)
        src_smi = mut_smi

    return branch


def _get_single_mut(
    src_smi: str,
    actions: List[str],
    mahala_thresh: float,
    weight: str,
) -> Optional[Tuple]:
    """
    Draw one mutant from a source compound.

    Returns the (SMILES, Action, At_atom, To_atom, Prop_dist) tuple, or None
    if no acceptable mutation exists.
    """
    mut_tups, _, failed = _get_muts_from_anc(
        src_smi, max_mut_len=MAX_MUT_LEN, num_muts=1, actions=actions,
        mahala_thresh=mahala_thresh, weight=weight,
    )
    if failed or not mut_tups:
        return None
    return tuple(mut_tups[0])


def _assemble(branches: List[List[dict]]) -> pd.DataFrame:
    """
    Flatten branches into a frame, assigning global IDs and parent links.

    Src_id points at the compound one edit closer to the anchor; the anchor
    points at itself.
    """
    if not branches:
        return pd.DataFrame(columns=SUPERMUT_COLS)

    # Parallel chunks complete out of order, so sort before numbering to keep
    # the output row order (and therefore the IDs) reproducible.
    branches = sorted(branches, key=lambda b: (b[0]['Anc_id'], b[0]['Branch_id']))

    rows = []
    next_id = 0
    for branch in branches:
        prev_id = None
        for row in branch:
            row = dict(row)
            row['ID'] = next_id
            row['Src_id'] = next_id if prev_id is None else prev_id
            prev_id = next_id
            next_id += 1
            rows.append(row)

    return pd.DataFrame(rows)[SUPERMUT_COLS]
