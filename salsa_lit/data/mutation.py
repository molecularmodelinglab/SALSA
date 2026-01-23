"""
Molecule mutation utilities for SALSA dataset creation.

This module provides functions to generate mutated versions of molecules
for contrastive learning. Mutations include adding, removing, and replacing atoms.
"""

import os
import copy
import random
from typing import List, Tuple, Optional, Union

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Atom, GetPeriodicTable, BondType
from rdkit.Chem.rdmolops import SanitizeMol

from salsa_lit.constants import (
    ATOM_TO_CNT, MAHALA_THRESH, N_MUTS, MAX_ANC_LEN, MAX_MUT_LEN,
    ATOMS, ACTIONS, get_prop_names, get_inv_cov_chembl
)
from salsa_lit.utils.io import read_ds, timestamp
from salsa_lit.utils.chemistry import get_props, get_cansmiles, count_atoms


# Lazy-load constants at module level for backward compatibility
PROP_NAMES = None
INV_COV_CHEMBL = None


def _get_prop_names():
    """Get property names, loading if necessary."""
    global PROP_NAMES
    if PROP_NAMES is None:
        PROP_NAMES = get_prop_names()
    return PROP_NAMES


def _get_inv_cov_chembl():
    """Get inverse covariance matrix, loading if necessary."""
    global INV_COV_CHEMBL
    if INV_COV_CHEMBL is None:
        INV_COV_CHEMBL = get_inv_cov_chembl()
    return INV_COV_CHEMBL


# Inverse atom counts for removal weighting
ATOM_TO_CNT_INV = {atom: 1 / cnt for atom, cnt in ATOM_TO_CNT.items()}


# =============================================================================
# Dataset filtering
# =============================================================================

def filter_source_ds(
    ds: Union[str, pd.DataFrame],
    max_len: int = MAX_ANC_LEN,
    max_atoms: int = 35
) -> pd.DataFrame:
    """
    Filter a dataset to match SALSA specs.

    Performs:
    - Canonicalization
    - Filtering by max SMILES length
    - Filtering by max number of heavy atoms

    Parameters
    ----------
    ds : str or pd.DataFrame
        Dataset with 'SMILES' column
    max_len : int
        Maximum SMILES length
    max_atoms : int
        Maximum number of heavy atoms

    Returns
    -------
    df : pd.DataFrame
        Filtered dataset
    """
    df = read_ds(ds)

    # Canonicalize smiles
    df['SMILES'] = df["SMILES"].apply(lambda x: get_cansmiles(x))

    # Filter by length
    df['Len'] = df["SMILES"].apply(lambda x: len(x))
    df = df[df.Len <= max_len]

    # Filter by num atoms
    df['N_atoms'] = df["SMILES"].apply(lambda x: count_atoms(x))
    df = df[df.N_atoms <= max_atoms]

    if "ID" not in df.columns:
        df.reset_index(drop=True, inplace=True)
        df['ID'] = df.index

    return df


# =============================================================================
# Main mutation function
# =============================================================================

def mutate_data(
    ds: Union[str, pd.DataFrame],
    out_name: str = None,
    out_dir: str = None,
    n_samp: int = -1,
    num_muts: int = N_MUTS,
    actions: List[str] = ACTIONS,
    mahala_thresh: float = MAHALA_THRESH,
    weight: str = 'chembl',
    use_tqdm: bool = False
) -> pd.DataFrame:
    """
    Mutate a dataset to create SALSA training data.

    Creates families of molecules where each anchor has multiple mutant
    variations for contrastive learning.

    Parameters
    ----------
    ds : pd.DataFrame or str
        DataFrame (or path to CSV) with SMILES and ID columns
    out_name : str, optional
        Output filename. Defaults to timestamped name.
    out_dir : str, optional
        Output directory. Defaults to cwd.
    n_samp : int
        Number of SMILES to use (-1 for all)
    num_muts : int
        Number of mutations per anchor
    actions : list of str
        Mutation actions: ['Add', 'Replace', 'Remove']
    mahala_thresh : float
        Max Mahalanobis distance for mutants
    weight : str
        Weighting scheme: 'uniform' or 'chembl'
    use_tqdm : bool
        Show progress bar

    Returns
    -------
    df : pd.DataFrame
        Mutated dataset with columns:
        [Is_anc, Anc_id, SMILES, Action, At_atom, To_atom, Prop_dist]
    """
    df = read_ds(ds)

    if n_samp > 0:
        df = df.sample(n_samp)
        df.reset_index(drop=True, inplace=True)
    else:
        n_samp = len(df)

    # Trim dataframe to relevant columns
    df = df[['ID', 'SMILES']]

    # Set outputs
    if out_name is None:
        out_name = f'mutants_{timestamp()}_{n_samp}-ancs_{num_muts}-muts_{mahala_thresh}-mthresh_{weight}-weight.csv'
    if out_dir is None:
        out_dir = os.getcwd()
    out_file = os.path.join(out_dir, out_name)

    # Write header
    with open(out_file, 'w') as f:
        f.write(",".join(['Is_anc', 'Anc_id', 'Smiles', 'Action', 'At_atom', 'To_atom', 'Prop_dist']) + "\n")

    # Process mutants for each SMILES
    rows = []
    for row in tqdm(df.values, total=len(df), disable=not use_tqdm):
        anc_id = row[0]
        anc_smi = row[1]
        mut_tups, _, failed = _get_muts_from_anc(
            anc_smi, num_muts=num_muts, actions=actions, weight=weight,
            mahala_thresh=mahala_thresh, max_mut_len=MAX_MUT_LEN
        )
        if failed:
            continue
        else:
            muts = [[False, anc_id, *x] for x in mut_tups]
            with open(out_file, 'a') as f:
                rows.append([True, anc_id, anc_smi, 'None', -1, 'None', -1])
                f.write(",".join([str(_) for _ in [True, anc_id, anc_smi, 'None', -1, 'None', -1]]) + "\n")

                for mut in muts:
                    f.write(",".join([str(_) for _ in mut]) + "\n")
                    rows.append(mut)

    cols = ['Is_anc', 'Anc_id', 'SMILES', 'Action', 'At_atom', 'To_atom', 'Prop_dist']
    df_out = pd.DataFrame(rows, columns=cols)

    return df_out


def consolidate_mut_ds_chunks(work_path: str, out_path: str = None) -> str:
    """
    Merge multiple mutation datasets into a single training dataset.

    Parameters
    ----------
    work_path : str
        Directory containing mutation datasets to merge
    out_path : str, optional
        Output path for merged dataset

    Returns
    -------
    out_path : str
        Path to merged dataset
    """
    import glob

    if out_path is None:
        out_path = os.path.join(os.getcwd(), timestamp() + "_merged_mutated_datasets.csv")

    csv_names = f'{work_path}/*.csv'
    csv_files = glob.glob(csv_names)

    df_list = []
    for fname in csv_files:
        subdf = pd.read_csv(fname)
        df_list.append(subdf)

    df = pd.concat(df_list, axis=0, ignore_index=True)
    df = df.sort_values(by=['Anc_id', 'Is_anc'], ascending=[True, False])
    df.to_csv(out_path, index=False, na_rep='None')

    return out_path


# =============================================================================
# Internal mutation helpers
# =============================================================================

def _get_muts_from_anc(
    anc_smi: str,
    max_mut_len: int,
    num_muts: int,
    actions: List[str],
    mahala_thresh: float,
    weight: str
) -> Tuple[List, List, bool]:
    """
    Get mutations for a given anchor SMILES.

    Parameters
    ----------
    anc_smi : str
        Anchor SMILES
    max_mut_len : int
        Maximum mutant SMILES length
    num_muts : int
        Number of mutants to generate
    actions : list
        Mutation actions
    mahala_thresh : float
        Mahalanobis distance threshold
    weight : str
        Weighting scheme

    Returns
    -------
    muts_samp : list
        Sampled mutants
    neg_muts : list
        Rejected mutants
    failed : bool
        True if mutation failed
    """
    # Check anc smi length
    if len(anc_smi) > MAX_MUT_LEN:
        return [], [], True

    # Make sure provided actions are valid
    if len(set(actions) - set(ACTIONS)) > 0:
        raise ValueError(f"Invalid actions! {actions} ... (Valid actions: {ACTIONS})")

    # Convert mahala_thresh from str to proper value
    if mahala_thresh == 'Record':
        mahala_thresh = np.inf

    try:
        # Get all possible mutants for a SMILES
        muts_full = _get_mut_pool_for_anc(anc_smi, max_mut_len=max_mut_len, actions=actions)

        # Sample only the number of mutants needed
        muts_samp = _sample_muts(
            muts_full, anc_smi, num_muts=num_muts, actions=actions,
            mahala_thresh=mahala_thresh, weight=weight
        )
        return muts_samp

    except Exception:
        return ([], [], True)


def _get_mut_pool_for_anc(
    anc_smi: str,
    max_mut_len: int,
    actions: List[str]
) -> List[Tuple]:
    """
    Generate all possible mutations for an anchor SMILES.

    Parameters
    ----------
    anc_smi : str
        Anchor SMILES
    max_mut_len : int
        Maximum mutant length
    actions : list
        Mutation actions

    Returns
    -------
    valid_muts : list of tuples
        Valid mutation tuples
    """
    mol = Chem.MolFromSmiles(anc_smi)

    # Apply actions to every atom in the anchor molecule
    atom_idc = [i for i in range(0, (mol.GetNumAtoms()))]
    mut_tups = []
    for i in atom_idc:
        for action_type in actions:
            if action_type == 'Remove':
                mut_tup = _generate_mut_tup(mol, action_type, i, atom_type=None, max_mut_len=max_mut_len)
                mut_tups.append(mut_tup)
            else:
                # For Add/Replace, loop through all possible atom types
                for atom_type in ATOMS:
                    mut_tup = _generate_mut_tup(mol, action_type, i, atom_type=atom_type, max_mut_len=max_mut_len)
                    mut_tups.append(mut_tup)

    # Do not return None mutant objects
    valid_mut_tups = [tuple(x) for x in mut_tups if x]
    return valid_mut_tups


def _sample_muts(
    muts: List[Tuple],
    anc_smi: str,
    num_muts: int,
    actions: List[str],
    mahala_thresh: float,
    weight: str
) -> Tuple[List, List, bool]:
    """
    Sample mutants with Mahalanobis distance filtering.

    Parameters
    ----------
    muts : list of tuples
        Pool of possible mutants
    anc_smi : str
        Anchor SMILES
    num_muts : int
        Number of mutants to sample
    actions : list
        Mutation actions
    mahala_thresh : float
        Mahalanobis distance threshold
    weight : str
        Weighting scheme: 'uniform' or 'chembl'

    Returns
    -------
    muts_samp : list
        Sampled mutants that passed filter
    neg_muts : list
        Rejected mutants
    failed : bool
        True if zero mutants generated
    """
    # Calculate properties for anc if mahala filtering is used
    if mahala_thresh is not None:
        anc_prop = get_props(anc_smi, prop_list=_get_prop_names())

    actions = np.array(actions)
    act_to_muts = {act: [x for x in muts if x[1] == act] for act in actions}

    muts_samp = []
    mut_smis = []
    neg_muts = []

    while True:
        if len(muts_samp) == num_muts:
            break

        muts_remaining = sum([len(muts) for muts in act_to_muts.values()])
        if muts_remaining == 0:
            break

        acts_remaining = [act for act, muts in act_to_muts.items() if len(muts) > 0]
        action = random.choice(acts_remaining)
        muts_at_act = act_to_muts[action]

        if not muts_at_act:
            continue

        if weight == 'uniform':
            random.shuffle(muts_at_act)
            mut_draw = muts_at_act.pop()
            mut_smi, mut_type, i, atom = mut_draw

        elif weight == 'chembl':
            atoms_at_act = list(set([x[3] for x in muts_at_act]))

            if action == 'Remove':
                atom_to_cnt = {k: v for k, v in ATOM_TO_CNT_INV.items() if k in atoms_at_act}
            else:
                atom_to_cnt = {k: v for k, v in ATOM_TO_CNT.items() if k in atoms_at_act}
            atom_draw = _get_weighted_random_atom(atom_to_cnt)

            muts_pool = [x for x in muts_at_act if x[3] == atom_draw]
            mut_draw = muts_pool.pop()
            mut_smi, mut_type, i, atom = mut_draw

            # Correct the mutant-pool value to account for draw
            corrected_muts_at_atc = set(muts_at_act) - {mut_draw}
            act_to_muts[action] = list(corrected_muts_at_atc)
        else:
            raise ValueError(f"Unknown weight setting: {weight}, must be from ['chembl', 'uniform']")

        if mut_smi in mut_smis:
            continue

        # Mahalanobis distance filtering
        if mahala_thresh is not None:
            mut_prop = get_props(mut_smi, prop_list=_get_prop_names())
            d = distance.mahalanobis(anc_prop, mut_prop, VI=_get_inv_cov_chembl())
            if d <= mahala_thresh:
                muts_samp.append([mut_smi, mut_type, i, atom, round(d, 2)])
                mut_smis.append(mut_smi)
            else:
                neg_muts.append([mut_smi, mut_type, i, atom, round(d, 2)])
        else:
            # Set prop_dist to -1 if no Mahala filter is used
            muts_samp.append([mut_smi, mut_type, i, atom, -1])
            mut_smis.append(mut_smi)

    # Up-sample if not enough mutants were generated
    if num_muts:
        while len(muts_samp) < num_muts:
            try:
                muts_samp.append(random.choice(muts_samp))
            except IndexError:
                return [], [], True

    return muts_samp, neg_muts, False


# =============================================================================
# Mutation actions
# =============================================================================

def _generate_mut_tup(
    anc_mol,
    action_type: str,
    i: int,
    atom_type: str = None,
    max_mut_len: int = MAX_MUT_LEN
) -> Optional[List]:
    """
    Generate a single mutation tuple.

    Parameters
    ----------
    anc_mol : Mol
        Anchor molecule
    action_type : str
        'Add', 'Replace', or 'Remove'
    i : int
        Atom index to mutate
    atom_type : str, optional
        Atom type for Add/Replace
    max_mut_len : int
        Maximum mutant SMILES length

    Returns
    -------
    mut_tup : list or None
        [mut_smi, action_type, i, atom_type] if valid
    """
    # Disable RDKit logging
    RDLogger.DisableLog('rdApp.*')

    mol = copy.copy(anc_mol)
    for atom in mol.GetAtoms():
        atom.SetProp("State", "Anc")

    try:
        if action_type == 'Remove':
            mut_mol, atom_type = remove_atom_from_mol(mol, i)
        elif action_type == 'Add':
            mut_mol = _add_atom_to_mol(mol, atom_type, i)
        elif action_type == 'Replace':
            curr_atype = mol.GetAtomWithIdx(i).GetSymbol()
            if atom_type == curr_atype:
                return
            else:
                mut_mol = replace_atom_on_mol(mol, atom_type, i)
        else:
            mut_mol = None
    except Exception:
        return

    if mut_mol is None:
        raise ValueError(f"failed to recognize action '{action_type}'")

    # Return none if mol is invalid
    if mut_mol.GetNumAtoms() == 0:
        return
    # Return none if smiles is >= max len
    elif len(Chem.MolToSmiles(mut_mol)) > max_mut_len:
        return
    # Else, return mut tuple
    else:
        mut_smi = get_cansmiles(Chem.MolToSmiles(mut_mol))
        mut_tup = [mut_smi, action_type, i, atom_type]
        return mut_tup


def _add_atom_to_mol(mol, atom_type: str, to_aidx: int, clean_aroms: bool = True, remake_mol: bool = True):
    """Add an atom to a molecule."""
    molgraph = get_rwmol(mol)
    update_mol_rep(molgraph, clean_aroms)
    atom = Atom(atom_type)
    atom.SetBoolProp("mutability", True)
    atom.SetProp('State', 'Aug')
    molgraph.AddAtom(atom)
    molgraph.AddBond(molgraph.GetNumAtoms() - 1, to_aidx, BondType.SINGLE)
    update_mol_rep(molgraph, clean_aroms)
    mol_new = molgraph.GetMol()
    if remake_mol:
        mol_new = Chem.MolFromSmiles(Chem.MolToSmiles(mol_new))
    return mol_new


def replace_atom_on_mol(mol, atom_type: str, to_aidx: int, clean_aroms: bool = True, remake_mol: bool = True):
    """Replace an atom on a molecule."""
    molgraph = get_rwmol(mol)
    update_mol_rep(molgraph, clean_aroms)
    molgraph.GetAtomWithIdx(to_aidx).SetAtomicNum(GetPeriodicTable().GetAtomicNumber(atom_type))
    molgraph.GetAtomWithIdx(to_aidx).SetFormalCharge(0)
    molgraph.GetAtomWithIdx(to_aidx).SetProp('State', 'Aug')
    update_mol_rep(molgraph, clean_aroms)
    mol_new = molgraph.GetMol()
    if remake_mol:
        mol_new = Chem.MolFromSmiles(Chem.MolToSmiles(mol_new))
    return mol_new


def remove_atom_from_mol(mol, aidx: int, clean_aroms: bool = True, remake_mol: bool = True):
    """Remove an atom from a molecule (must have single bond)."""
    num_bonds = len(mol.GetAtomWithIdx(aidx).GetNeighbors())
    if num_bonds != 1:
        return Chem.Mol(), None
    molgraph = get_rwmol(mol)
    update_mol_rep(molgraph, clean_aroms)
    atom_removed = molgraph.GetAtomWithIdx(aidx).GetSymbol()
    molgraph.RemoveAtom(aidx)
    update_mol_rep(molgraph, clean_aroms)
    mol_new = molgraph.GetMol()
    if remake_mol:
        mol_new = Chem.MolFromSmiles(Chem.MolToSmiles(mol_new))
    return mol_new, atom_removed


# =============================================================================
# Accessory functions
# =============================================================================

def copy_atom(atom, atommap: bool = True):
    """Copy an RDKit atom."""
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap:
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def get_rwmol(mol):
    """Convert a Mol to an editable RWMol."""
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_atom.SetProp("State", atom.GetProp("State"))
        aroms = ['N']
        if atom.GetIsAromatic() and atom.GetSymbol() in aroms:
            new_atom.SetNumExplicitHs(atom.GetTotalNumHs())
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def update_mol_rep(molgraph, clean_aroms: bool = False, sanitize: bool = False):
    """Update RDKit molecular representation."""
    if sanitize:
        SanitizeMol(molgraph)
        molgraph.ClearComputedProps()
    if clean_aroms:
        Chem.Kekulize(molgraph)
        for i in range(molgraph.GetNumAtoms()):
            molgraph.GetAtomWithIdx(i).SetIsAromatic(False)
        for i in range(molgraph.GetNumAtoms()):
            for j in range(molgraph.GetNumAtoms()):
                bond = molgraph.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bond.SetIsAromatic(False)
    for i in range(molgraph.GetNumAtoms()):
        molgraph.GetAtomWithIdx(i).UpdatePropertyCache()
    molgraph.UpdatePropertyCache(strict=False)


def _get_weighted_random_atom(symb_to_count: dict) -> str:
    """
    Sample an atom type weighted by counts.

    Parameters
    ----------
    symb_to_count : dict
        Mapping of atom symbol to count/weight

    Returns
    -------
    atom_type : str
        Sampled atom type
    """
    a = np.array([x for x in symb_to_count.keys()])
    p = np.array([x for x in symb_to_count.values()])
    p = p / np.sum(p)
    atom_type = np.random.choice(a=a, p=p)
    return atom_type
