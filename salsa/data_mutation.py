import os
import copy
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance

from rdkit import Chem
from rdkit.Chem.rdchem import Atom, GetPeriodicTable, BondType
from rdkit.Chem.rdmolops import SanitizeMol


from salsa.utils import get_props, get_cansmiles, count_atoms, read_ds, timestamp
from salsa.constants import (ATOM_TO_CNT, MAHALA_THRESH, N_MUTS, MAX_ANC_LEN, MAX_MUT_LEN, ATOMS, ACTIONS,
                             PROP_NAMES, INV_COV_CHEMBL, WEIGHT)

ATOM_TO_CNT_INV = {atom: 1 / cnt for atom, cnt in ATOM_TO_CNT.items()}


# Functions for cleaning datasets #
def filter_source_ds(ds, max_len=MAX_ANC_LEN, max_atoms=35):
    """
    Given a dataset containing a 'Smiles' column!!! ...
    Return a dataset that is agreement with Salsa specs:
        • Canonicalized
        • Filtered by max SMILES length
        • Filtered by max number of heavy atoms 
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
        df['ID'] = df.index()

    return df


def mutate_data(ds, out_name=None, out_dir=None, n_samp=-1, num_muts=N_MUTS, actions=ACTIONS,
                mahala_thresh=MAHALA_THRESH, weight=WEIGHT, use_tqdm=False):
    """
    This function mutates a dataset. It must contain a Smiles column AND an Id!!
    It returns the dataframe and also writes it to output.

    Parameters
    ----------
    ds: pd.Dataframe or str
        dataframe (or path to csv file) with SMILES to "mutate"
    out_name: str = None
        name of the output file of mutated data, defaults to timestamped name if not passed
    out_dir: str = None
        directory to save mutated data, defaults to `cwd` if not passed
    n_samp: int = -1
        max number of SMILES to use when building the dataset
        if left as -1, will default to all SMILES passed in the dataframe
    num_muts: int = N_MUTS
        number of mutations to do for each anchor SMILE
        defaults to the constant set in `constants.py`
    actions: list of action literals
        which type of mutation actions to take, right now possible actions are ["Add", "Replace", "Remove"]
        defaults to the list set in `constants.py`
    mahala_thresh: int or float
        threshold for the max mahala distance between the mutated SMILES and the anchor
        defaults to the values set in `constants.py`
    weight: literal
        Whether to weight mutations uniformly or by ChEMBL distribution
        'uniform' or 'chembl'
        defaults to the value set in `constants.py`
    use_tqdm: bool
        render tqdm progress bar

    Returns
    -------
    mutated_dataset: pd.Dataframe
        Dataframe with anchor and mutated SMILES for training a SALSA model

    Notes
    -----
    This will also write the mutated data to a file on the fly, as well as return the dataframe
    """
    df = read_ds(ds)

    if n_samp > 0:
        df = df.sample(n_samp)
        df.reset_index(drop=True, inplace=True)
    else:
        n_samp = len(df)

    # trim dataframe to relevant columns
    df = df[['ID', 'SMILES']]

    # set outputs
    if out_name is None:
        out_name = f'mutants_{timestamp()}_{n_samp}-ancs_{num_muts}-muts_{mahala_thresh}-mthresh_{weight}-weight.csv'
    if out_dir is None:
        out_dir = os.getcwd()
    out_file = os.path.join(out_dir, out_name)

    # write header
    with open(out_file, 'w') as f:
        f.write(",".join(['Is_anc', 'Anc_id', 'Smiles', 'Action', 'At_atom', 'To_atom', 'Prop_dist']) + "\n")

    # process mutants for each SMILES
    rows = []
    for row in tqdm(df.values, total=len(df), disable=not use_tqdm):
        anc_id = row[0]
        anc_smi = row[1]
        mut_tups, _, failed = _get_muts_from_anc(anc_smi, num_muts=num_muts, actions=actions, weight=weight,
                                                 mahala_thresh=mahala_thresh, max_mut_len=MAX_MUT_LEN)
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


def consolidate_mut_ds_chunks(work_path, out_path: None):
    """
    Use to merge a set of mutation datasets into a single mutation dataset for training of model

    Parameters
    ----------
    work_path: str
        path to the directory containing ONLY the mutation datasets to be merged
    out_path: str
        absolute path to where the output should be saved (directory and filename)

    Returns
    -------

    """
    import glob

    if out_path is None:
        out_path = os.path.join(os.getcwd(), timestamp()+"_merged_mutated_datasets.csv")

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


# Functions to get 'deterministic' mutants #
def _get_muts_from_anc(anc_smi, max_mut_len, num_muts, actions, mahala_thresh, weight):
    """
    This is an internal helper function to get the mutations from a given anchor SMILES
    This function does the following:
        For a single anchor smiles ...
            1) Makes a call to get the pool of mutants for the anchor.
            2) Makes a call to sample n mutants (num_muts) from the entire pool.

    Parameters
    ----------
    anc_smi: str
        the SMILES to have mutates generated for
    max_mut_len: int
        max length of the SMILES string generated for mutants
    num_muts: int
        number of mutants to generate from the given anchor
    actions: list of action literals
        which type of mutation actions to take, right now possible actions are ["Add", "Replace", "Remove"]
    mahala_thresh: int or float
        threshold for the max mahala distance between the mutated SMILES and the anchor
        for no filter, use 'None'
        for no filter but record Mahala, use 'Record'
        for filter, use threshold value (e.g. 6.75)
    weight: literal
        Whether to weight mutations uniformly or by ChEMBL distribution
        'uniform' or 'chembl'

    Returns
    -------
    muts_samp: tuple
        Sample of mutants for given anchor.
    """

    # check anc smi length
    if len(anc_smi) > MAX_MUT_LEN:
        return [], [], True

    # make sure provided actions are valid
    if len(set(actions) - set(ACTIONS)) > 0:
        raise ValueError(f"Invalid actions! {actions} ... (Valid actions: {ACTIONS})")

    # convert mahala_thresh from str to proper value
    if mahala_thresh == 'Record':
        mahala_thresh = np.inf

    try:
        # get all possible mutants for a SMILES
        # TODO this could be dangerous later on if the number of actions and valid mutants gets large
        muts_full = _get_mut_pool_for_anc(anc_smi, max_mut_len=max_mut_len, actions=actions)

        # sample only the number of mutants needed
        muts_samp = _sample_muts(muts_full, anc_smi, num_muts=num_muts, actions=actions,
                                 mahala_thresh=mahala_thresh, weight=weight)
        return muts_samp

    except Exception as e:  # TODO come back and make sure we check the right error
        return ([], [], True)


def _get_mut_pool_for_anc(anc_smi, max_mut_len, actions):
    """
    Internal function iterates through the atoms and the set of mutation
    actions to produce all possible mutants for a given anchor smiles.

    Parameters
    ----------
    anc_smi: str
        the SMILES to have mutates generated for
    max_mut_len: int
        max length of the SMILES string generated for mutants
    actions: list of action literals
        which type of mutation actions to take, right now possible actions are ["Add", "Replace", "Remove"]

    Returns
    -------
    valid_muts: list of tuples
        the valid mutants that can be generated for the given anchor SMILES
    """
    mol = Chem.MolFromSmiles(anc_smi)

    # Apply actions to every atom in the anchor molecule.
    atom_idc = [i for i in range(0, (mol.GetNumAtoms()))]
    mut_tups = []
    for i in atom_idc:
        for action_type in actions:
            if action_type == 'Remove':
                mut_tup = _generate_mut_tup(mol, action_type, i, atom_type=None, max_mut_len=max_mut_len)
                mut_tups.append(mut_tup)
            else:
                # If mutate action is based on an incoming atom (replace with new atom,
                # add new atom), then loop through all possible atom types. 
                for atom_type in ATOMS:
                    mut_tup = _generate_mut_tup(mol, action_type, i, atom_type=atom_type, max_mut_len=max_mut_len)
                    mut_tups.append(mut_tup)
    # Do not return `None` mutant objects
    valid_mut_tups = [tuple(x) for x in mut_tups if x]
    return valid_mut_tups


def _sample_muts(muts, anc_smi, num_muts, actions, mahala_thresh, weight):
    """
    Samples a subset of the mutants generated for a given anc_smi based on the weights.
     If the number of muts is less than the desired num_muts, will up-sample the muts until num_muts is meet.
    Will return a fail code if no muts were generated.

    Parameters
    ----------
    muts: list of tuples
        the mutants generated for the given anchor
    anc_smi: str
        the SMILES to have mutates generated for
    num_muts: int
        number of mutants to generate from the given anchor
    actions: list of action literals
        which type of mutation actions to take, right now possible actions are ["Add", "Replace", "Remove"]
    mahala_thresh: int or float
        threshold for the max mahala distance between the mutated SMILES and the anchor
        for no filter, use 'None'
        for no filter but record Mahala, use 'Record'
        for filter, use threshold value (e.g. 6.75)
    weight: literal
        Whether to weight mutations uniformly or by ChEMBL distribution
        'uniform' or 'chembl'

    Returns
    -------
    list, list, bool
        list of muts that passed the mahala filter
        list of muts that failed to pass the filter
        `True` if zero mutants were generated, `False` otherwise

    Notes
    -----
    if not enough mutants were generated, will return two empty lists instead of partially filled ones
    if `mahala_thresh` is None, the second list will always be empty
    """

    # calculate properties for anc if mahala is -1 (Record) or a float.
    if mahala_thresh is not None:
        anc_prop = get_props(anc_smi, prop_list=PROP_NAMES)

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
            mut_prop = get_props(mut_smi, prop_list=PROP_NAMES)
            d = distance.mahalanobis(anc_prop, mut_prop, VI=INV_COV_CHEMBL)
            if d <= mahala_thresh:
                muts_samp.append([mut_smi, mut_type, i, atom, round(d, 2)])
                mut_smis.append(mut_smi)
            else:
                neg_muts.append([mut_smi, mut_type, i, atom, round(d, 2)])
        else:  # Set prop_dist to -1 if not Mahala filter is used.
            muts_samp.append([mut_smi, mut_type, i, atom, -1])
            mut_smis.append(mut_smi)

    # Sample from pool of mutants until there are enough mutations in the samp.
    if num_muts:
        while len(muts_samp) < num_muts:
            try:
                muts_samp.append(random.choice(muts_samp))
            except IndexError as e:
                return [], [], True

    return muts_samp, neg_muts, False


# Mutation actions #
def _generate_mut_tup(anc_mol, action_type, i, atom_type=None, max_mut_len=MAX_MUT_LEN):
    """
    This function wraps the three different mutate actions. Will perform
    whichever is specified on the provided anchor mol. Generates a
    mutation tuple (mutated smiles, action type, atom index)
    Parameters
    ----------
    anc_mol: Mol object
        molecule to be anchored
    action_type: literal
        action to be performed for mutation
        valid actions are ["Add", "Replace", "Remove"]
    i: int
        index of atom to be muted on anchor mol
    atom_type: literal str
        which atom type to Add or Replace. If None, Remove.
    max_mut_len: int
        max length of the SMILES string generated for mutants

    Returns
    -------
    tuple or `None`
        the mutated chemical in form '[mut_smi, action_type, i, atom_type]', if valid
        `None` if otherwise
    """

    # turn off the rdkit logger for this part, or you'll flood the console with warnings and errors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    mol = copy.copy(anc_mol)
    for atom in mol.GetAtoms():
        atom.SetProp("State", "Anc")  # Set 'state' to track muts.

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
    except Exception:  # RDKit throws a Boost Exception here, which python cannot detect ahead of time, hence this line
        return

    if mut_mol is None:
        raise ValueError(f"failed to recognize action '{action_type}'")

    # return none if mol is invalid
    if mut_mol.GetNumAtoms() == 0:
        return
    # return none if smiles is >= max len
    elif len(Chem.MolToSmiles(mut_mol)) > max_mut_len:
        return
    # else, return mut tuple
    else:
        mut_smi = get_cansmiles(Chem.MolToSmiles(mut_mol))
        mut_tup = [mut_smi, action_type, i, atom_type]
        return mut_tup


def _add_atom_to_mol(mol, atom_type, to_aidx, clean_aroms=True, remake_mol=True):
    molgraph = get_rwmol(mol)
    update_mol_rep(molgraph, clean_aroms)
    atom = Atom(atom_type)
    atom.SetBoolProp("mutability", True)
    atom.SetProp('State', 'Aug')  # set state to "Aug"
    molgraph.AddAtom(atom)
    molgraph.AddBond(molgraph.GetNumAtoms() - 1, to_aidx, BondType.SINGLE)
    update_mol_rep(molgraph, clean_aroms)
    mol_new = molgraph.GetMol()
    if remake_mol:
        mol_new = Chem.MolFromSmiles(Chem.MolToSmiles(mol_new))
    return mol_new


def replace_atom_on_mol(mol, atom_type, to_aidx, clean_aroms=True, remake_mol=True):
    molgraph = get_rwmol(mol)
    update_mol_rep(molgraph, clean_aroms)
    molgraph.GetAtomWithIdx(to_aidx).SetAtomicNum(GetPeriodicTable().GetAtomicNumber(atom_type))
    molgraph.GetAtomWithIdx(to_aidx).SetFormalCharge(0)
    molgraph.GetAtomWithIdx(to_aidx).SetProp('State', 'Aug')  # set state to "Aug"
    update_mol_rep(molgraph, clean_aroms)
    mol_new = molgraph.GetMol()
    if remake_mol:
        mol_new = Chem.MolFromSmiles(Chem.MolToSmiles(mol_new))
    return mol_new


def remove_atom_from_mol(mol, aidx, clean_aroms=True, remake_mol=True):
    num_bonds = len(mol.GetAtomWithIdx(aidx).GetNeighbors())
    if num_bonds != 1:
        return Chem.Mol()
    molgraph = get_rwmol(mol)
    update_mol_rep(molgraph, clean_aroms)
    atom_removed = molgraph.GetAtomWithIdx(aidx).GetSymbol()
    molgraph.RemoveAtom(aidx)
    update_mol_rep(molgraph, clean_aroms)
    mol_new = molgraph.GetMol()
    if remake_mol:
        mol_new = Chem.MolFromSmiles(Chem.MolToSmiles(mol_new))
    return mol_new, atom_removed


# Accessory functions for mutation actions #
def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap:
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def get_rwmol(mol):
    # TODO can this function not just be replaced by `Chem.RWMol(mol)`? See issue #2
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


def update_mol_rep(molgraph, clean_aroms=False, sanitize=False):
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
    # Updating RDKit representation
    molgraph.UpdatePropertyCache(strict=False)


def _get_weighted_random_atom(symb_to_count):
    """
    Given a dictionary of atom types to counts, return a randomly chosen atom,
    weighted according to the given count ratios.
    """
    tot = sum(symb_to_count.values())
    symb_to_prob = {}
    for k, v in symb_to_count.items():
        symb_to_prob[k] = v / tot

    a = np.array([x for x in symb_to_prob.keys()])
    p = np.array([x for x in symb_to_count.values()])
    p = p / np.sum(p)
    atom_type = np.random.choice(a=a, p=p)

    return atom_type
