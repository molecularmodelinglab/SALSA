"""
Chemistry utility functions using RDKit.
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Disable RDKit logging by default
RDLogger.DisableLog('rdApp.*')


def get_cansmiles(smiles, chiral=False):
    """
    Convert SMILES to canonical form.
    
    Parameters
    ----------
    smiles : str
        Input SMILES string
    chiral : bool
        Whether to include stereochemistry
    
    Returns
    -------
    canonical : str
        Canonical SMILES or empty string if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ''
        return Chem.MolToSmiles(mol, isomericSmiles=chiral)
    except Exception:
        return ''


def count_atoms(smiles):
    """
    Count heavy atoms in a molecule.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    
    Returns
    -------
    count : int
        Number of heavy atoms, or 0 if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()


def neutralize_atoms(smi):
    """
    Neutralize charges in a molecule.
    
    Parameters
    ----------
    smi : str
        Input SMILES string
    
    Returns
    -------
    neutral_smi : str
        SMILES with neutralized atoms
    """
    mol = Chem.MolFromSmiles(smi)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return Chem.MolToSmiles(mol)


def get_props(smiles, prop_list):
    """
    Calculate molecular properties for a SMILES.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    prop_list : list
        List of RDKit descriptor names
    
    Returns
    -------
    props : torch.Tensor
        Property values as float tensor
    """
    if 'QED' in prop_list and prop_list[-1] != 'QED':
        raise ValueError('QED must be last in the property list!')

    props_no_qed = [p for p in prop_list if p != 'QED']
    calc = MolecularDescriptorCalculator(props_no_qed)
    props = [x for x in calc.CalcDescriptors(Chem.MolFromSmiles(smiles))]

    if 'QED' in prop_list:
        qed = Descriptors.qed(Chem.MolFromSmiles(smiles))
        props.append(qed)

    return torch.nan_to_num(torch.tensor(props), 0.0).float()


def get_atom_cnts(smileses):
    """
    Count atom types across a list of SMILES.
    
    Parameters
    ----------
    smileses : list
        List of SMILES strings
    
    Returns
    -------
    atom_to_cnt : dict
        Dictionary mapping atom symbols to counts
    """
    atom_to_cnt = {}

    for smiles in smileses:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        for atom in mol.GetAtoms():
            atom_type = atom.GetSymbol()
            atom_to_cnt[atom_type] = atom_to_cnt.get(atom_type, 0) + 1

    return atom_to_cnt


def show_atom_index(mol):
    """
    Create a copy of molecule with atom indices as map numbers.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object
    
    Returns
    -------
    new_mol : rdkit.Chem.Mol
        Molecule with atom map numbers set to indices
    """
    import copy
    atoms = mol.GetNumAtoms()
    new_mol = copy.deepcopy(mol)
    for idx in range(atoms):
        new_mol.GetAtomWithIdx(idx).SetProp(
            'molAtomMapNumber',
            str(mol.GetAtomWithIdx(idx).GetIdx())
        )
    return new_mol
