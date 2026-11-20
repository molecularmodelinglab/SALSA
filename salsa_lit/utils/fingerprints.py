"""
Molecular fingerprint utility functions.
"""

import numpy as np
from scipy import sparse
from joblib import Parallel, delayed

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFpGen


def get_fp(smiles, fp_type='morgan', counts=False, bits=1024,
           radius=2, chiral=False, sparsed=False):
    """
    Compute molecular fingerprint for a SMILES string.
    
    Parameters
    ----------
    smiles : str
        SMILES string
    fp_type : str
        Fingerprint type: 'morgan', 'fcfp', 'rdkit', 'atom_pair'
    counts : bool
        Use count vector instead of bit vector
    bits : int
        Length of fingerprint vector
    radius : int
        Radius for circular fingerprints
    chiral : bool
        Include chirality information
    sparsed : bool
        Return sparse matrix
    
    Returns
    -------
    fp : np.ndarray or sparse matrix or False
        Fingerprint vector, or False if invalid SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False

    gen = None

    # Extended-connectivity fingerprint (Topological circular)
    if fp_type == 'morgan':
        gen = rdFpGen.GetMorganGenerator(
            radius=radius, fpSize=bits, includeChirality=chiral
        )
    # Functional class fingerprint (Topological circular)
    elif fp_type == 'fcfp':
        gen = rdFpGen.GetMorganGenerator(
            radius=radius, fpSize=bits, includeChirality=chiral,
            atomInvariantsGenerator=rdFpGen.GetMorganFeatureAtomInvGen()
        )
    # Daylight-esque fingerprint (Topological path-based)
    elif fp_type == 'rdkit':
        gen = rdFpGen.GetRDKitFPGenerator(
            maxPath=radius * 2, fpSize=bits, numBitsPerFeature=2
        )
    # Atom pair fingerprint
    elif fp_type == 'atom_pair':
        gen = rdFpGen.GetAtomPairGenerator(
            maxDistance=radius, fpSize=bits, includeChirality=chiral
        )
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")

    if counts:
        _fp = gen.GetCountFingerprint(mol)
    else:
        _fp = gen.GetFingerprint(mol)

    fp = np.zeros(bits, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(_fp, fp)

    if sparsed:
        return sparse.csr_matrix(fp)
    else:
        return fp


def get_fps_in_parallel(smiles_list, fp_type='morgan', counts=False, bits=1024,
                        radius=2, chiral=False, sparsed=False, n_jobs=-1):
    """
    Compute fingerprints in parallel for a list of SMILES.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings
    fp_type : str
        Fingerprint type
    counts : bool
        Use count vector
    bits : int
        Fingerprint length
    radius : int
        Circular fingerprint radius
    chiral : bool
        Include chirality
    sparsed : bool
        Return sparse matrices
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    
    Returns
    -------
    fps : np.ndarray
        Stacked fingerprint array, shape (n_molecules, bits)
    """
    parallelizer = Parallel(n_jobs=n_jobs, backend='multiprocessing')
    fp_tasks = (
        delayed(get_fp)(sm, fp_type, counts, bits, radius, chiral, sparsed)
        for sm in smiles_list
    )
    fps = parallelizer(fp_tasks)
    fps = np.vstack(fps)
    return fps


def get_normed_fps(smiles_list, counts=False, bits=1024, radius=2):
    """
    Compute normalized Morgan fingerprints in parallel.
    
    Parameters
    ----------
    smiles_list : list
        List of SMILES strings
    counts : bool
        Use count vector
    bits : int
        Fingerprint length
    radius : int
        Morgan fingerprint radius
    
    Returns
    -------
    fps : np.ndarray
        L2-normalized fingerprints, shape (n_molecules, bits)
    """
    fps = get_fps_in_parallel(
        smiles_list, 'morgan', counts=counts, bits=bits, radius=radius
    )
    fps = np.stack(fps)
    norm = np.linalg.norm(fps, ord=2, axis=1)[:, np.newaxis]
    fps = fps / norm
    return fps
