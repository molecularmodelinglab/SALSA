'''
Start date: Monday, Mar. 27, 2023
'''
#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

import pandas as pd
import csv
import numpy as np
import os
import sys
sys.path.insert(0, '..')

from salsa.utils import get_props, get_cansmiles
from salsa.constants import MAHALA_THRESH, N_MUTS, MAX_ANC_LEN, MAX_MUT_LEN, ACTIONS
from salsa.constants import ATOMS, PROP_NAMES, INV_COV_CHEMBL

#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions to get SUPER (extended) mutants ! ! ! ! # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

sup_cols = ['Id','Anc_id','Src_id','Depth', 'Smiles',
            'Is_anc','Action','At_atom','Prop_dist']

# This is because the super 
#     mutation function can only operate on a df having these attributes:
#         Id: Unique compound id
#         Is_anc: True if compound is from anchor set
#         Anc_id: Unique anchor id. (depth=0)
#         Src_id: Given a mutant, this is the id of the parent compound (depth=n-1) 
#         Depth: Current GED depth (n) 

import copy
from rdkit import Chem
def _format_sup_mut_ds(df_in):
    '''
    This function simply constructs a dataframe. 
    Args: 
        df_in: Input dataframe ... MUST have a 'Smiles' column !!!!      
    Output:
        df: Properly formatted supermutant dataframe with required attributes. 
    '''
    
    df = df_in.copy() #copy.deepcopy(df_in)
    ## Reset index to generate anc ids starting at 0.
    df.reset_index(inplace=True, drop=True)
    df['Id'] = df.index
    ## Initial ds is all anchors, having ids of idx values.
    df['Is_anc'] = True
    df['Anc_id'] = df['Id']
    df['Src_id'] = df['Id']
    df['Depth'] = 0
    df['Prop_dist'] = 0    
    df['Action'] = None
    df['At_atom'] = None
    df = df[sup_cols]
    return df


def _yield_supmut_df_per_depth(anc_df, n_ancs, depth, use_tqdm=False,
                               actions=ACTIONS, mahala_thresh=MAHALA_THRESH):
    '''
    This function ... 
    Args: ...
    Output: ...
    '''    
    
    src_df = anc_df.copy()
    
    step = 1
    while step <= depth:
        
        muts_at_step = []
        for i in tqdm(range(len(src_df)), total=len(src_df), disable=not use_tqdm):
            row = src_df.iloc[i]
            muts_df = _get_supmut_ser_from_src(row, step, pop_size=n_ancs, #   pop_size=len(anc_df), 
                                               actions=actions, mahala_thresh=mahala_thresh)
            if len(muts_df) == 0:
                continue
            else:
                muts_at_step.append(muts_df)
                yield muts_df
            
        src_df = pd.concat(muts_at_step)
        step += 1
        

def _get_supmut_ser_from_src(src_row, step, pop_size, 
                             actions=ACTIONS, mahala_thresh=MAHALA_THRESH):   
    '''
    This function gets one series of mutants from a src ('anc') row item.
    Args: 
        src_row: Source ('anchor') smiles.
        step: Depth in mutation 'tree'.
        pop_size: ...
        actions: ...
        mahala_thresh: ...
        num_muts: Num of mutants to generate per src ('anc') mol at each depth.
    Output:
        ...
    '''    
    
    src_smi = src_row['Smiles']
    ## Only doing 1 mut per anc !!!!! (num_muts=1)
    mut_tups, _, _ = get_muts_from_anc(src_smi, num_muts=1, actions=actions,  
                                       mahala_thresh=mahala_thresh)
    
    if len(mut_tups) < 1:
        return []
    
    muts_df = pd.DataFrame(mut_tups,columns=['Smiles','Action','At_atom','Prop_dist'])
    muts_df['Is_anc'] = False
    muts_df['Anc_id'] = src_row['Anc_id']
    muts_df['Src_id'] = src_row['Id']
    muts_df['Depth'] = step
    muts_df['Id'] = muts_df['Anc_id'] + (step*pop_size)
    muts_df = muts_df[sup_cols]
    
    return muts_df



#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions to get 'deterministic' mutants ! ! ! !  # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Final wrapper function
def get_muts_from_anc(anc_smi, max_mut_len=MAX_MUT_LEN, num_muts=N_MUTS, 
                      actions=ACTIONS, mahala_thresh=MAHALA_THRESH): 
    '''
    This function does the following: 
    For a single anchor smiles ...
    1) Makes a call to get the pool of mutants for the anchor.
    2) Makes a call to sample n mutants (num_muts) from the entire pool. 
    
    Args: 
        anc_smi: Anchor smiles.
        mahala_thresh: Whether or not to filter out mutants that are a pre-determined
            Mahalanobis prop distance from the anchor. 
            For no filter, use 'None'. 
            For no filter but record Mahala, use 'Record'. 
            For filter, use threshold value (e.g. 6.75). May be string or float.
        num_muts: Number of mutants to generate from the given anchor.
        max_mut_len: ...
        actions: The set of actions to be performed. Can be any combination of the
            superset of (add, replace, remove).        
    Output:
        ...
    '''    

    # make sure provided actions are valid
    valid_actions = set(ACTIONS)
    input_actions = set(actions)
    if len(input_actions - valid_actions) > 0:
        raise Exception(f"Invalid actions! (Valid actions: {ACTIONS})")
        
    # convert mahala_thresh from str to proper value
    if type(mahala_thresh) is str:
        if mahala_thresh=='None':
            mahala_thresh = None
        if mahala_thresh=='Record':
            mahala_thresh = np.inf
        else:
            mahala_thresh = float(mahala_thresh)
    
    muts_full = _get_mut_pool_for_anc(anc_smi, actions, max_mut_len)
    muts_samp = _sample_muts(muts_full, anc_smi, num_muts, actions, mahala_thresh)
    return muts_samp


import copy
def _get_mut_pool_for_anc(anc_smi, actions=ACTIONS, max_mut_len=MAX_MUT_LEN):
    '''
    This function iterates through the atoms and the set of mutation 
    actions to produce all possible mutants for a given anchor smiles.
    Args: 
        anc_smi: Anchor smiles.
        actions: The set of mutations to be performed. Can be any combination 
        of the superset of (add, replace, remove).
    Output:
        ...   
    '''
    
    mol = Chem.MolFromSmiles(anc_smi)

    # Apply mutate actions to every atom in the anchor molecule.
    atom_idc = [i for i in range(0,(mol.GetNumAtoms()))]
    mut_tups = []
    for i in atom_idc:       
        for action_type in actions:
            if action_type=='Remove':
                mut_tup = _generate_mut_tup(mol, action_type, i, max_mut_len)
                mut_tups.append(mut_tup)
            else:
            # If mutate action is based on an incoming atom (replace with new atom,
            # add new atom), then loop through all possible atom types. 
                for atom_type in ATOMS:
                    mut_tup = _generate_mut_tup(mol, action_type, i, atom_type, max_mut_len)
                    mut_tups.append(mut_tup)
    # Do not return 'Null' mutant objects 
    valid_mut_tups = [x for x in mut_tups if x]
    return valid_mut_tups


import random
from scipy.spatial import distance
def _sample_muts(muts, anc_smi, num_muts=N_MUTS, actions=ACTIONS, mahala_thresh=None):
    '''
    This function ...
    
    Args: 
        muts: Pool of mutants from which to sample.
        actions: Which mutation actions to sample. 
        mahala_thresh: If a value is provided, property filtering 
            will be applied. That is, mutants will be disgarded if 
            their Mahalanobis property distance (from their anchor) 
            is greater than the provided threshold. 
                Valid args: None, float (e.g. 6.75), np.inf
        Mahalanobis prop distance from the anchor.
        num_muts: Number of mutants to sample for the anchor, from the pool.
            If None, then return ALL muts. ...(i think??) ... 
            i think this is what the ToDo below was trying to get at ... 
    Output:
        ...
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    TODO: Add functionality to return all mutants. num_muts = 'all'?
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    '''
    
    # calculate properties for anc if mahala is -1 or a float.
    if mahala_thresh:
        anc_prop = get_props(anc_smi, prop_list=PROP_NAMES)

    actions = np.array(actions)
    act_subsets = {act:[x for x in muts if x[1]==act] for act in actions}
    
    muts_samp = []
    mut_smis = []
    neg_muts = []
    
    while True:
        if num_muts is None: 
            pass
        elif type(num_muts)==int: 
            if len(muts_samp) >= num_muts: break
        else:
            raise Exception (f"Requested num_muts is not valid. (num_muts = {num_muts})")
        
        muts_remaining = sum([len(v) for v in act_subsets.values()])
        if muts_remaining==0:
            break
            
        action = random.choice(actions)
        muts_subset = act_subsets[action]
        if not muts_subset: continue
            
        random.shuffle(muts_subset)
        mut_draw = muts_subset.pop()        
        mut_smi, mut_type, i = mut_draw
        
        if mut_smi in mut_smis: continue

        # Mahalanobis distance filtering
        if mahala_thresh:  
            mut_prop = get_props(mut_smi, prop_list=PROP_NAMES)
            d = distance.mahalanobis(anc_prop, mut_prop, VI=INV_COV_CHEMBL)
            if d <= mahala_thresh:
                muts_samp.append([mut_smi, mut_type, i, round(d,2)])
                mut_smis.append(mut_smi)  
            else:
                neg_muts.append([mut_smi, mut_type, i, round(d,2)])
        else:
            muts_samp.append([mut_smi, mut_type, i])
            mut_smis.append(mut_smi)                     
                
    if num_muts:
        while len(muts_samp) < num_muts:
            try:
                muts_samp.append( random.choice(muts_samp) )
            except Exception as e:
#                 print (e)
                return ([], [], True)     

    return (muts_samp, neg_muts, False)




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Mutation actions (and wrapper) ! ! ! !  # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from rdkit import RDLogger                                                  
RDLogger.DisableLog('rdApp.*')  

def _generate_mut_tup(anc_mol, action_type, i, atom_type=False, max_mut_len=MAX_MUT_LEN):
    '''
    This function wraps the three different mutate actions. Will perform
    whichever is specified on the provided anchor mol. Generates a
    mutation tuple (mutated smiles, action type, atom index)
    Args: 
        anc_mol: Anchor molecule.
        action_type: 
        i:
        atom_type:
    Output:
        ...
    '''
    
    mol = copy.copy(anc_mol)
    for atom in mol.GetAtoms():
        atom.SetProp("State","Anc") # Set 'state' to track augs.
        
    try:
        if action_type=='Remove':
            mut_mol = remove_atom_from_mol(mol, i)
        elif action_type=='Add':
            mut_mol = add_atom_to_mol(mol, atom_type, i)
        elif action_type=='Replace':
            curr_atype = mol.GetAtomWithIdx(i).GetSymbol()
            if atom_type==curr_atype: return
            else: mut_mol = replace_atom_on_mol(mol, atom_type, i)
    except Exception as e:
        return
            
    # return none if mol is invalid
    if mut_mol.GetNumAtoms()==0: 
        return 
    # return none if smiles is >= max len
    elif len(Chem.MolToSmiles(mut_mol)) > max_mut_len:
        return
    # else, return mut tuple
    else:
        mut_tup = [Chem.MolToSmiles(mut_mol), action_type, i]
        return mut_tup
    

from rdkit import Chem
from rdkit.Chem.rdchem import RWMol, Atom, GetPeriodicTable, Bond, BondType

def add_atom_to_mol(mol, atom_type, to_aidx, clean_aroms=True, remake_mol=True): 
    molgraph = get_rwmol(mol) # molgraph.UpdatePropertyCache()
    update_mol_rep(molgraph, clean_aroms)
    atom = Atom(atom_type)
    atom.SetBoolProp("mutability", True)
    atom.SetProp('State','Aug') # set state to "Aug"
    molgraph.AddAtom(atom)        
    molgraph.AddBond(molgraph.GetNumAtoms() - 1, to_aidx, BondType.SINGLE)
    update_mol_rep(molgraph, clean_aroms)
    mol_new = molgraph.GetMol() 
    if remake_mol:
        mol_new = Chem.MolFromSmiles( get_cansmiles( Chem.MolToSmiles(mol_new) ))
    return mol_new

def replace_atom_on_mol(mol, atom_type, to_aidx, clean_aroms=True, remake_mol=True):
    molgraph = get_rwmol(mol) # molgraph.UpdatePropertyCache()
    update_mol_rep(molgraph, clean_aroms)
    molgraph.GetAtomWithIdx(to_aidx).SetAtomicNum(GetPeriodicTable().GetAtomicNumber(atom_type))
    molgraph.GetAtomWithIdx(to_aidx).SetFormalCharge(0)
    molgraph.GetAtomWithIdx(to_aidx).SetProp('State','Aug') # set state to "Aug"
    update_mol_rep(molgraph, clean_aroms) 
    mol_new = molgraph.GetMol() 
    if remake_mol:
        mol_new = Chem.MolFromSmiles( get_cansmiles( Chem.MolToSmiles(mol_new) ))
    return mol_new
    
def remove_atom_from_mol(mol, aidx, clean_aroms=True, remake_mol=True):
    num_bonds = len(mol.GetAtomWithIdx(aidx).GetNeighbors())
    if num_bonds != 1: 
        return Chem.Mol()
    molgraph = get_rwmol(mol) # molgraph.UpdatePropertyCache()
    update_mol_rep(molgraph, clean_aroms)
    molgraph.RemoveAtom(aidx)
    update_mol_rep(molgraph, clean_aroms) 
    mol_new = molgraph.GetMol() 
    if remake_mol:
        mol_new = Chem.MolFromSmiles( get_cansmiles( Chem.MolToSmiles(mol_new) ))
    return mol_new




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Accessory functions for mutation actions ! ! ! !  # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
# Inspo taken from:
# github.com/wengong-jin/hgraph2graph/blob/master/hgraph/chemutils.py

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def get_rwmol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
        new_atom.SetProp("State",atom.GetProp("State"))
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

from rdkit.Chem.rdmolops import SanitizeMol
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
    #Chem.FastFindRings(molgraph)
    


    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Writing of mutant sets. # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import glob
def consolidate_subdfs(version='chembl_05'):
    '''
    This function writes out the final mutated dataset by
    combining the chunks written in parallel by write_mut_ds_chunk.
    The only argument is version which is defaul chembl_05 
    (the version Salsa was trained on).
    '''    
    
    csv_names = f'../data/processed/{version}/full/*.csv'
    csv_files = glob.glob(csv_names)
    print(f"Consolidating {len(csv_files)} mutated dfs ...")

    res = []
    for fname in csv_files:
        subdf = pd.read_csv(fname)
        res.append(subdf)
    df = pd.concat(res, axis=0, ignore_index=True)
    df = df.sort_values(by=['Anc_id','Is_anc'], ascending=[True,False])
    
    os.makedirs(f'../data/processed/{version}/train/',exist_ok=True)
    df.to_csv(f'../data/processed/{version}/train/full.csv',index=False)



def write_mut_ds_chunk(idx_s, idx_e, df, ds_name, split):
    '''
    This function writes out the mutants for a given data (given by df) 
    subset (determined by idx_s and idx_e).
    
    Args: 
        idx_s: position of first anchor in the partitioned chunk
        idx_e: position-1 of last anchor in the partitioned chunk
    Output:
        Writes each chunk to a separate csv file.
        Additionally, it also saves the 'hard negatives' which are mutants
        that had a Mahalanobis distance above the threshold, as well as the 
        Smiles for which no valid mutants could be generated.
    '''
    
    df = df[idx_s:idx_e].reset_index(drop=True)
    
    ddir = '../data/processed/'
    mut_fname = f'{ddir}/{ds_name}/{split}/chunk_{idx_s}-{idx_e}.csv'
    neg_fname = f'{ddir}/{ds_name}/{split}/hard_negs/chunk_{idx_s}-{idx_e}.csv'
    bad_fname = f'{ddir}/{ds_name}/{split}/failed_smiles/chunk_{idx_s}-{idx_e}.csv'

    with open(mut_fname,'w', newline='') as mut_file, open(neg_fname,'w', newline='') as neg_file, \
    open(bad_fname,'w', newline='') as bad_file:
        mut_writer = csv.writer(mut_file, delimiter=',')
        mut_writer.writerow(['Is_anc','Anc_id','Smiles','Action','At_atom','Prop_dist'])
        neg_writer = csv.writer(neg_file, delimiter=',')
        neg_writer.writerow(['Is_anc','Anc_id','Smiles','Action','At_atom','Prop_dist'])
        bad_writer = csv.writer(bad_file, delimiter=',')
        bad_writer.writerow(['Anc_id','Smiles'])
        
        for i,row in enumerate(df.values):
            
            anc_id = row[0]
            anc_smi = row[1]
            
            mut_tups, neg_objs, failed = get_muts_from_anc( anc_smi )
            
            if failed:
                bad_writer.writerows([[anc_id, anc_smi]])
                
            else:
                anc = [True, anc_id, anc_smi, 'None', -1, -1]
                
                muts = [[False, anc_id, *x] for x in mut_tups]
                mut_rows = [anc] + muts
                mut_writer.writerows(mut_rows)

                negs = [[False, anc_id, *x] for x in neg_objs]
                neg_rows = [anc] + negs
                neg_writer.writerows(neg_rows)                
            
            


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MISC / LEGENDARY  # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def write_whole_supmut_ds_chunk(idx_s, idx_e, df_in, work_path, depth=5): #, tag):
    '''
    This function writes out the SUPER mutants for a given data (given by df) 
    subset (determined by idx_s and idx_e).
    Args: 
        idx_s: position of first anchor in the partitioned chunk
        idx_e: position-1 of last anchor in the partitioned chunk
        df: input dataframe !!!
        work_path:
        out_path: 
        tag:
    Output:
        Writes each chunk to a separate csv file.
    '''
    
    df = df_in[idx_s:idx_e].reset_index(drop=True)
    df_out = get_supmut_df(df, use_tqdm=False, depth=depth) 
    out_fname = f'{work_path}chunk_{idx_s}-{idx_e}.csv'
    df_out.to_csv(out_fname, index=None)
    

from tqdm import tqdm
def get_supmut_df(anc_df, use_tqdm=False, depth=5):
    '''
    This function returns a supermutant DataFrame. 
    Args: 
        anc_df: Source ('anchor') smiles.
        depth: Total depth of mutation 'tree'.
        num_muts: Num of mutants to generate per src ('anc') mol at each depth.
    Output:
        full_df: ...
    '''    
    
    src_df = anc_df.copy()
    step = 1
    df_list = [anc_df]
    
    while step <= depth:
        
        muts_at_step = []
        for i in tqdm(range(len(src_df)), total=len(src_df), disable=not use_tqdm):
#         for i in range(len(src_df)):
            row = src_df.iloc[i]
            muts_df = _get_mut_ser_from_src(row, step, pop_size=len(anc_df))
            if len(muts_df) == 0:
                continue
            else:
                muts_at_step.append(muts_df)
            
        src_df = pd.concat(muts_at_step)
        df_list.append(src_df)
        step += 1
        
    full_df = pd.concat(df_list)
    return full_df


