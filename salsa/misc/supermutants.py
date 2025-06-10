'''
Start date: Monday, Apr. 3, 2023
'''
#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

import pandas as pd
import csv
import numpy as np
import os
import sys
sys.path.insert(0, '..')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Writing of SUPER mutant series. # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

COLS = ['Id','Anc_id','Src_id','Depth', 'Smiles',
       'Is_anc','Action','At_atom','Prop_dist']

import copy
from rdkit import Chem
def format_sup_mut_ds(df_in):
    '''
    This function simply constructs a dataframe. This is because the super 
    mutation function can only operate on a df having these attributes:
        Id: Unique compound id
        Is_anc: True if compound is from anchor set
        Anc_id: Unique anchor id. (depth=0)
        Src_id: Given a mutant, this is the id of the parent compound (depth=n-1) 
        Depth: Current GED depth (n) 
    '''
    df = copy.deepcopy(df_in)
    # Reset index to generate anc ids starting at 0.
    df.reset_index(inplace=True, drop=True)
    df['Id'] = df.index
    # Initial ds is all anchors, having ids of idx values.
    df['Is_anc'] = True
    df['Anc_id'] = df['Id']
    df['Src_id'] = df['Id']
    df['Depth'] = 0
    df['Prop_dist'] = 0    
    df['Action'] = None
    df['At_atom'] = None
    df = df[COLS]
    return df


from salsa.data_mutation import _get_muts_from_anc
def get_mut_ser_from_src(src_row, step, pop_size, actions=['Add','Remove','Replace'], 
                         mahala_thresh=6.75, num_muts=1):
    src_smi = src_row['Smiles']
    mut_tups, neg_tups, _ = _get_muts_from_anc(src_smi, actions=actions,
                                               mahala_thresh=mahala_thresh, num_muts=num_muts)
    
    if len(mut_tups) < 1:
        return []
    
    muts_df = pd.DataFrame(mut_tups,columns=['Smiles','Action','At_atom','Prop_dist'])
    muts_df['Is_anc'] = False
    muts_df['Anc_id'] = src_row['Anc_id']
    muts_df['Src_id'] = src_row['Id']
    muts_df['Depth'] = step
    muts_df['Id'] = muts_df['Anc_id'] + (step*pop_size)
    muts_df = muts_df[COLS]
    return muts_df






def write_expand_super_muts(idx_s, idx_e, anc_df, ds_name='chembl_05',
                      split='super_mutants',depth=5):
    
    anc_df = anc_df[idx_s:idx_e].reset_index(drop=True)

    src_df = anc_df
    step = 1
    df_list = [anc_df]
    
    while step <= depth:
        
        muts_at_step = []
#         for i in tqdm(range(len(src_df)), total=len(src_df)):
        for i in range(len(src_df)):
            row = src_df.iloc[i]
            muts_df = get_mut_ser_from_src(row, step, pop_size=len(anc_df))
            if len(muts_df) == 0:
                continue
            else:
                muts_at_step.append(muts_df)
            
        src_df = pd.concat(muts_at_step)
        df_list.append(src_df)
        step += 1
        
    full_df = pd.concat(df_list)
    
    ddir = '../data/processed/'
    fname = f'{ddir}/{ds_name}/{split}/chunk_{idx_s}-{idx_e}.csv'
    full_df.to_csv(fname,index=None)
#     return full_df



def _write_expand_super_muts(idx_s, idx_e, anc_df, ds_name='chembl_05',
                      split='super_mutants',depth=5):
    
    anc_df = anc_df[idx_s:idx_e].reset_index(drop=True)

    src_df = anc_df
    step = 1
    df_list = [anc_df]
    
    while step <= depth:
        
        muts_at_step = []
#         for i in tqdm(range(len(src_df)), total=len(src_df)):
        for i in range(len(src_df)):
            row = src_df.iloc[i]
            muts_df = get_mut_ser_from_src(row, step, pop_size=len(anc_df))
            if len(muts_df) == 0:
                continue
            else:
                muts_at_step.append(muts_df)
            
        src_df = pd.concat(muts_at_step)
        df_list.append(src_df)
        step += 1
        
    full_df = pd.concat(df_list)
    
    ddir = '../data/processed/'
    fname = f'{ddir}/{ds_name}/{split}/chunk_{idx_s}-{idx_e}.csv'
    full_df.to_csv(fname,index=None)
#     return full_df




def _expand_super_muts(anc_df, depth=5):
    
    src_df = anc_df
    step = 1
    df_list = [anc_df]
    
    while step <= depth:
        
        muts_at_step = []
        for i in tqdm(range(len(src_df)), total=len(src_df)):
            row = src_df.iloc[i]
            muts_df = get_mut_ser_from_src(row, step, pop_size=len(anc_df))
            if len(muts_df) == 0:
                continue
            else:
                muts_at_step.append(muts_df)
            
        src_df = pd.concat(muts_at_step)
        df_list.append(src_df)
        step += 1
        
    full_df = pd.concat(df_list)
    return full_df