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

import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

#from salsa.constants import MAX_MUT_LEN, MAHALA_THRESH, ACTIONS, WEIGHT
#from salsa.data_mutation import _get_muts_from_anc

from constants import MAX_MUT_LEN, MAHALA_THRESH, ACTIONS, WEIGHT
from data_mutation import _get_muts_from_anc

#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Functions to get SUPER (extended) mutants ! ! ! ! # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

sup_cols = ['Id','Anc_id','Src_id','Depth', 'Smiles',
            'Is_anc','Action','At_atom','To_atom','Prop_dist']

# Super mutation can only operate on a df having these attributes:
#  Id: Unique compound id
#  Is_anc: True if compound is from anchor set
#  Anc_id: Unique anchor id. (depth=0)
#  Src_id: Given a mutant, this is the id of the parent compound (depth=n-1) 
#  Depth: Current GED depth (n) 


import glob
def consolidate_supmut_chunks(work_path, out_path, tag=None):
    '''
    This function ... 
    Args: ...
    Output: csv_name
    '''   
    
    if tag is None:
        tag = datetime.today().strftime('%Y%m%d%H%M') 
    if out_path is None:
        out_path = f'../data/processed/scratch/{tag}_supermutants/'
    os.makedirs(f'{out_path}', exist_ok=True)
    csv_name = f'{out_path}supermutants_{tag}.csv'
    
    print(f"Consolidating supermutant dataset from path '{work_path}..'.")    
    print(f" > Saving final dataset to {csv_name}.") 
    
    csv_names = f'{work_path}/*.csv'
    csv_files = glob.glob(csv_names) 
    
    df_list = []
    for fname in csv_files:
        subdf = pd.read_csv(fname, names=sup_cols)
        df_list.append(subdf)
        
    df = pd.concat(df_list, axis=0, ignore_index=True)
    # df = df[df.Is_anc!='Is_anc']
    # df.drop_duplicates(keep=False, inplace=True)
    
    # Get rid of mixtures that may have crept in.
    # display(df)
    # to_drop = []
    # for i,row in df.iterrows():
        # if '.' in row.Smiles:
            # to_drop.append(i)
            # print("hi")
    # df = df.drop(index=to_drop)
    # display(df[~df['Smiles'].str.contains('.')])
    # return df, df
    # display(df)

    df = df.sort_values(by=['Anc_id','Is_anc'], ascending=[True,False])
    # df = df.sort_values('Id')
    df.reset_index(inplace=True, drop=True)

    id_implicit = df.index.values.tolist()
    id_true = df['Id'].values.tolist()
    if id_true != id_implicit:
        print("Warning: DataFrame index does not match IDs!")
#         display(df)
#         raise Exception("DataFrame index does not match IDs!")

    df.to_csv(csv_name, index=False, na_rep='None')
    
    return (df, csv_name)



def parallel_write_supmut_chunks(df_path, n_samp=-1, copies=1, work_path=None, tag=None, 
                                 depth=1, actions=ACTIONS, mahala_thresh=MAHALA_THRESH, weight=WEIGHT):
    '''
    This function ... 
    Args: ...
    Output: ...
    ''' 
    
    ## Read in source (anc) df.
    _df = pd.read_csv(df_path) #df = df.copy()
    
    if n_samp==-1:
        n_samp = len(_df)
    df = _df.sample(n_samp) ## This effectively shuffles the df !!!

    ## If you want multiple mutants at each depth ... 
    if copies > 1:
        dfs = [df.copy() for _ in range(copies)]
        df = pd.concat(dfs)

    df.reset_index(inplace=True,drop=True)
    
    ## Format df.
    df = _format_sup_mut_ds(df) 
    
    ## Prep dir paths.
    if tag is None:
        # tag = datetime.today().strftime('%Y%m%d-%Hh%Mm')
        acts = ''.join(actions) 
        dt = datetime.today().strftime('%Y%m%d%H%M') 
        tag = f'{dt}-{n_samp}anc-depth{depth}-{acts}-{mahala_thresh}-w{weight}'.replace('.','pt')
    if work_path is None:
        work_path = f'../data/processed/scratch/{tag}/'
    os.makedirs(f'{work_path}', exist_ok=True)
        
    ## Compute 'chunk' indices for multiprocessing.
    num_per_thread = len(df)//(multiprocessing.cpu_count()) 
    
    if num_per_thread == 0:
        raise Exception("Sample of anchors too small for multiprocessing!")
    partitions = np.arange(0, len(df), num_per_thread)
    partitions[-1] = len(df)+1
    idc_chunks = [[s,e] for s,e in zip(partitions, partitions[1:])]    
    
    print(f"Supermutating dataset of {len(df)} compounds.")
    print(f" > Parallel jobs: {len(idc_chunks)}\n > Compounds per job: {num_per_thread}")
    print(f" > Saving intermediate sets to path '{work_path}chunk_<s>-<e>.csv'")
       
    ## Run in parallel ... 
    parallelizer = Parallel(n_jobs=multiprocessing.cpu_count(), backend= 'multiprocessing' )  
    # print(actions, mahala_thresh)  
    tasks = (delayed(_writing_supmut_ds_chunk)(s, e, df, work_path, 
                                               n_ancs=len(df), 
                                               depth=depth,
                                               actions=actions, 
                                               mahala_thresh=mahala_thresh,
                                               weight=weight) for s,e in idc_chunks)
    parallelizer(tasks)
    
    return work_path, tag



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
    df['To_atom'] = None
    df = df[sup_cols]
    return df



def _writing_supmut_ds_chunk(s, e, df, work_path, n_ancs, depth,
                             actions=ACTIONS, mahala_thresh=MAHALA_THRESH, weight=WEIGHT):
    '''
    This function ... 
    Args: ...
    Output: ...
    '''   
    
    df = df[s:e]
        
    out_fname = f'{work_path}chunk_{s}-{e}.csv'
    with open(out_fname,'w', newline='') as ofile:
        
        writer = csv.writer(ofile, delimiter=',')        
        for out in _yield_supmut_df_per_depth(df, n_ancs, depth, use_tqdm=False,
                                              actions=actions, mahala_thresh=mahala_thresh,
                                              weight=weight):
            rows = out.values
            writer.writerows(rows)
            
            ofile.flush()
            


def _yield_supmut_df_per_depth(anc_df, n_ancs, depth, use_tqdm=False,
                               actions=ACTIONS, mahala_thresh=MAHALA_THRESH,
                               weight=WEIGHT):
    '''
    This function ... 
    Args: ...
    Output: ...
    '''    
    
    src_df = anc_df.copy()
    
    for i in tqdm(range(len(src_df)), total=len(src_df), disable=not use_tqdm):
        
        src_row = src_df.iloc[i]
        
        supermut_branch = [src_row] ## List of [anc, mut_ged1, mut_ged2, mut_ged3, ...]
        smis_chain = [src_row.Smiles]
        
        step = 1
        full_branch = True
        

        while step <= depth:            
            mut_row = _get_supmut_row_from_src(src_row, step, pop_size=n_ancs,
                                               actions=actions, mahala_thresh=mahala_thresh,
                                               weight=weight)  
            
            ## Do NOT return branch if it dies !!!
            if len(mut_row) == 0:
                full_branch = False
                break 

            mut_row = mut_row.iloc[0]
            if mut_row.Smiles in smis_chain:
                # yield None, ['x'] + smis_chain + [mut_row.Smiles]
                full_branch = False
                break 

            supermut_branch.append(mut_row)
            smis_chain.append(mut_row.Smiles)

            src_row = mut_row
            step += 1    

        # print(supermut_branch)   
        if full_branch:
            supermut_df = pd.concat(supermut_branch, axis=1).T
            # return supermut_df, smis_chain
            yield supermut_df #, smis_chain
        
from salsa.data_mutation import _get_muts_from_anc
def _get_supmut_row_from_src(src_row, step, pop_size, 
                             actions=ACTIONS, mahala_thresh=MAHALA_THRESH, weight=WEIGHT):   
    '''
    This function gets one mutant row item from a src ('anc') row item.
    Args: 
        src_row: Source ('anchor') smiles.
        step: Depth in mutation 'tree'.
        pop_size: ...
        actions: ...
        mahala_thresh: ...
    Output:
        mut_row: DataFrame object having one (1) row.
    '''    

    src_smi = src_row['Smiles']
    ## Only doing 1 mut per anc !!!!! (num_muts=1)
    mut_tup, _, _ = _get_muts_from_anc(src_smi, max_mut_len=MAX_MUT_LEN, num_muts=1,
                                       actions=actions, mahala_thresh=mahala_thresh,
                                       weight=weight)
    
    # get_muts_from_anc(anc_smi, max_mut_len=MAX_MUT_LEN, num_muts=N_MUTS, 
    #                   actions=ACTIONS, mahala_thresh=MAHALA_THRESH, 
    #                   weight=WEIGHT)
    
    
    if len(mut_tup) < 1:
        return []
    
    mut_row = pd.DataFrame(mut_tup,columns=['Smiles','Action','At_atom','To_atom','Prop_dist'])
    
    mut_row['Is_anc'] = False
    mut_row['Anc_id'] = src_row['Anc_id']
    mut_row['Src_id'] = src_row['Id']
    mut_row['Depth'] = step
    mut_row['Id'] = mut_row['Anc_id'] + (step*pop_size)
    mut_row = mut_row[sup_cols]
    
    return mut_row           
        
        
        
             
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





# def _yield_supmut_df_per_depth(anc_df, n_ancs, depth, use_tqdm=False,
#                                actions=ACTIONS, mahala_thresh=MAHALA_THRESH):
#     '''
#     This function ... 
#     Args: ...
#     Output: ...
#     '''    
    
#     src_df = anc_df.copy()
    
#     step = 1
#     while step <= depth:
        
#         muts_at_step = []
#         for i in tqdm(range(len(src_df)), total=len(src_df), disable=not use_tqdm):
#             row = src_df.iloc[i]
#             muts_df = _get_supmut_ser_from_src(row, step, pop_size=n_ancs, #   pop_size=len(anc_df), 
#                                                actions=actions, mahala_thresh=mahala_thresh)
#             if len(muts_df) == 0:
#                 continue
#             else:
#                 muts_at_step.append(muts_df)
#                 yield muts_df
            
#         src_df = pd.concat(muts_at_step)
#         step += 1





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# OLD 
# #
# def _yield_supmut_df_per_depth(anc_df, n_ancs, depth, use_tqdm=False,
#                                actions=ACTIONS, mahala_thresh=MAHALA_THRESH):
#     '''
#     This function ... 
#     Args: ...
#     Output: ...
#     '''    
    
#     src_df = anc_df.copy()
    
#     step = 1
#     while step <= depth:
        
#         muts_at_step = []
#         for i in tqdm(range(len(src_df)), total=len(src_df), disable=not use_tqdm):
#             row = src_df.iloc[i]
#             muts_df = _get_supmut_row_from_src(row, step, pop_size=n_ancs, #   pop_size=len(anc_df), 
#                                                actions=actions, mahala_thresh=mahala_thresh)
#             if len(muts_df) == 0:
#                 continue
#             else:
#                 muts_at_step.append(muts_df)
#                 yield muts_df
            
#         src_df = pd.concat(muts_at_step)
#         step += 1            
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #