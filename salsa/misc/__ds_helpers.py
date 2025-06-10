import random
import numpy as np
import pandas as pd
import os
import csv
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '..')

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, Sampler
from typing import Iterator, List
import copy
from utils import get_cansmiles


def get_clique_map_implicit(df, n_augs):
    clique_idc = range(len(df)//(n_augs+1))
    clique_map = {i: np.arange((n_augs+1)*i,(n_augs+1)*(i+1)) for i in clique_idc}
    clique_map = {k:list(v) for k,v in clique_map.items()}
    return clique_map


def get_salsa_df(df_in, samp_size=0, max_len=150, keep_cols=False, reset_idc=True):
    df_out = df_in.rename({'SMILES':'Smiles', 'Anc_idx':'Clique_idx'}, axis=1,errors='ignore')

#     print(df_out.index)
    df_out["Index before drops"] = df_out.index
    
    if samp_size > 0:
        df_out = random.sample(df_out)

    # Reset index ... 
    #df_out.reset_index(drop=True, inplace=True)
    
    # Fill in missing columns.    
    for col in ['Smiles','Is_anc','Clique_idx']:
        if col not in df_out.columns:
            if col=='Is_anc':
                df_out[col] = True
            elif col=='Clique_idx':
#                 if 'Anc_idx' in df_out.columns:
#                     df_out[col] = df_out['Anc_idx']
                df_out[col] = df_in.index
#                 else:
#                     df_out[col] = df_in.index
            elif col=='Smiles':
                raise ValueError("No SMILES column in dataframe!")
                
    # canonicalize smiles
    # TODO: un-comment this !!!! rn it's hard coded to make SALSA training run faster !!!
#    df_out['Smiles'] = df_out.Smiles.apply(lambda x: get_cansmiles(x))
    
    if max_len:
        df_temp = df_out[df_out.Smiles.str.len() <= max_len]
        n_toss = len(df_out) - len(df_temp)
        print(f'Removed {n_toss} compounds having SMILES longer than {max_len}.')
        df_out = df_temp

    if keep_cols==False:
        df_out = df_out[['Smiles','Is_anc','Clique_idx']]
        
    '''
    if reset_idc==True:
        df_out = df_out.reset_index(drop=True)
        if 'Idx' in df_out.columns:
            df_out.Idx = df_out.index
    '''
            
    if 'Anc_idx' in df_out.columns:
        print('Dataframe contains Anc_idx attribute. Be careful!')
            
    # Reset index ...
    if reset_idc==True:
        df_out.reset_index(drop=True, inplace=True)
        
    return df_out
    
    
def get_df_and_clique_map(df_in, n_augs, anc_samp_size=0, max_len=150):
    # renaming to fix issues with anc_idx and clique_idx ... 
    df_out = df_in.rename({'SMILES':'Smiles','Anc_idx':'Clique_idx'},
                           axis=1,errors='ignore')
    if anc_samp_size > 0:
        idc = list(df_out.Clique_idx.unique())
        rand_idc = random.sample(idc,anc_samp_size)
        df_out = df_out[df_out.Clique_idx.isin(rand_idc)]
                
    # canonicalize smiles
# TODO .. uncomment this !!!! commented out to make salsa train faster.
#    df_out['Smiles'] = df_out.Smiles.apply(lambda x: get_cansmiles(x))
    
    if max_len:
        big_cliques = set(df_out[df_out.Smiles.str.len() > max_len].Clique_idx)
        if big_cliques:
            n_toss = len(big_cliques)
            print(f'Removed {n_toss} cliques containing SMILES longer than {max_len}.')
            df_out = df_out[~(df_out.Clique_idx.isin(big_cliques))]
    
    # TODO: keep track of dropped compounds !!! (?????)
    df_out.reset_index(drop=True, inplace=True)
    df_out['Index'] = df_out.index
    
    # Get clique map.
    clique_map = get_clique_map_implicit(df_out, n_augs)
    
    # Get idx-to-clique dict. If none, clique is index.
    idx_to_clique = {}
    for k,v in clique_map.items():
        for i in v:
            idx_to_clique[i] = k
    if 'Clique_idx' in df_out.columns:
        df_out['Clique_idx'] = df_out['Index'].apply(lambda x: idx_to_clique[x])
    
    # Fill in missing columns.
    for col in ['Smiles','Is_anc','Clique_idx']:
        if col not in df_out.columns:
            if col=='Is_anc':
                df_out[col] = True
            elif col=='Clique_idx':
                df_out[col] = df_in.index
            elif col=='Smiles':
                raise ValueError("No SMILES column in dataframe!")  
                
    # TODO: Why does this print twice ????
    print("Got df and clique map.")           
    return (df_out[['Smiles','Is_anc','Clique_idx']], clique_map)


def get_batching_df(df_in, sort=True):
    df_anc = df_in[df_in['Is_anc']==True]
    
    df_anc.rename({'Anc_idx':'Clique_idx'},axis=1,inplace=True,errors='ignore')
    df_anc = df_anc[['Clique_idx','Smiles']].sort_values('Clique_idx')

    lengths = df_anc.apply(lambda x: len(x['Smiles']),axis=1)
    tups = []
    for i in tqdm(range(len(df_anc)), total=len(df_anc),
                  desc=f"Constructing df_batching"):
        tups.append((lengths.iloc[i], df_anc.iloc[i]["Clique_idx"]))
    index = pd.MultiIndex.from_tuples(tups)

    index.set_names(['length','clique_idx'], inplace=True)
    df_anc.index = index

    df_anc["Sampled"] = False
    
    if sort: 
        df_anc = df_anc.sort_index(level='length', inplace=False, sort_remaining=True)
    return df_anc


def get_clique_batches(df, bs, flank, bounds=None):
    if bounds:
        s_idx, e_idx = bounds
        df_out = copy.deepcopy(df[s_idx:e_idx]) 
    else:
        df_out = copy.deepcopy(df)    
    ordered_clique_batches = []
    for i in range( int(len(df_out)/bs) ):
        # Get stuff that hasn't been sampled yet.
        df_out = df_out[df_out["Sampled"] == False]
        # If nothing's left, break.
        if len(df_out) < bs: break
        # Get all remaining lengths.
        lens_remaining = list(set(df_out.index.get_level_values('length')))
        # Get set of cliques given randomly drawn length.
        if len(lens_remaining) < 3:
            len_pool = lens_remaining
        else:
            draw_idx = random.choice(range(1,len(lens_remaining)-1))
            len_pool = lens_remaining[draw_idx-1:draw_idx+2] 
            len_pool = [l for l in len_pool if (abs(len_pool[1] - l) <= flank)]
        # Get the sub df respective to the length interval
        sub_df = df_out.loc[len_pool]
        # Sample from the sub df
        df_samp = sub_df if len(sub_df)<bs else sub_df.sample(bs)
        samp_multiidx = list(df_samp.index)
        # Indicate sampled points in running df.
        df_out.loc[samp_multiidx,"Sampled"] = True
        # If sample doesn't reach batchsize, fill with random cliques ... 
        if len(samp_multiidx) < bs:      
            df_temp = df_out[df_out["Sampled"] == False]            
            filler_samp = df_temp.sample(bs - len(samp_multiidx))
            filler_multiidx = list(filler_samp.index)
            df_out.loc[filler_multiidx,"Sampled"] = True
            samp_multiidx.extend(filler_multiidx)      
        
        # Append new batch to list of batches.
        clique_batch = list(df_out.loc[samp_multiidx]["Clique_idx"].unique())        
        ordered_clique_batches.append(clique_batch)   
      
    return ordered_clique_batches


