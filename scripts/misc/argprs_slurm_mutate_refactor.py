
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# SCRIPT  ... # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# EXAMPLE SCRIPT:
# python argprs_slurm_mutat.py

import sys
sys.path.insert(0, '..')
import random
import numpy as np
import pandas as pd
import os
import csv
import warnings
warnings.filterwarnings("ignore")

import argparse
import copy
from datetime import datetime

from salsa.data_mutation import mutate_data
from salsa.constants import ACTIONS

def main( 
        fname: str,
        n_samp: int,
        num_muts: int,
        mahala_thresh: str,
        weight: str,
        working_dir: str,
        out_dir: str,
        workplace: str
        ):

    print(f"Mutating dataset.")

    if workplace=='longleaf': slurm = True
    else: slurm = False

    msgs = [f" > {k}: {v}" for k,v in locals().items()]
    print(('\n').join(msgs))

    df, out_name = _get_df_and_outname(fname=fname, working_dir=working_dir, out_dir=out_dir, workplace=workplace)
    print(f" > out file name: {out_name}.csv")

    if n_samp > 0:
        df = df.sample(n_samp)
        df.reset_index(drop=True, inplace=True)
        
    ## Assuming that first col is id and second col is smiles.
    if len(df.columns) >= 2:
        df = df.iloc[:, [0,1]]
    df.columns = ['Id','Smiles']

    mutate_data(ds=df, n_samp=-1, out_name=out_name, num_muts=num_muts, actions=ACTIONS,
                mahala_thresh=mahala_thresh, weight=weight, use_tqdm=False, slurm=slurm)

    
def _get_df_and_outname(fname=None, working_dir=None, out_dir=None, workplace=None):
 
    if fname=='None': fname = None
    if working_dir=='None': working_dir = None
    if out_dir=='None': out_dir = None
  
    if workplace=='longleaf':
        if fname is None:
            fname = '/nas/longleaf/home/k4thryn/Repos/Salsa/data/source/chembl/chembl_valid_lte100_smi_len.csv'
        if working_dir is None: 
            working_dir = '/work/users/k/4/k4thryn/Repos/Salsa/data/scratch/'
        if out_dir is None:
            out_dir = 'outs/'
        else:
            os.makedirs(f'{working_dir}/{out_dir}', exist_ok=True)
        prefix = fname.split('/')[-1]
        df = pd.read_csv(f'{working_dir}/chunks/{prefix}', header=None)
    
    out_name = f'{working_dir}{out_dir}{prefix}'
    return df, out_name

if __name__=="__main__":
    parser = argparse.ArgumentParser("Mutating dataset.")
    parser.add_argument("--fname", type=str, required=False, help="Name of input file.")
    parser.add_argument("--n_samp", type=int, default=-1, required=False, 
                        help="Num compounds to sample from src file if desired. -1 if full.")
    parser.add_argument("--mahala_thresh", type=str, default='Record', required=False, 
                        help="For no filter, use None. For no filter but record Mahala, use Record (default).")
    parser.add_argument("--num_muts", type=int, default=5, required=False,
                        help="Number of mutants per anchor in src dataset.")
    parser.add_argument("--weight", type=str, default='chembl', required=True)
    parser.add_argument("--working_dir", type=str, default='../data/scratch/', required=False, help="Working directory.")
    parser.add_argument("--out_dir", type=str, default='mutants/', required=False, help="Out directory. Default: outs")
    parser.add_argument("--workplace", type=str, default='longleaf', required=False)
    args = parser.parse_args()

    main(
        fname = args.fname,
        n_samp = args.n_samp, 
        mahala_thresh = args.mahala_thresh, 
        num_muts = args.num_muts,
        weight = args.weight,
        working_dir = args.working_dir,
        out_dir = args.out_dir,
        workplace = args.workplace
        )  
    
