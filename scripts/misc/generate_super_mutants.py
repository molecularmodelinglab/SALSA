import random
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import csv
import numpy as np
import os
import sys
sys.path.insert(0, '..')

#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This script produces super mutants for a user-provided source # # # # # 
# (anchor) dataset in a parallel fashion. # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
# Load source dataset (set of anchors).  
# Required columns: "Smiles"


# # # # # # # # # # # #
SOURCE_DS = 'chembl/chembl_valid.csv'
DS_NAME = 'chembl_05'
SPLIT = 'super_mutants'
N_SAMP = 5000
N_PAD = 20
DEPTH = 5
# # # # # # # #

from salsa.constants import MAX_SMI_LEN
# from salsa.super_mutants import *
from salsa.super_mutants import format_sup_mut_ds, write_expand_super_muts

ds_dir = '../data/source'
df = pd.read_csv(os.path.join(ds_dir, SOURCE_DS), usecols=['Smiles'])
df['Id'] = df.index # Idx are the ids.
df = df[['Id','Smiles']]
df = df[df.Smiles.str.len() <= MAX_SMI_LEN]
df = df.sample(N_SAMP)
df.reset_index(inplace=True,drop=True)
df = _format_sup_mut_ds(df)


os.makedirs(f'../data/processed/{DS_NAME}/{SPLIT}/', exist_ok=True)

# Partioning across cpus.
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from salsa.data_mutation import write_mut_ds_chunk

num_per_thread = len(df)//(multiprocessing.cpu_count()-2) + 1
partitions = np.arange(0, len(df)+N_PAD, num_per_thread)
idc_chunks = [[s,e] for s,e in zip(partitions, partitions[1:])]
assert(idc_chunks[-1][1]>=len(df))


print(f"Super mutating dataset of {len(df)}. Running {len(idc_chunks)} processes.")
print(f"There are on sets of {num_per_thread} compounds per process.")
print(f"... Saving to path ../data/processed/{DS_NAME}/{SPLIT}/")

start_time = datetime.now()
parallelizer = Parallel(n_jobs=multiprocessing.cpu_count(), backend= 'multiprocessing' )

tasks = (delayed(write_expand_super_muts)(s, e, df, ds_name=DS_NAME, split=SPLIT, depth=DEPTH) for s,e in idc_chunks)

# expand_super_muts()

parallelizer(tasks)
r_time = datetime.now() - start_time
print(f"Done!")
print(f"Runtime:{r_time.seconds//60%60} mins {r_time.seconds%60} secs.")

