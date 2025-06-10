import sys
sys.path.insert(0, '..')

import pandas as pd
from salsa.constants import ACTIONS, MAHALA_THRESH, WEIGHT
from salsa.data_mutation import mutate_data

df = pd.read_csv('../data/source/ready_for_modeling/2023chembl_novel_cleaned_header_filtered.csv')
out_name = f'mutated_2023chembl_10000ancs-10muts-6pt5thresh-chemblw'.replace('.','pt')
out_name = f'../data/test/{out_name}'

mutate_data(ds=df, n_samp=10000, out_name=out_name, num_muts=10, actions=ACTIONS, 
            mahala_thresh=MAHALA_THRESH, weight=WEIGHT, use_tqdm=True, slurm=False)