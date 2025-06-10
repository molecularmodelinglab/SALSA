import random
import pandas as pd
import csv
import numpy as np
import os
import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
from rdkit.Chem import Descriptors
import torch
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from salsa.constants import ATOMS, PROP_NAMES, INV_COV_CHEMBL
from tqdm import tqdm

def get_props(smiles, prop_list=PROP_NAMES):
#     print(PROP_NAMES)
    calc = MolecularDescriptorCalculator(prop_list)
    props = calc.CalcDescriptors(Chem.MolFromSmiles(smiles))
    return torch.nan_to_num(torch.tensor(props),0.0).float()

## Load ChEMBL.
path = '../data/source/chembl/chembl_valid_lte100_smi_len.csv'
_df = pd.read_csv(path)

## Get sample...
df_samp = _df.sample(100000)
df_samp = df_samp.reset_index(drop=True)

## Calculate MolBert properties on sample.
_chembl_props = []
for sm in tqdm(df_samp.Smiles.values,total=len(df_samp)):
    props = get_props(sm, prop_list=PROP_NAMES) # surface_predictor(sm) 
    _chembl_props.append(props)
chembl_props = np.vstack([np.array(x) for x in _chembl_props])


## Calculate covariance matrix.
cov = np.cov(chembl_props.T)
## Perturb covariance matrix to avoid complex values in inverse ...
# # # # # # # # # #
epsilon = 1e-4
# # # # # # # # # #
cov_perturbed = cov + epsilon*np.eye(len(cov))

## Compute inverse of covariance matrix.
# inv_cov = np.linalg.inv(cov)
inv_cov = np.linalg.inv(cov_perturbed)
print("Complex numbers:", np.iscomplex(inv_cov).any())
print("Infs:", np.isinf(inv_cov).any())



## Get mahala prop distances for 100,000 anchors.
muts_f ='/nas/longleaf/home/k4thryn/Repos/Salsa/data/100000_ancs_10_muts.csv'
df = pd.read_csv(muts_f)
df.sort_values(by=['Anc_id','Is_anc'], ascending=[True,False], inplace=True)
df.reset_index(drop=True,inplace=True)


from scipy.spatial import distance
from itertools import combinations as combo

anc_ids = df.Anc_id.unique().tolist()
random.shuffle(anc_ids)

dists = []
for anc_id in tqdm(anc_ids, total=len(anc_ids)):
    
    df_fam = df[df.Anc_id==anc_id]
    
    smis = df_fam.Smiles.values
    anc_smi = smis[0]
    mut_smis = smis[1:]

    anc_prop = get_props(anc_smi)
    
    for mut_smi in mut_smis:
        mut_prop = get_props(mut_smi)
        d = distance.mahalanobis(anc_prop, mut_prop, VI=inv_cov)
        dists.append(d)
        
        
dists_arr = np.array(dists)
np.savetxt('dists_arr.txt',dists_arr)
        
        
        