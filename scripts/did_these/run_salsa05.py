'''
Start date: May 2, 2023
'''
import pandas as pd
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import warnings
warnings.filterwarnings("ignore")

from salsa.datasets import TrainingDataset, RandomFamSampler
from salsa.constants import NAMED_LOSSES
from salsa.modules import SalsaNet
from salsa.training import fit

# # # # # # # # # # # # ARGS # # # # # # # # # # # #
training_set = 'mutated_01_chembllte100-full-5-Record-chembl.csv'
model_dir = '/work/users/k/4/k4thryn/Salsa/models/'
use_cuda = [0] 
version = '05'

batch_size = 96
lr = 0.0001
losses = NAMED_LOSSES
n_epochs = 3
save_step = 1000

dim_emb = 512   # <<<<<<<<<<<<<<<<< 
dim_salsa = 512   # <<<<<<<<<<<<<<<<< 
n_heads = 8
L_enc = 8
L_dec = 8
dim_ff = 2048
proj_head = False
pool_func = 'avg'   # <<<<<<<<<<<<<<<<< 
actv = 'relu'
# # # # # # # # # # # # # # # # # # # # # # # # # # #

ddir = '../data/processed/train/'
_df = pd.read_csv(f'{ddir}{training_set}')

ds_salsa = TrainingDataset(_df)
sampler = RandomFamSampler(ds_salsa.anc_ids, batch_size=batch_size, drop_last=True)
loader = DataLoader(ds_salsa, batch_sampler=sampler, pin_memory=True)

model = SalsaNet(dim_emb, dim_salsa, n_heads, L_enc, L_dec, dim_ff, 
                 proj_head, pool_func, actv)

fit(model, sampler, loader, use_cuda, version, batch_size, 
    lr, losses, n_epochs, save_step, model_dir ) 
