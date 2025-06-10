'''
Start date: Apr. 24, 2023
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
use_cuda = [0] 
version = '00' 
batch_size = 96 #3*6*16 # must be divisible by num cores and FAM_SIZE
lr = 0.0001
losses = NAMED_LOSSES
n_epochs = 3
save_step = 1000

dim_emb = 128 # must be divisible by num_heads
dim_salsa = 256 #512 #32
n_heads = 8
L_enc = 8
L_dec = 8
dim_ff = 512 # dim_emb*4 (as witnessed in lit)
proj_head = True
pool_func = 'set'
actv = 'gelu'
layer_norm_eps = 0.6 # < < < < < < < < < < <
# # # # # # # # # # # # # # # # # # # # # # # # # # #

# df = pd.read_csv('../data/processed/chembl_05/train/full.csv')

ddir = '../data/processed/chembl/'
f = 'chembllte100_mahala4pt5_20230419_header.csv'
_df = pd.read_csv(f'{ddir}{f}')
ds_salsa = TrainingDataset(_df) # , n_samp=3000)

sampler = RandomFamSampler(ds_salsa.anc_ids, batch_size=batch_size, drop_last=True)
loader = DataLoader(ds_salsa, batch_sampler=sampler, pin_memory=True)

model = SalsaNet(dim_emb, dim_salsa, n_heads, L_enc, L_dec, dim_ff, 
                 proj_head, pool_func, actv, layer_norm_eps)

fit(model, sampler, loader, use_cuda, version, batch_size, 
    lr, losses, n_epochs, save_step)
