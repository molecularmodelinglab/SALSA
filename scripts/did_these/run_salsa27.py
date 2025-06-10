'''
Start date: May 30, 2023
'''
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import warnings
warnings.filterwarnings("ignore")

from salsa.chef import cook_salsa

# # # # # # # # # # # # ARGS # # # # # # # # # # # #
my_params = {
    'version': '27',
    'train_csv': 'legacy_chembl_SALSA2022_v03.csv',
    'model_dir': '../../Salsa/models/',
    'use_cuda': '0',  # '0-1-2-3'
    'batch_size': 100,
    'lr': 0.00001,
    'losses': 'Recon-SupCon',
    'n_epochs': 3,
    'save_step': 10000,
    'dim_emb': 512,
    'dim_salsa': 32,
    'n_heads': 8,
    'L_enc': 12,
    'L_dec': 12,
    'dim_ff': 2048,
    'proj_head': False,
    'pool_func': 'avg', # 'set',
    'actv': 'relu', # 'gelu'
    'layer_norm_eps': 0.6
    }

cook_salsa(**my_params)
