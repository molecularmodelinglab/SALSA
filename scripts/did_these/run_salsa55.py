'''
Start date: June 22, 2023
'''
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import warnings
warnings.filterwarnings("ignore")

from salsa.chef import cook_salsa

# # # # # # # # # # # # ARGS # # # # # # # # # # # #
my_params = {
    'version': '55',
    'train_csv': 'chembl35atms_full_10_Record_truechembl.csv',
    'model_dir': '../../Salsa/models/',
    'use_cuda': '0',  # '0-1-2-3'
    'batch_size': 200,
    'lr': 0.0001,
    'losses': 'SupCon',
    'n_epochs': 3,
    'save_step': 10000,
    'dim_emb': 512,
    'dim_salsa': 16,
    'n_heads': 8,
    'L_enc': 8,
    'L_dec': 8,
    'dim_ff': 2048,
    'proj_head': False,
    'pool_func': 'avg', 
    'actv': 'relu',
    'layer_norm_eps': 1e-5
}

cook_salsa(**my_params)
