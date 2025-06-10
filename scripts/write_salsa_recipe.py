import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../salsa/')
from salsa.chef import write_salsa_recipe

# cols = ['version_name','dim_salsa', 'losses','lr', 
#         'pool_func','actv','proj_head',
#         'train_csv'] 

cols = ['version_name', 'dim_salsa', 'losses', 'pool_func','proj_head',
        'L_dec', 'L_enc', 'actv', 'layer_norm_eps',
        'batch_size', 'dim_ff','dim_emb', 'n_heads',
        'lr', 'n_epochs',  'save_step', 'train_csv', 'use_cuda']

df = write_salsa_recipe(cols=cols)