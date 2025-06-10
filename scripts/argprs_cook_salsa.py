'''
Start date: May 25, 2023
'''

import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")
import argparse

from salsa.chef import cook_salsa

def main(
    version: str,
    train_csv: str,
    model_dir: str,
    use_cuda: str,
    batch_size: int,
    lr: float,
    losses: str,
    n_epochs: int,
    save_step: int,
    dim_emb: int,
    dim_salsa: int,
    n_heads: int,
    L_enc: int,
    L_dec: int,
    dim_ff: int,
    proj_head: str,
    pool_func: str, 
    actv: str, 
    ):

    if proj_head=='True': proj_head = True
    elif proj_head=='False': proj_head = False

    my_params = {
        'version': version,
        'train_csv': train_csv,
        'model_dir': model_dir,
        'use_cuda': use_cuda,  
        'batch_size': batch_size,
        'lr': lr,
        'losses': losses,
        'n_epochs': n_epochs,
        'save_step': save_step,
        'dim_emb': dim_emb,
        'dim_salsa': dim_salsa,
        'n_heads': n_heads,
        'L_enc': L_enc,
        'L_dec': L_dec,
        'dim_ff': dim_ff,
        'proj_head': proj_head,
        'pool_func': pool_func, 
        'actv': actv
    }    
    
    cook_salsa(**my_params)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser("Cooking Salsa.")
    
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True, 
                        default='chembl35atms_full_10_6pt5_chembl.csv')
    parser.add_argument("--model_dir",  type=str, required=False, default='../../Salsa/models/')  
    parser.add_argument("--use_cuda", type=str, required=False, default='0',
                        help='Can do 0, 0-1, 0-1-2-3, etc.')   
    
    parser.add_argument("--batch_size", type=int, required=False, default=200)
    parser.add_argument("--lr", type=float, required=False, default=0.0001)
    parser.add_argument("--losses", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True) 
    
    parser.add_argument("--save_step", type=int, required=False, default=10000)
    parser.add_argument("--dim_emb", type=int, required=False, default=512)
    parser.add_argument("--dim_salsa", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=False, default=8)
    parser.add_argument("--L_enc", type=int, required=False, default=8)
    parser.add_argument("--L_dec", type=int, required=False, default=8)
    parser.add_argument("--dim_ff", type=int, required=False, default=2048)
    parser.add_argument("--proj_head", type=str, required=False, default='False')
    parser.add_argument("--pool_func", type=str, required=False, default='avg', 
                        help='Choose from avg, max, set.')
    parser.add_argument("--actv", type=str, required=False, default='relu')

    args = parser.parse_args()

main(
    version = args.version,
    train_csv = args.train_csv, 
    model_dir = args.model_dir, 
    use_cuda = args.use_cuda,
    batch_size = args.batch_size,
    lr = args.lr,
    losses = args.losses,
    n_epochs = args.n_epochs,
    save_step = args.save_step,
    dim_emb = args.dim_emb,
    dim_salsa = args.dim_salsa,
    n_heads = args.n_heads,
    L_enc = args.L_enc,
    L_dec = args.L_dec,
    dim_ff = args.dim_ff,
    proj_head = args.proj_head,
    pool_func = args.pool_func,
    actv = args.actv,
    )  


