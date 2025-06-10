import sys
sys.path.insert(0, '..')
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import argparse

from salsa.data_mutation import parallel_mutate, consolidate_mut_ds_chunks

'''Ex.
python argprs_parallel_mutate.py --df_path ~/Repos/Salsa/data/source/ready_for_modeling/5000ancs_from_supermuts.csv 
--num_muts 30 --mahala_thresh 6.5 --weight chembl 
'''

def main(
        df_path: str,
        n_samp: int,
        tag: str,
        work_path: str,
        num_muts: int,
        mahala_thresh: str,
        weight: str,
        actions: list
        ):

    msgs = [f" > {k}: {v}" for k,v in locals().items()]
    print(('\n').join(msgs))
    
    df = pd.read_csv(df_path)
    actions = actions.split('-')

    work_path, tag = parallel_mutate(df, n_samp=n_samp, tag=tag, work_path=work_path, 
                                     num_muts=num_muts, actions=actions, 
                                     mahala_thresh=mahala_thresh, weight=weight)
    
    csv_name = consolidate_mut_ds_chunks(work_path, out_path=None, tag=tag)
    
    print(f"Done! Saved at: {csv_name}")
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser("Parallel mutations.")
    
    parser.add_argument("--df_path", type=str, required=False, 
                        default='../data/source/chembl_35atms_id_smi_natoms_len.csv',
                        help="Path to input file. Default is chembl_35atms_id_smi_natoms_len.csv.")
    parser.add_argument("--n_samp", type=int, default=-1, required=False, 
                        help="Num anchors to sample from src file if desired.")
    parser.add_argument("--tag", type=str, required=False, 
                        help="Output tag if desired. Else is set to date and params.")    
    parser.add_argument("--work_path", type=str, required=False, 
                        help="Path to save intermediate chunk files. Default to create a scratch dir.")   
    
    parser.add_argument("--num_muts", type=int, default=5, required=False,
                        help="Number of mutants per anchor in src dataset.")    
    parser.add_argument("--mahala_thresh", type=str, default='Record', required=False, 
                        help="For no filter, use None. For no filter but record Mahala, use Record (default).")
    parser.add_argument("--weight", type=str, default='chembl', required=False, 
                        help="Mutation atom-weighting scheme. Choices: chembl, uniform")
    parser.add_argument("--actions", required=False, default='Add-Replace-Remove',
                        help="Actions. List out with dash in between each.")
    
    args = parser.parse_args()

    main(
        df_path = args.df_path,
        n_samp = args.n_samp, 
        tag = args.tag, 
        work_path = args.work_path,
        num_muts = args.num_muts,
        mahala_thresh = args.mahala_thresh,
        weight = args.weight,
        actions = args.actions
        )  
    
    