import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")
import argparse

from salsa.supermutant_construction import parallel_write_supmut_chunks, consolidate_supmut_chunks 

def main(
        df_path: str,
        n_samp: int,
        tag: str,
        work_path: str,
        depth: int,
        mahala_thresh: str,
        weight: str,
        actions: list,
        ):
    

## ARGS # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# src_path = '../data/source/chembl_valid_lte100_smi_len.csv'
# n_samp = 5000
# work_path = None   #'../data/processed/chembl/scratch/sups/'
# tag = '5000anc-depth5-add-threshNone-wChembl'
# depth = 5
# actions = ['Add']
# mahala_thresh = 'Record'
# weight = 'chembl'
# out_path = '../data/processed/supermutated/'
 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



    work_path, tag = parallel_write_supmut_chunks(df_path=df_path, n_samp=n_samp, work_path=work_path, 
                                                  tag=tag, depth=depth, actions=actions, 
                                                  mahala_thresh=mahala_thresh, weight=weight)
    _ = consolidate_supmut_chunks(work_path, out_path, tag)


if __name__=="__main__":
    parser = argparse.ArgumentParser("Parallel supermutations.")
    
    parser.add_argument("--df_path", type=str, required=False, 
                        default='../data/source/chembl/chembl_valid_lte100_smi_len.csv',
                        help="Path to input file. Default is chembl_valid_lte100_smi_len.csv.")
    parser.add_argument("--n_samp", type=int, default=-1, required=False, 
                        help="Num anchors to sample from anchor df if desired.")
    parser.add_argument("--tag", type=str, required=False, 
                        help="Output tag if desired. Else is set to params.")    
    parser.add_argument("--work_path", type=str, required=False, 
                        help="Path to save intermediate chunk files.")   
       
    parser.add_argument("--depth", type=str, default=5, required=True, 
                        help="How many mutation steps to take. Default is 5.")
    parser.add_argument("--mahala_thresh", type=str, default='Record', required=True, 
                        help="For no filter, use None. For no filter but want to save, use Record.")
    parser.add_argument("--weight", type=str, default='chembl', required=True, 
                        help="Mutation atom-weighting scheme. Choices: chembl, uniform")
    parser.add_argument("--actions", required=True, action='append', nargs='+',
                        help="Actions. List out without commas. Can be any superset of: \
                        Add Remove Replace")
    
    args = parser.parse_args()

    main(
        df_path = args.df_path,
        n_samp = args.n_samp, 
        tag = args.tag, 
        work_path = args.work_path,
        mahala_thresh = args.mahala_thresh,
        weight = args.weight,
        actions = args.actions
        )  
    