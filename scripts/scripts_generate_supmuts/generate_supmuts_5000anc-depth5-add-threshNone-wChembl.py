import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")

## ARGS # # # # # # # # # # # # # # # # # # # # # # # # # # # #
src_path = '../data/source/chembl_valid_lte100_smi_len.csv'
n_samp = 5000
work_path = None   #'../data/processed/chembl/scratch/sups/'
tag = '5000anc-depth5-add-threshNone-wChembl'
depth = 5
actions = ['Add']
mahala_thresh = 'Record'
weight = 'chembl'
out_path = '../data/processed/supermutated/'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from salsa.data_supermutation import parallel_write_supmut_chunks, consolidate_supmut_chunks 

work_path, tag = parallel_write_supmut_chunks(src_path, n_samp=n_samp, work_path=work_path, 
                                              tag=tag, depth=depth, actions=actions, 
                                              mahala_thresh=mahala_thresh, weight=weight)
csv_name = consolidate_supmut_chunks(work_path, out_path, tag)

