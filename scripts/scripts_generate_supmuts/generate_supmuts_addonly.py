import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")

## ARGS # # # # # # # # # # # # # # # # # # # # # # # # # # # #
src_path = '../data/source/chembl/chembl_valid_lte100_smi_len.csv'
n_samp = 10000
depth = 6
tag = '10000anc_thresh4pt5_depth6_addonly'
work_path = None   #'../data/processed/chembl/scratch/sups/'
out_path = '../data/processed/chembl/supermutated/'

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from salsa.data_supermutation import parallel_write_supmut_chunks, consolidate_supmut_chunks 

work_path, tag = parallel_write_supmut_chunks(src_path, n_samp, work_path, tag, 
                                              depth, actions=['Add'], mahala_thresh=4.5)
csv_name = consolidate_supmut_chunks(work_path, out_path, tag)