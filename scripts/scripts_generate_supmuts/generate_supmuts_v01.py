import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")

## ARGS # # # # # # # # # # # # # # # # # # # # # # # # # # # #
src_path = '../data/source/chembl/chembl_valid_lte100_smi_len.csv'
n_samp = 5000
tag = '5000anc_thresh4pt5_depth5'
work_path = None   #'../data/processed/chembl/scratch/sups/'
out_path = '../data/processed/chembl/supermutated/'
depth = 5
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from salsa.data_supermutation import parallel_write_supmut_chunks, consolidate_supmut_chunks 

work_path, tag = parallel_write_supmut_chunks(src_path, n_samp, work_path, tag, depth, mahala_thresh=4.5)
csv_name = consolidate_supmut_chunks(work_path, out_path, tag)
# df = pd.read_csv(csv_name, na_values=None, keep_default_na=False)