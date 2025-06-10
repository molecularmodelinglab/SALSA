import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")
from salsa.constants import ACTIONS

## ARGS # # # # # # # # # # # # # # # # # # # # # # # # # # # #
src_path = '../data/source/ready_for_modeling/notchembl_samp10000_20230716.csv'
n_samp = -1
copies = 1
work_path = None   #'../data/processed/chembl/scratch/sups/'
tag = None 
depth = 6
mahala_thresh = 6.0
weight = 'chembl'
out_path = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from salsa.supermutant_construction import parallel_write_supmut_chunks, consolidate_supmut_chunks 

work_path, tag = parallel_write_supmut_chunks(src_path, n_samp, copies, work_path, 
                                              tag, depth, ACTIONS, mahala_thresh, weight)
csv_name = consolidate_supmut_chunks(work_path, out_path, tag)
# df = pd.read_csv(csv_name, na_values=None, keep_default_na=False)
  
