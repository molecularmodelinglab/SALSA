import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")
from salsa.constants import ACTIONS

## ARGS # # # # # # # # # # # # # # # # # # # # # # # # # # # #
src_path = '../data/source/ready_for_modeling/chembl35atms_id_smi_natoms_len.csv'
n_samp = 5000
work_path = None   #'../data/processed/chembl/scratch/sups/'
tag = None 
depth = 6
mahala_thresh = 6.5
weight = 'chembl'
out_path = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from salsa.supermutant_construction import parallel_write_supmut_chunks, consolidate_supmut_chunks 

work_path, tag = parallel_write_supmut_chunks(src_path, n_samp, work_path, 
                                              tag, depth, ACTIONS, mahala_thresh, weight)
csv_name = consolidate_supmut_chunks(work_path, out_path, tag)
# df = pd.read_csv(csv_name, na_values=None, keep_default_na=False)
  
