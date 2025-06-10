import sys
sys.path.insert(0, '..')

from salsa.eval_utils import save_umap_transform

versions = ['28_202306172142',
            '31_202306081747',
            '32_202306081747']

ds_path = f'../data/source/ready_for_modeling/chembl_35atms_samp100000_id_smi_natoms_len.csv'

for version in versions:
    out_umap, out_latents = save_umap_transform(ds_path, version)
