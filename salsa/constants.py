import os

import numpy as np
import pandas as pd

# Augmented dataset # # # # # # # # # # # # # # # # # # # # # # # # # #
MAX_ANC_LEN = 100
MAX_MUT_LEN = 120
MAX_VEC_LEN = 122
MAX_SMI_LEN = 120
MAX_POS_LEN = 5000

MAHALA_THRESH = 6.0
N_MUTS = 5
# FAM_SIZE = N_MUTS + 1
ACTIONS = ['Add', 'Replace', 'Remove']
WEIGHT = 'chembl'

NAMED_LOSSES = ['Recon', 'SupCon']

VOCAB = '#%()+-0123456789<=>BCFHILNOPRSX[]cnosp$'

_pack_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INV_COV_CHEMBL = np.loadtxt(os.path.join(_pack_dir, 'data/config/pseudo_chembl_inv_cov_20230510.txt'))

prop_csv = pd.read_csv(os.path.join(_pack_dir, 'data/config/selected_props_20230510.csv'))
PROP_NAMES = prop_csv['prop_name'].values.tolist()

ATOMS = {'C', 'O', 'N', 'S', 'Br', 'I', 'F', 'Cl', 'P', 'B'}

ATOM_TO_CNT = {"C": 3806500,
               "O": 571807,
               "N": 562208,
               "S": 73267,
               "Br": 8730,
               "I": 996,
               "F": 66513,
               "Cl": 41958,
               "P": 3018,
               "B": 1}

# Vectorization # # # # # # # # # # # # # # # # # # # # # # # # # # #
N_TOKENS = 39  # ... is len(VOCAB)
UNK_TOKEN = '$'
S_TOKEN = '<'  # start token
E_TOKEN = '>'  # end token
P_TOKEN = 'X'  # pad token
_tokens = list(set(VOCAB + S_TOKEN + E_TOKEN + P_TOKEN + UNK_TOKEN))
TOKENS = ''.join(list(np.sort(_tokens)))
