import re
from difflib import SequenceMatcher
from itertools import combinations, permutations

# A function to find exit vector numbers in smiles e.g. 'c1([*:3])cc([*:4])cn1[*:6]' -> ['3', '4', '6']
def find_dummy_numbers(smiles):
    single_bond_matches = re.findall(r'(?<!\=)\[\*:(\d+)\]', smiles)
    double_bond_matches = re.findall(r'\=\[\*:(\d+)\]', smiles)

    return single_bond_matches, double_bond_matches

# A function to remove exit vector numbers from strings e.g. [4*] -> [*]
def remove_dummy_numbers(smiles):
    return re.sub(r'\[(\d+)\*\]', '[*]', smiles)
