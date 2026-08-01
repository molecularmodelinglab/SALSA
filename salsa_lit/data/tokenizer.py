"""
SMILES tokenizer for SALSA model.
"""

from typing import List, Dict, Tuple
import numpy as np
import torch

from salsa_lit.constants import (
    VOCAB, TOKENS, N_TOKENS,
    S_TOKEN, E_TOKEN, P_TOKEN, UNK_TOKEN,
    MAX_SMI_LEN, MAX_VEC_LEN
)


class SMILESTokenizer:
    """
    Tokenizer for converting SMILES strings to/from tensor representations.

    This class handles:
    - Br/Cl conversion to single characters (R/L)
    - Unknown token replacement
    - Padding to fixed length
    - Mask generation for transformer

    Parameters
    ----------
    max_len : int, optional
        Maximum sequence length (default uses MAX_SMI_LEN from constants)

    Attributes
    ----------
    vocab : str
        Character vocabulary for tokenization
    tokens : str
        Sorted string of all tokens including special tokens
    max_len : int
        Maximum SMILES length (before adding start/end tokens)
    max_vec_len : int
        Maximum vector length (max_len + 2 for start/end)
    n_tokens : int
        Number of tokens in vocabulary
    """

    def __init__(self, max_len: int = None):
        self.vocab = VOCAB
        self.tokens = TOKENS
        self.n_tokens = N_TOKENS
        self.max_len = max_len or MAX_SMI_LEN
        self.max_vec_len = self.max_len + 2  # +2 for start and end tokens

        # Special token indices
        self.s_token = S_TOKEN
        self.e_token = E_TOKEN
        self.p_token = P_TOKEN
        self.unk_token = UNK_TOKEN

        self.s_idx = self.tokens.index(S_TOKEN)
        self.e_idx = self.tokens.index(E_TOKEN)
        self.p_idx = self.tokens.index(P_TOKEN)
        self.unk_idx = self.tokens.index(UNK_TOKEN)

        self._lut = np.full(256, self.unk_idx, dtype=np.int64)
        for i, char in enumerate(self.tokens):
            self._lut[ord(char)] = i

        out_mask = torch.zeros(self.n_tokens)
        out_mask[self.s_idx] = 1.0
        out_mask[self.p_idx] = 1.0
        self._out_mask = out_mask.unsqueeze(0)

    def encode(self, smiles: str) -> torch.Tensor:
        """
        Convert a SMILES string to a tensor of token indices.

        Parameters
        ----------
        smiles : str
            Input SMILES string

        Returns
        -------
        vec : torch.Tensor
            Tensor of shape (max_vec_len,) with token indices
        """
        # Convert Br/Cl to single characters
        smi = self._do_BrCl_singles(smiles)
        n = len(smi)

        total = self.max_vec_len if n <= self.max_len else n + 2

        vec = np.full(total, self.p_idx, dtype=np.int64)
        vec[0] = self.s_idx
        if n:
            raw = np.frombuffer(smi.encode('ascii', 'replace'), dtype=np.uint8)
            vec[1:1 + n] = self._lut[raw]
        vec[1 + n] = self.e_idx

        return torch.from_numpy(vec)

    def decode(self, vec: torch.Tensor, remove_special: bool = True) -> str:
        """
        Convert a tensor of token indices back to a SMILES string.

        Parameters
        ----------
        vec : torch.Tensor
            Tensor of token indices
        remove_special : bool
            Whether to remove start, end, and padding tokens

        Returns
        -------
        smiles : str
            Decoded SMILES string
        """
        if vec.dim() > 1:
            # Handle batch - return list
            return [self.decode(v, remove_special) for v in vec]

        # Convert indices to characters
        smi_arr = np.array(list(self.tokens))[vec.cpu().detach().numpy()]
        smi = ''.join(smi_arr)

        # Convert back Br/Cl
        smi = self._undo_BrCl_singles(smi)

        if remove_special:
            smi = self._remove_extra_tokens(smi)

        return smi

    def batch_encode(self, smiles_list: List[str]) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of SMILES strings with masks.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        batch : dict
            Dictionary with keys:
            - 'seq': token indices, shape (batch, max_vec_len)
            - 'pad_mask': padding mask, shape (batch, max_vec_len)
            - 'avg_mask': averaging mask, shape (batch, max_vec_len)
            - 'out_mask': output mask, shape (batch, n_tokens)
        """
        seqs = []
        pad_masks = []
        avg_masks = []
        out_masks = []

        for smi in smiles_list:
            vec = self.encode(smi)
            pad_mask, avg_mask, out_mask = self.get_masks(vec)
            seqs.append(vec)
            pad_masks.append(pad_mask)
            avg_masks.append(avg_mask)
            out_masks.append(out_mask)

        return {
            'seq': torch.stack(seqs),
            'pad_mask': torch.stack(pad_masks),
            'avg_mask': torch.stack(avg_masks),
            'out_mask': torch.stack(out_masks),
        }

    def get_masks(self, vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate attention masks for a tokenized sequence.

        Parameters
        ----------
        vec : torch.Tensor
            Tokenized SMILES vector

        Returns
        -------
        pad_mask : torch.Tensor
            Boolean mask for padding tokens
        avg_mask : torch.Tensor
            Float mask excluding padding, start, end tokens (for pooling)
        out_mask : torch.Tensor
            Mask for output logits (masks start and pad tokens)
        """
        # pad mask: masks pad tokens
        pad_mask = (vec == self.p_idx)

        # avg mask: masks pad, start, end tokens
        avg_mask = 1.0 - (
            pad_mask | (vec == self.e_idx) | (vec == self.s_idx)
        ).float()

        # out mask: masks start and pad tokens in output logits. Constant for
        # every sample, so it is built once in __init__ and shared here.
        return pad_mask, avg_mask, self._out_mask

    def _do_BrCl_singles(self, smiles: str) -> str:
        """
        Convert Br and Cl to single characters (R and L).

        Parameters
        ----------
        smiles : str
            Input SMILES

        Returns
        -------
        smiles : str
            SMILES with Br->R, Cl->L
        """
        smiles = smiles.replace('Br', 'R')
        smiles = smiles.replace('Cl', 'L')
        return smiles

    def _undo_BrCl_singles(self, smiles: str) -> str:
        """
        Convert R and L back to Br and Cl.

        Parameters
        ----------
        smiles : str
            Input SMILES with R/L

        Returns
        -------
        smiles : str
            SMILES with R->Br, L->Cl
        """
        smiles = smiles.replace('R', 'Br')
        smiles = smiles.replace('L', 'Cl')
        return smiles

    def _replace_unks(self, smiles: str) -> str:
        """
        Replace unknown characters with UNK token.

        Parameters
        ----------
        smiles : str
            Input SMILES

        Returns
        -------
        smiles : str
            SMILES with unknown chars replaced
        """
        return ''.join([
            char if char in self.vocab else self.unk_token
            for char in smiles
        ])

    def _remove_extra_tokens(self, smiles: str) -> str:
        """
        Remove padding, start, and end tokens from SMILES.

        Parameters
        ----------
        smiles : str
            SMILES with special tokens

        Returns
        -------
        smiles : str
            Clean SMILES string
        """
        smiles = smiles.replace(self.p_token, '')
        smiles = smiles.replace(self.s_token, '')
        smiles = smiles.replace(self.e_token, '')
        return smiles
