"""
Neural network components for SALSA model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from salsa_lit.constants import MAX_POS_LEN


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.

    Parameters
    ----------
    d_model : int
        Model dimension
    dropout : float
        Dropout probability
    max_len : int
        Maximum sequence length
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = None):
        super().__init__()
        max_len = max_len or MAX_POS_LEN
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        x : torch.Tensor
            Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ProjHead(nn.Module):
    """
    Projection head for contrastive learning.

    Two-layer MLP with ReLU activation and L2 normalization.

    Parameters
    ----------
    dim : int
        Input and output dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc_layer1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU(inplace=False)
        self.fc_layer2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projection head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        x_normed : torch.Tensor
            L2-normalized output
        """
        x = self.fc_layer1(x)
        x = self.fc_layer2(self.relu(x))
        x_normed = F.normalize(x, p=2.0, dim=-1)
        return x_normed


class MAB(nn.Module):
    """
    Multihead Attention Block for set-based pooling.

    Parameters
    ----------
    dim_Q : int
        Query dimension
    dim_K : int
        Key dimension
    dim_V : int
        Value/output dimension
    num_heads : int
        Number of attention heads
    ln : bool
        Whether to use layer normalization
    """

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention block.

        Parameters
        ----------
        Q : torch.Tensor
            Query tensor
        K : torch.Tensor
            Key tensor (also used as value)

        Returns
        -------
        O : torch.Tensor
            Output tensor
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.

    Learns a set of seed vectors that attend to input sequence
    to produce fixed-size output.

    Parameters
    ----------
    dim : int
        Model dimension
    num_heads : int
        Number of attention heads
    num_seeds : int
        Number of output vectors (typically 1 for pooling)
    ln : bool
        Whether to use layer normalization
    """

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1, ln: bool = False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Pool input sequence to fixed-size output.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch, seq_len, dim)

        Returns
        -------
        out : torch.Tensor
            Pooled output of shape (batch, num_seeds, dim)
        """
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


def get_downstream_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    Create causal mask for transformer decoder.

    Parameters
    ----------
    seq : torch.Tensor
        Input sequence tensor of shape (batch, seq_len)

    Returns
    -------
    mask : torch.Tensor
        Boolean causal mask of shape (seq_len, seq_len)
    """
    _, ls = seq.size()
    dvc = seq.device
    mask = torch.triu(torch.ones((ls, ls), device=dvc), diagonal=1).bool()
    return mask


def get_subsequent_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    Create subsequent mask for masking future positions.

    Parameters
    ----------
    seq : torch.Tensor
        Input sequence tensor

    Returns
    -------
    mask : torch.Tensor
        Boolean subsequent mask
    """
    _, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device), diagonal=1
    ).bool()
    return subsequent_mask


def count_params(model: nn.Module) -> int:
    """
    Count total parameters in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model

    Returns
    -------
    n_params : int
        Total number of parameters
    """
    return sum([p.nelement() for p in model.parameters()])
