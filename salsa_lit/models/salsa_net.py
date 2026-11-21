"""
SalsaNet model for learning semantically aware latent space autoencoder.
"""

import os
import math
import warnings
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer as EncLayer
from torch.nn import TransformerEncoder as Encoder
from torch.nn import TransformerDecoderLayer as DecLayer
from torch.nn import TransformerDecoder as Decoder

from salsa_lit.constants import N_TOKENS, MAX_VEC_LEN, TOKENS
from salsa_lit.config import ModelConfig
from salsa_lit.models.components import (
    PositionalEncoding, ProjHead, PMA, 
    get_downstream_mask, get_subsequent_mask
)


class SalsaNet(nn.Module):
    """
    SalsaNet model for learning semantically aware latent space autoencoder.

    The model consists of:
    1. Token embedding + positional encoding
    2. Transformer encoder
    3. Pooling layer (avg, max, or attention-based)
    4. Linear projection to SALSA latent space
    5. Transformer decoder for reconstruction

    Parameters
    ----------
    config : ModelConfig
        Model configuration object
    fam_size : int, optional
        Family size for training (anchor + mutants).
        Required for training, not needed for inference.

    Attributes
    ----------
    dim_emb : int
        Embedding dimension
    pool_func : str
        Pooling function type
    fam_size : int
        Family size for reshaping during training
    """

    def __init__(self, config: ModelConfig = None, fam_size: int = None, **kwargs):
        super().__init__()

        # Handle both config object and direct kwargs (for backward compatibility)
        if config is None:
            config = ModelConfig(**kwargs)

        self.config = config
        self.dim_emb = config.dim_emb
        self.pool_func = config.pool_func
        self.proj_head = config.proj_head
        self.fam_size = fam_size

        _dim_salsa = config.dim_salsa
        _nhead = config.n_heads
        _num_layers_enc = config.L_enc
        _num_layers_dec = config.L_dec
        _dim_feedforward = config.dim_ff
        _activation = config.actv
        _layer_norm_eps = config.layer_norm_eps
        _dropout = config.dropout
        _batch_first = True

        # Initial embedding
        self.embedding = nn.Embedding(
            num_embeddings=N_TOKENS,
            embedding_dim=self.dim_emb
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(
            d_model=self.dim_emb,
            dropout=_dropout
        )

        # Encoder
        enc_layer = EncLayer(
            d_model=self.dim_emb,
            nhead=_nhead,
            dim_feedforward=_dim_feedforward,
            dropout=_dropout,
            activation=_activation,
            layer_norm_eps=_layer_norm_eps,
            batch_first=_batch_first
        )
        self.enc = Encoder(
            encoder_layer=enc_layer,
            num_layers=_num_layers_enc
        )

        # Set-based pooling (Attention aggregation)
        if self.pool_func == 'set':
            self.pma = PMA(
                dim=self.dim_emb,
                num_heads=_nhead,
                num_seeds=1,
                ln=False
            )

        # Squeeze to SALSA dimensions
        self.salsa_layer = nn.Linear(
            in_features=self.dim_emb,
            out_features=_dim_salsa
        )

        # Projection head
        if self.proj_head:
            self.proj_net = ProjHead(dim=_dim_salsa)

        # Upsample to embedding dimensions
        self.upsamp_layer = nn.Linear(
            in_features=_dim_salsa,
            out_features=self.dim_emb * MAX_VEC_LEN
        )

        # Decoder
        dec_layer = DecLayer(
            d_model=self.dim_emb,
            nhead=_nhead,
            dim_feedforward=_dim_feedforward,
            dropout=_dropout,
            activation=_activation,
            layer_norm_eps=_layer_norm_eps,
            batch_first=_batch_first
        )
        self.dec = Decoder(
            decoder_layer=dec_layer,
            num_layers=_num_layers_dec
        )

        # Decoding output layer
        self.decode_out = nn.Linear(
            in_features=self.dim_emb,
            out_features=N_TOKENS
        )

    def aggregate_vec(self, avg_mask: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Aggregate encoded sequence to fixed-size vector.

        Parameters
        ----------
        avg_mask : torch.Tensor
            Mask for averaging (excludes padding, start, end)
        enc_out : torch.Tensor
            Encoder output, shape (batch, seq_len, dim_emb)

        Returns
        -------
        enc_agg : torch.Tensor
            Aggregated vector, shape (batch, dim_emb)
        """
        if self.pool_func == 'avg':
            enc_sum = (avg_mask.unsqueeze(2) * enc_out).sum(axis=1)
            enc_agg = enc_sum / (avg_mask.sum(axis=1).unsqueeze(1))
        elif self.pool_func == 'max':
            a = (avg_mask.unsqueeze(2) * enc_out)
            ii = torch.argmax(a, dim=1)
            enc_agg = torch.gather(a, dim=1, index=ii.unsqueeze(1)).squeeze(1)
        elif self.pool_func == 'set':
            enc_agg = self.pma(enc_out)
            enc_agg = enc_agg.squeeze()
        else:
            raise ValueError(f'Unrecognized pooling function: {self.pool_func}')
        return enc_agg

    def forward(self, seq: torch.Tensor, pad_mask: torch.Tensor = None,
                avg_mask: torch.Tensor = None, out_mask: torch.Tensor = None,
                **kwargs):
        """
        Forward pass for training.

        Parameters
        ----------
        seq : torch.Tensor
            Input token indices, shape (batch, seq_len)
        pad_mask : torch.Tensor, optional
            Padding mask
        avg_mask : torch.Tensor, optional
            Averaging mask for pooling
        out_mask : torch.Tensor, optional
            Output mask for decoder

        Returns
        -------
        salsa_latent : torch.Tensor
            SALSA latent vectors, shape (n_fams, fam_size, dim_salsa)
        inter_latent : torch.Tensor
            Intermediate latents (after projection head if used)
        dec_out : torch.Tensor
            Decoder output logits
        """
        if len(seq.shape) == 1:
            seq = seq.unsqueeze(0)

        # Masks
        mask = get_downstream_mask(seq)  # causal mask
        if avg_mask is None:
            avg_mask = torch.ones_like(seq)
        if out_mask is None:
            out_mask = torch.zeros(N_TOKENS).to(seq.device)

        # Encode
        emb_seq = self.pos_enc(self.embedding(seq))
        enc_out = self.enc(src=emb_seq, mask=mask, src_key_padding_mask=pad_mask)

        enc_agg = self.aggregate_vec(avg_mask, enc_out)

        salsa_latent = self.salsa_layer(enc_agg)

        salsa_latent = F.normalize(salsa_latent, p=2.0, dim=-1)

        if self.proj_head:
            proj_latent = self.proj_net(salsa_latent)
            inter_latent = proj_latent
        else:
            inter_latent = salsa_latent

        # Decode
        salsa_out = self.upsamp_layer(salsa_latent)
        salsa_out = salsa_out.reshape(-1, MAX_VEC_LEN, self.dim_emb)
        dec_out = self.dec(
            tgt=emb_seq,
            memory=salsa_out,
            tgt_mask=mask,
            memory_mask=mask,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=pad_mask
        )
        dec_out = self.decode_out(dec_out).masked_fill(out_mask == 1, -1e4)

        # Reshape for contrastive loss
        salsa_latent = torch.stack(torch.split(salsa_latent, self.fam_size), dim=0)
        inter_latent = torch.stack(torch.split(inter_latent, self.fam_size), dim=0)

        return salsa_latent, inter_latent, dec_out

    def embed(self, seq: torch.Tensor, pad_mask: torch.Tensor = None,
              avg_mask: torch.Tensor = None, out_mask: torch.Tensor = None,
              **kwargs):
        """
        Encode SMILES to latent vectors (inference mode).

        Parameters
        ----------
        seq : torch.Tensor
            Input token indices
        pad_mask : torch.Tensor, optional
            Padding mask
        avg_mask : torch.Tensor, optional
            Averaging mask
        out_mask : torch.Tensor, optional
            Output mask

        Returns
        -------
        salsa_latent : torch.Tensor
            SALSA latent vectors, shape (batch, dim_salsa)
        inter_latent : torch.Tensor
            Intermediate latents
        """
        if len(seq.shape) == 1:
            seq = seq.unsqueeze(0)

        mask = get_downstream_mask(seq)
        if avg_mask is None:
            avg_mask = torch.ones_like(seq)
        if out_mask is None:
            out_mask = torch.zeros(N_TOKENS).to(seq.device)

        emb_seq = self.pos_enc(self.embedding(seq))
        enc_out = self.enc(src=emb_seq, mask=mask, src_key_padding_mask=pad_mask)

        enc_agg = self.aggregate_vec(avg_mask, enc_out)

        salsa_latent = self.salsa_layer(enc_agg)

        salsa_latent = F.normalize(salsa_latent, p=2.0, dim=-1)

        if self.proj_head:
            proj_latent = self.proj_net(salsa_latent)
            inter_latent = proj_latent
        else:
            inter_latent = salsa_latent

        if len(salsa_latent.shape) == 1:
            salsa_latent = salsa_latent.unsqueeze(0)
        if len(inter_latent.shape) == 1:
            inter_latent = inter_latent.unsqueeze(0)

        return salsa_latent, inter_latent

    def generate(self, latents: torch.Tensor, device: torch.device, 
                 generate_cnt: int = 1) -> torch.Tensor:
        """
        Generate SMILES from latent vectors.

        Parameters
        ----------
        latents : torch.Tensor or np.ndarray
            Latent vectors to decode
        device : torch.device
            Device for computation
        generate_cnt : int
            Number of samples to generate per latent

        Returns
        -------
        seq : torch.Tensor
            Generated token sequences
        """
        s_idx = TOKENS.index('<')  # start index
        e_idx = TOKENS.index('>')  # end index
        p_idx = TOKENS.index('X')  # pad index

        self.eval()

        if latents is None:
            latents = torch.randn((generate_cnt, self.config.dim_salsa)).to(device)

        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents)

        latents = latents.to(device)

        if latents.dim() == 1:
            latents = latents.unsqueeze(0).repeat([generate_cnt, 1])
            latents = latents.to(device)

        if latents.dim() == 2:
            latents = self.upsamp_layer(latents)
            latents = latents.reshape(-1, MAX_VEC_LEN, self.dim_emb)

        out_mask = torch.ones(N_TOKENS)
        idx_tens = torch.tensor([s_idx, p_idx])
        out_zeros = torch.zeros_like(out_mask)
        out_mask = out_zeros.scatter_(0, idx_tens, out_mask).to(device)
        causal_mask = get_subsequent_mask(latents[..., 0])

        gen_range = [s_idx for _ in range(generate_cnt)]
        seq = torch.tensor(gen_range).unsqueeze(1).to(device)

        with torch.no_grad():
            for _ in range(MAX_VEC_LEN):
                dec_mask = causal_mask[:seq.shape[-1], :seq.shape[-1]]
                mem_mask = causal_mask[:seq.shape[-1]]
                emb_seq = self.pos_enc(self.embedding(seq))
                model_out = self.dec(
                    emb_seq, latents,
                    tgt_mask=dec_mask,
                    memory_mask=mem_mask
                )
                model_out = self.decode_out(model_out)
                logits = model_out.masked_fill(out_mask == 1, -1e9)
                top_i = torch.distributions.categorical.Categorical(
                    logits=logits[:, -1]
                ).sample()
                top_i = top_i.masked_fill(
                    (seq[:, -1] == e_idx) | (seq[:, -1] == p_idx), p_idx
                )
                seq = torch.cat([seq, top_i.unsqueeze(1)], dim=-1)

            close_seq = torch.tensor([e_idx for _ in range(generate_cnt)]).to(device)
            close_seq = close_seq.masked_fill(
                (seq[:, -1] == e_idx) | (seq[:, -1] == p_idx), p_idx
            )
            seq = torch.cat([seq, close_seq.unsqueeze(1)], dim=-1)

        return seq

    @classmethod
    def from_pretrained(cls, path: str, config: ModelConfig = None, 
                        map_location: str = None):
        """
        Load a pretrained model from checkpoint.

        Parameters
        ----------
        path : str
            Path to model checkpoint file or directory containing model and params
        config : ModelConfig, optional
            Model configuration (loaded from params.yml if not provided)
        map_location : str, optional
            Device to map model to

        Returns
        -------
        model : SalsaNet
            Loaded model
        """
        # Determine paths
        if os.path.isdir(path):
            # Directory with model and params
            params_path = os.path.join(path, 'params.yml')
            model_files = [f for f in os.listdir(path) if f.endswith('.pt')]
            if 'model_trained_model.pt' in model_files:
                weights_path = os.path.join(path, 'model_trained_model.pt')
            elif model_files:
                weights_path = os.path.join(path, model_files[0])
            else:
                raise FileNotFoundError(f"No .pt files found in {path}")
        else:
            # Direct path to weights
            weights_path = path
            params_path = os.path.join(os.path.dirname(path), 'params.yml')

        obj = torch.load(weights_path, map_location=map_location, weights_only=False)

        if isinstance(obj, dict) and 'state_dict' in obj:
            hparams = obj.get('hyper_parameters') or {}
            if config is None and hparams.get('model_config'):
                config = ModelConfig.from_dict(hparams['model_config'])
            prefix = 'model.'
            state_dict = {k[len(prefix):]: v for k, v in obj['state_dict'].items()
                          if k.startswith(prefix)}
        else:
            state_dict = obj

        if config is None:
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)
                config = ModelConfig.from_dict(params)
            else:
                config = ModelConfig()

        # Create model and load weights
        model = cls(config)
        incompatible = model.load_state_dict(state_dict, strict=False)

        n_loaded = len(model.state_dict()) - len(incompatible.missing_keys)
        if n_loaded == 0:
            raise RuntimeError(
                f"No weights loaded from {weights_path}: none of its keys match "
                f"SalsaNet. Found keys like {list(state_dict)[:3]}"
            )
        if incompatible.missing_keys:
            warnings.warn(
                f"{len(incompatible.missing_keys)} parameters were absent from "
                f"{weights_path} and keep their initial values "
                f"(e.g. {incompatible.missing_keys[:3]})"
            )

        return model

    def save_pretrained(self, path: str):
        """
        Save model weights and configuration.

        Parameters
        ----------
        path : str
            Directory to save model
        """
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'params.yml'), 'w') as f:
            yaml.dump(self.config.to_dict(), f)

        torch.save(self.state_dict(), os.path.join(path, 'model_trained_model.pt'))
