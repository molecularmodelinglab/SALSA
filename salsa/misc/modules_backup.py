'''
Start date: Tuesday, Apr. 12, 2023
'''
#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer as DecLayer
from torch.nn import TransformerDecoder as Decoder
from torch.nn import TransformerEncoderLayer as EncLayer
from torch.nn import TransformerEncoder as Encoder

from salsa.constants import N_TOKENS, MAX_VEC_LEN

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Primary nn module # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class SalsaNet(nn.Module):
#     def __init__(self,
#                  dim_emb=512, heads=8, dim_salsa=32, 
#                  L_enc=6, L_dec=6, dim_ff=2048, 
#                  proj_head=False, pool_func="set",
#                  drpt=0.1, actv='relu', eps=0.6, batch_first=True, **kwargs):
    def __init__(self, dim_emb, dim_salsa, n_heads, L_enc, L_dec, dim_ff, 
                 fam_size=None,
                 proj_head=True, 
                 pool_func='set', 
                 actv='relu',
                 layer_norm_eps=1e-5,
                 **kwargs):
        '''
        dim_emb: dimensions of initial embedding 
            (fed from PositionalEncoding to EncLayer)
        n_heads: number of attention heads
        dim_salsa: dimensions of latent space (aka 'Salsa space')
        L_enc: number of layers in encoder
        L_dec: number of layers in decoder
        dim_ff: dimensions of feed forward networks in encoder/decoder
        fam_size: !!!!
        '''
        super().__init__()
        
        # If only referced in init, are prefaced with '_'
        self.dim_emb = dim_emb
        _dim_salsa = dim_salsa
        _nhead = n_heads
        _num_layers_enc = L_enc
        _num_layers_dec = L_dec
        _dim_feedforward = dim_ff
        self.proj_head = proj_head
        self.pool_func = pool_func
        self.fam_size = fam_size
        _activation = actv # default: 'gelu'        
        _layer_norm_eps = layer_norm_eps  # default: 1e-5
        # Hard-coded attributes ...
        _dropout = 0.1 # default: 0.1
        _batch_first = True
        
        # Initial embedding. 
        self.embedding = nn.Embedding(num_embeddings=N_TOKENS, 
                                      embedding_dim=self.dim_emb)
        # Positional encoding.
        self.pos_enc = PositionalEncoding(d_model=self.dim_emb, 
                                          dropout=_dropout)

        # Encoder. .... TODO: change dim_emb for SmallSA?
        enc_layer = EncLayer(d_model=self.dim_emb, 
                             nhead=_nhead,
                             dim_feedforward=_dim_feedforward, 
                             dropout=_dropout,
                             activation=_activation, 
                             layer_norm_eps=_layer_norm_eps,
                             batch_first=_batch_first)
        
        self.enc = Encoder(encoder_layer=enc_layer, 
                           num_layers=_num_layers_enc)
        
        # Set-based pooling (Attention aggregation).
        self.pma = PMA(dim=self.dim_emb, 
                       num_heads=_nhead, 
                       num_seeds=1, 
                       ln=False)
        
        # Squeeze to Salsa dims.
        # TODO: non-linear squeezee for SmallSA ? (for bigger dim reduction)
        self.salsa_layer = nn.Linear(in_features=self.dim_emb,
                                     out_features=_dim_salsa) 
        
        # Projection head
        self.proj_net = ProjHead(dim=_dim_salsa)
        
        # Upsample to embedding dims.
        self.upsamp_layer = nn.Linear(in_features=_dim_salsa,
                                      out_features=self.dim_emb*MAX_VEC_LEN)
        
        # Decoder.
        dec_layer = DecLayer(d_model=self.dim_emb, 
                             nhead=_nhead,
                             dim_feedforward=_dim_feedforward, 
                             dropout=_dropout,
                             activation=_activation, 
                             layer_norm_eps=_layer_norm_eps,
                             batch_first=_batch_first)
        
        self.dec = Decoder(decoder_layer=dec_layer, 
                           num_layers=_num_layers_dec)
        
        # Decoding out layer.
        self.decode_out = nn.Linear(in_features=self.dim_emb, 
                                    out_features=N_TOKENS)
        
    
    
    def aggregate_vec(self, avg_mask, enc_out, pool_func):
        if pool_func=='avg':
            enc_sum = (avg_mask.unsqueeze(2)*enc_out).sum(axis = 1)
            enc_agg = enc_sum/(avg_mask.sum(axis = 1).unsqueeze(1))
        elif self.pool_func=='max':
            a = (avg_mask.unsqueeze(2)*enc_out)
            ii = torch.argmax(a, dim=1)
            enc_agg = torch.gather(a, dim=1, index=ii.unsqueeze(1)).squeeze(1)
        elif self.pool_func=='set':
            enc_agg = self.pma(enc_out)
            enc_agg = enc_agg.squeeze()  
        return enc_agg
    
    
    def forward(self, seq, pad_mask=None, avg_mask=None, out_mask=None, **kwargs):        
        '''
        Output:
            salsa_latent: The latent 'Salsa' vectors. 
            inter_latent: The intermediate vectors on which the SupCon loss is computed.
                May be one of two things... projection head latents OR salsa latents.
            dec_out: Decoder outputs that are needed for decoding latents into SMILES.
        '''        
        
        
        if len(seq.shape)==1:
            seq = seq.unsqueeze(0)
#         print('\nseq:',seq.shape)
        
        # Masks
        mask = get_downstream_mask(seq) # casual mask ... 
        if avg_mask is None: avg_mask = torch.ones_like(seq)
        if out_mask is None: out_mask = torch.zeros(N_TOKENS).to(seq.device)
            
        # Encode
        emb_seq = self.pos_enc( self.embedding(seq) )
#         print('emb_seq:',emb_seq.shape)
        enc_out = self.enc(src=emb_seq, mask=mask, src_key_padding_mask=pad_mask)
#         print('enc_out:',enc_out.shape)
        
        # Aggregate the latent vector
        enc_agg = SalsaNet.aggregate_vec(self, avg_mask, enc_out, self.pool_func)
#         print('enc_agg:',enc_agg.shape)
        
        # Squeeze to salsa dims.
        salsa_latent = self.salsa_layer(enc_agg)
#         print('salsa_vec:',salsa_vec.shape)
        
        # Normalize.
#         salsa_vec = F.normalize(salsa_vec, p=2.0, dim=-1)
        salsa_latent = F.normalize(salsa_latent, p=2.0, dim=-1)
#         print('salsa_vec normed:',salsa_vec.shape)        
        
        # Get final Salsa latents.
#         salsa_latent = torch.stack(torch.split(salsa_vec, FAM_SIZE), dim=0)  
#         if len(salsa_latent.shape)==1: salsa_latent = salsa_latent.unsqueeze(0)
#         print('salsa_latent:',salsa_latent.shape) 

        # Get projection head latents if needed.
        if self.proj_head: 
            proj_latent = self.proj_net(salsa_latent)
            inter_latent = proj_latent
#             print('proj_out:',proj_out.shape) 
#             inter_latent = torch.stack(torch.split(proj_out, FAM_SIZE), dim=0)
#             if len(proj_out.shape)==1: proj_out = proj_out.unsqueeze(0)
            
        else: 
            inter_latent = salsa_latent
#         print('inter_latent:',inter_latent.shape)
        
        # Decode
        salsa_out = self.upsamp_layer(salsa_latent)
#         print('salsa_out:',salsa_out.shape)
        salsa_out = salsa_out.reshape(-1, MAX_VEC_LEN, self.dim_emb)
#         print('salsa_out reshaped:',salsa_out.shape)
        dec_out = self.dec(tgt=emb_seq, 
                           memory=salsa_out,
                           tgt_mask=mask, 
                           memory_mask=mask,
                           tgt_key_padding_mask=pad_mask,
                           memory_key_padding_mask=pad_mask)
        # Get decoder outputs.
        dec_out = self.decode_out(dec_out).masked_fill(out_mask==1,-1e9) 
#         if len(dec_out.shape)==1: dec_out = dec_out.unsqueeze(0)
#         print('dec_out:',dec_out.shape)

        if self.training: #==False: # If in eval mode ... 
#             print('eval!')
            salsa_latent = torch.stack(torch.split(salsa_latent, self.fam_size), dim=0)  
            if len(salsa_latent.shape)==1: salsa_latent = salsa_latent.unsqueeze(0)
            inter_latent = torch.stack(torch.split(inter_latent, self.fam_size), dim=0)
            if len(inter_latent.shape)==1: inter_latent = inter_latent.unsqueeze(0)
            if len(dec_out.shape)==1: dec_out = dec_out.unsqueeze(0)
                
        else:
            if len(salsa_latent.shape)==1: salsa_latent = salsa_latent.unsqueeze(0)
            if len(inter_latent.shape)==1: inter_latent = inter_latent.unsqueeze(0)
            if len(dec_out.shape)==1: dec_out = dec_out.unsqueeze(0)            
            
        return salsa_latent, inter_latent, dec_out
#         return salsa_latent, salsa_vec, inter_latent, dec_out
    
    
# seq: torch.Size([30, 122])
# emb_seq: torch.Size([30, 122, 512])
# enc_out: torch.Size([30, 122, 512])
# enc_agg: torch.Size([30, 512])
# salsa_vec: torch.Size([30, 32])
# salsa_vec normed: torch.Size([30, 32])
# salsa_latent: torch.Size([5, 6, 32])
# proj_out: torch.Size([30, 32])
# proj_latent: torch.Size([5, 6, 32])
# salsa_out: torch.Size([30, 62464])
# salsa_out reshaped: torch.Size([30, 122, 512])
# dec_out: torch.Size([30, 122, 39])
     
        
#         latent_vec = self.salsa_layer(enc_agg)
#         latent_vec = F.normalize(latent_vec, p=2.0, dim=-1)

#         if self.proj_head: 
#             proj_out = self.proj_net(latent_vec)
#         else: 
#             proj_out = latent_vec

#         latent_out = self.upsamp_layer(proj_out)
#         latent_out = latent_out.reshape(-1, MAX_VEC_LEN, self.dim_emb)
        


#         # Decode
#         dec_out = self.dec(tgt=emb_seq, memory=latent_out,
#                            tgt_mask=mask, memory_mask=mask,
#                            tgt_key_padding_mask=pad_mask,
#                            memory_key_padding_mask=pad_mask)
#         dec_out = self.decode_out(dec_out).masked_fill(out_mask==1,-1e9) 
        
#         if len(latent_vec.shape)==1: latent_vec = latent_vec.unsqueeze(0)
#         if len(proj_out.shape)==1: proj_out = latent_vec.unsqueeze(0)
#         if len(dec_out.shape)==1: dec_out = dec_out.unsqueeze(0)
            
#         return latent_vec, proj_out, dec_out    
    
    
    
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Utility functions # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_downstream_mask(seq):
    _, ls = seq.size()
    dvc = seq.device
    mask = torch.triu(torch.ones((ls, ls), device=dvc), diagonal=1).bool()
    return mask

def count_params(model):
    return sum([p.nelement() for p in model.parameters()])




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Auxillary nn modules  # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(MAX_VEC_LEN, d_model)
        position = torch.arange(0, MAX_VEC_LEN, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() \
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)
    
# TODO: Make mutable through hyperparams. See Travis's property pred nn code.
class ProjHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc_layer1 = nn.Linear(dim, dim)
        self.relu = nn.ReLU(inplace=False)
        self.fc_layer2 = nn.Linear(dim, dim)
    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.fc_layer2( self.relu(x) )
        x_normed = F.normalize(x, p=2.0, dim=-1)
        return x_normed
    

# Set-wise attention  # # # # # # # # # # # # # # # # # # # # # # # # #
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
        
        
        
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Loss functions  # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# import math
# import random
# from tqdm.auto import trange, tqdm

def CrossEntropyLoss(dec_out, seq, pad_mask, out_mask, **kwargs):
    pad_mask = torch.ones_like(pad_mask.float()) - pad_mask.float()
    pad_mask = torch.flatten(pad_mask[:,1:]).float()
    weights = torch.ones_like(out_mask[0,0]) - out_mask[0,0]
    criterion = nn.CrossEntropyLoss(reduction = 'none',weight = weights)
    unred_loss = criterion(torch.flatten(dec_out[:,:-1],
                                         end_dim = -2),
                           torch.flatten(seq[:,1:]))
    return torch.matmul(unred_loss,pad_mask)/torch.sum(pad_mask)

import torch
import torch.nn as nn
class SupConLoss(nn.Module): 
    def __init__(self, temp=0.07, contrast_mode='one', base_temp=0.07):
        super(SupConLoss, self).__init__()

        self.temperature = temp
        self.contrast_mode = contrast_mode
        self.base_temp = base_temp

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).\
                                    view(-1, 1).to(device), 0 )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temp) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss