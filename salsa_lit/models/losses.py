"""
Loss functions for SALSA training.
"""

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Pulls together samples from the same family (anchor + mutants)
    while pushing apart samples from different families.

    Reference: https://arxiv.org/abs/2004.11362

    Parameters
    ----------
    temp : float
        Temperature for softmax scaling
    contrast_mode : str
        'one' or 'all' - whether to use one or all views as anchors
    base_temp : float
        Base temperature for loss scaling
    """

    def __init__(self, temp: float = 0.07, contrast_mode: str = 'one', base_temp: float = 0.07):
        super().__init__()
        self.temperature = temp
        self.contrast_mode = contrast_mode
        self.base_temp = base_temp

    def forward(self, features: torch.Tensor, labels: torch.Tensor = None, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Parameters
        ----------
        features : torch.Tensor
            Hidden vectors of shape (bsz, n_views, dim)
        labels : torch.Tensor
            Ground truth labels of shape (bsz,)
        mask : torch.Tensor, optional
            Contrastive mask of shape (bsz, bsz)

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value
        """
        device = features.device if features.is_cuda else torch.device('cpu')

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

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -(self.temperature / self.base_temp) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def cross_entropy_loss(dec_out: torch.Tensor, seq: torch.Tensor, 
                       pad_mask: torch.Tensor, out_mask: torch.Tensor, 
                       **kwargs) -> torch.Tensor:
    """
    Masked cross-entropy loss for sequence reconstruction.

    Parameters
    ----------
    dec_out : torch.Tensor
        Decoder output logits, shape (batch, seq_len, n_tokens)
    seq : torch.Tensor
        Target token indices, shape (batch, seq_len)
    pad_mask : torch.Tensor
        Padding mask, shape (batch, seq_len)
    out_mask : torch.Tensor
        Output mask for special tokens, shape (batch, 1, n_tokens) or (batch, n_tokens)

    Returns
    -------
    loss : torch.Tensor
        Scalar loss value
    """
    # Invert pad mask (1 for real tokens, 0 for padding)
    pad_mask = torch.ones_like(pad_mask.float()) - pad_mask.float()
    pad_mask = torch.flatten(pad_mask[:, 1:]).float()

    # Get weights from out_mask - handle both 2D and 3D
    if out_mask.dim() == 3:
        weights = torch.ones_like(out_mask[0, 0]) - out_mask[0, 0]
    else:
        weights = torch.ones_like(out_mask[0]) - out_mask[0]

    criterion = nn.CrossEntropyLoss(reduction='none', weight=weights)
    unred_loss = criterion(
        torch.flatten(dec_out[:, :-1], end_dim=-2),
        torch.flatten(seq[:, 1:])
    )

    return torch.matmul(unred_loss, pad_mask) / torch.sum(pad_mask)


def aligniform_loss(latent: torch.Tensor, alpha: float = 2, t: float = 2) -> torch.Tensor:
    """
    Alignment + Uniformity loss for contrastive learning.

    Alignment loss encourages similar samples to have similar representations.
    Uniformity loss encourages the distribution of representations to be uniform.

    Parameters
    ----------
    latent : torch.Tensor
        Latent representations, shape (batch, fam_size, dim)
    alpha : float
        Power for alignment loss
    t : float
        Temperature for uniformity loss

    Returns
    -------
    loss : torch.Tensor
        Combined alignment and uniformity loss
    """
    def _uniform_loss(latent, t=2):
        x = latent[:, 0, :]
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def _align_loss(latent, alpha=2):
        n_augs = latent.shape[1] - 1
        combos = [[0, i] for i in range(1, n_augs + 1)]
        tens = latent[:, combos].flatten(0, 1)
        x = tens[:, 0, :]
        y = tens[:, 1, :]
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    aloss = _align_loss(latent, alpha)
    uloss = _uniform_loss(latent, t)

    return aloss + uloss
