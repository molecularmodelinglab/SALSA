import os
from datetime import datetime
from collections import OrderedDict
from typing import Literal, List

from tqdm import tqdm

import torch
from torch.cuda.amp import GradScaler
from torch import autocast

from salsa.constants import NAMED_LOSSES
from salsa.modules import SupConLoss, CrossEntropyLoss, get_aligniform_loss


def fit(model, checkpoint_dir, sampler, loader, use_cuda: int = 0, lr: float = 0.00001,
        losses: List[Literal['Recon', 'SupCon']] = NAMED_LOSSES, n_epochs=1,
        save_step=10000):
    """
    Train a salsa model

    Parameters
    ----------
    model : torch.nn.Module
        The model to train initialized as an object
    checkpoint_dir: str
        path to place to save checkpoints
    sampler: salsa.datasets.RandomFamSampler
        The family sampler for the dataloader (see object documentation for details)
    loader: torch.utils.data.DataLoader
        The dataloader wrapper around a training dataset
    use_cuda: int = 0
        Which cuda device to use for training
        -1 means cpu
    lr: float = 0.00001
        learning rate for training
    losses: List[Literal['Recon', 'SupCon']] = ['Recon', 'SupCon']
        which losses to use during training (either Recon, SupCon or both)
    n_epochs: int = 1
        number of training epochs
    save_step: int = 10000
        save checkpoint every `save_step` steps
    """

    device = torch.device("cpu") if use_cuda == -1 else torch.device(f"cuda:{use_cuda}")
    model.to(device)
    model.train()
    torch.set_grad_enabled(True)  # what does this do???
    
    start = datetime.now()

    loss_data = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler()

    for i in range(n_epochs):
        for j, samp in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {i}"):
            optimizer.zero_grad()
            # why 16-bit precision???
            # with autocast(device_type='cuda' if use_cuda == -1 else "cuda", dtype=torch.float16):
            for k, v in samp.items():
                if torch.is_tensor(v):
                    samp[k] = v.to(device)
            _, inter_latent, dec_out = model.forward(**samp)
            loss, run_data = get_loss_data(losses, samp, inter_latent, dec_out, sampler.fams_per_batch)
            loss = loss.to(device).float()
            loss_data.append(run_data)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Save models at specified steps.
            if ((j+1) % save_step) == 0:
                save_model(model, checkpoint_dir, f"epoch{i}_batch{j}")

        # Report progress.
        epoch_end = datetime.now()
        lap = epoch_end - start
        print(f"epoch {i} done in {str(lap)}")

        save_model(model, checkpoint_dir, f"epoch{i}")
    save_model(model, checkpoint_dir, f"trained_model")
    return loss_data


def get_loss_data(losses, samp, inter_latent, dec_out, fams_per_batch):
    loss_vals = torch.full((len(losses),1), False)
    loss_dict = OrderedDict(zip(losses, loss_vals))

    if 'SupCon' in losses:
        fam_ids = torch.tensor(range(fams_per_batch))
        loss_dict['SupCon'] = SupConLoss()(features=inter_latent, labels=fam_ids)
    if 'Recon' in losses:
        loss_dict['Recon'] = CrossEntropyLoss(dec_out, **samp)  
    if 'Aligniform' in losses:
        loss_dict['Aligniform'] = get_aligniform_loss(latent=inter_latent)

    loss = sum(loss_dict[k] for k in list(set(losses)))
    loss_items = {k: v.item() for k, v in loss_dict.items()}
    return loss, [v for _, v in loss_items.items()]


def save_model(model, checkpoint_dir, checkpoint_id):
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    path_to_wgts = os.path.join(checkpoint_dir, f"model_{checkpoint_id}.pt")
    torch.save(state_dict, path_to_wgts)
