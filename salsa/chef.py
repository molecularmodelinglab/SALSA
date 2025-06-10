import pickle
from typing import List, Literal

import yaml
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from salsa.datasets import TrainingDataset, InferenceDataset, RandomFamSampler
from salsa.modules import SalsaNet, count_params
from salsa.modeling import fit

import torch
from torch.utils.data import DataLoader


def serve_salsa(smiles_file: str, model: str or SalsaNet, param_file: str = None,
                output_dir: str = None, batch_size: int = 50):
    """

    Parameters
    ----------
    smiles_file
    model
    param_file
    output_dir
    batch_size

    Returns
    -------

    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.getcwd(), f"latents_{datetime.now().strftime('%Y%m%d%H%M%S')}"))
        os.makedirs(output_dir, exist_ok=False)

    ds_salsa = InferenceDataset(smiles_file)

    loader = DataLoader(ds_salsa, batch_size=batch_size, shuffle=False, drop_last=False)

    if not isinstance(model, SalsaNet):
        if param_file is None:
            raise ValueError("must pass param yaml file for model")
        params = yaml.load(open(param_file, "r"), Loader=yaml.FullLoader)
        weights = model
        model = SalsaNet(**params)
        model.load_state_dict(torch.load(weights), strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_file = open(os.path.join(output_dir, "latents.csv"), "w")

    for samp in tqdm(loader, total=len(loader)):
        for k, v in samp.items():
            if torch.is_tensor(v): samp[k] = v.to(device)
        _, inter_latent = model.forward(**samp)
        salsa_latent = inter_latent.cpu().detach().numpy()
        _smiles = samp["smiles"]
        for i in range(len(_smiles)):
            output_file.write(f"{_smiles[i]},{','.join(map(str, salsa_latent[i]))}\n")


def cook_salsa(
        train_csv: str or TrainingDataset or pd.DataFrame,
        save_dir: str = None,
        use_cuda: int = 0,  # -1 is use cpu to avoid 0 being "False"
        batch_size: int = 50,
        lr: float = 0.00001,
        losses: List[Literal['Recon', 'SupCon']] = ('Recon', 'SupCon'),
        n_epochs: int = 1,
        dim_emb: int = 512,
        dim_salsa: int = 32,
        n_heads: int = 8,
        L_enc: int = 8,
        L_dec: int = 8,
        dim_ff: int = 2048,
        proj_head: bool = False,
        pool_func: bool = 'avg',
        actv: Literal["relu", "gelu"] = 'relu',
        save_step: int = 10000
):
    """
    Train ("cook") a SALSA model using the passed parameters and a SALSA training set

    Parameters
    ----------
    train_csv: str or TrainingDataset or pd.DataFrame
        the training dataset to use for training
        can be the path, the loaded csv as a pandas dataframe, or the TrainingDataset pytorch object
    save_dir: str, optional
        the path to where the model(s) should be saved
        defaults to the CDW in a folder named SALSA_<current_time>
        can also be a SalsaNet model object
    use_cuda: int = 0
        which CUDA device to use
        if you have one gpu, the gpu device id is 0
        -1 means cpu training
    batch_size: int
        the size of the batches to use for training
    lr: float
        the overall learning rate to use in the model
    losses: list
        which loss functions to use
        defaults to SupCon and Reconstruction
    n_epochs: int
        number of epochs to use for training
    dim_emb: int
        the latent dimension of the embedder
    dim_salsa: int
        the dimension of the latent space between embedder and decoder
        this is also the dimension of the SALSA descriptors
    n_heads: int
        number of attention heads to use in transformer layers
    L_enc: int
        number of encoder layers
    L_dec: int
        number of decoder layers
    dim_ff: int
        dimension of the feedforward network in the transformer layers
    proj_head: bool
        use the projection head or not
        THIS IS BROKEN, WILL BE IGNORED
    pool_func: literal['avg', 'max', 'set'] = "avg"
        the pooling function to use aggregate encoded vectors
    actv: literal['relu', 'gelu']
        the activation function to use in the encoding transformer layer(s)
    save_step: int
        number or training step to take inbetween saving model checkpoints

    Returns
    -------

    """

    if save_dir is None:
        save_dir = os.path.abspath(os.path.join(os.getcwd(), f"SALSA_{datetime.now().strftime('%Y%m%d%H%M%S')}"))
        os.makedirs(save_dir, exist_ok=False)

    if isinstance(train_csv, (str, pd.DataFrame)):
        training_data = TrainingDataset(train_csv)
    elif isinstance(train_csv, TrainingDataset):
        training_data = train_csv
    else:
        raise TypeError('train_csv must be path, pd.DataFrame or TrainingDataset object')

    fam_size = training_data.fam_size
    batch_size = batch_size - (batch_size % fam_size)

    _params = {
        'use_cuda': use_cuda,
        'batch_size': batch_size,
        'lr': lr,
        'losses': losses,
        'n_epochs': n_epochs,
        'dim_emb': dim_emb,
        'dim_salsa': dim_salsa,
        'n_heads': n_heads,
        'L_enc': L_enc,
        'L_dec': L_dec,
        'dim_ff': dim_ff,
        'proj_head': proj_head,
        'pool_func': pool_func,
        'actv': actv,
        'save_step': save_step
    }

    yaml.dump(_params, open(os.path.join(save_dir, 'params.yml'), 'w'))

    # sampler and data loader
    sampler = RandomFamSampler(training_data.anc_ids, batch_size=batch_size, drop_last=True)
    loader = DataLoader(training_data, batch_sampler=sampler, pin_memory=True)

    model = SalsaNet(dim_emb, dim_salsa, n_heads, L_enc, L_dec, dim_ff, fam_size, proj_head, pool_func, actv)

    losses = fit(model, save_dir, sampler, loader, use_cuda, lr, losses, n_epochs, save_step)
    pickle.dump(losses, open(os.path.join(save_dir, "losses.pkl"), "wb"))


if __name__ == '__main__':
    # cook_salsa("C:\\Users\\James\\OneDrive\\PopovLab\\Programs\\Popov-SALSA-Clean\\data\\test_data\\mutants_17095142105336978_12-ancs_5-muts_6.0-mthresh_chembl-weight.csv",
    #            use_cuda=-1,
    #            n_epochs=3)
    serve_salsa("../data/test_data/test_smiles.csv", "./SALSA_20240326000201/model_trained_model.pt",
                "./SALSA_20240326000201/params.yml")