import sys
sys.path.insert(0, '..')
import random
import numpy as np
import pandas as pd
import os
import sys
import csv
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from salsa.constants import NAMED_LOSSES
from salsa.modules import SupConLoss, CrossEntropyLoss

from collections import OrderedDict
def get_loss_data(losses, samp, inter_latent, dec_out, fams_per_batch):
    loss_dict = OrderedDict(zip(NAMED_LOSSES, torch.full((len(NAMED_LOSSES),1), False) ))
    if 'SupCon' in losses:
        fam_ids = torch.tensor(range(fams_per_batch))
        loss_dict['SupCon'] = SupConLoss()(features=proj_out, labels=fam_ids)
    if 'Recon' in losses:
        loss_dict['Recon'] = CrossEntropyLoss(dec_out, **samp)  
    loss = sum(loss_dict[k] for k in list(set(losses)))
    loss_items = {k:v.item() for k,v in loss_dict.items()}
    return (loss, [v for k,v in loss_items.items()])


def fit(model, optimizer, device, loader, version, batch_size, 
        lr=0.00001, losses=NAMED_LOSSES, n_epochs=1, save_step=10000):
        
#     device = torch.device("cuda" if use_cuda else "cpu")
#     torch.cuda.empty_cache()
#     if use_cuda:
#         print(f"Using {len(use_cuda)} GPUs: {use_cuda}!")
#         model = torch.nn.DataParallel(model, device_ids = use_cuda)
#         model.to(device)
#     else: 
#         model = model.to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
#     model.train()
#     torch.set_grad_enabled(True)

    today = datetime.today().strftime('%Y%m%d%H')
    tag = f'{today}_{version}'
    model_dir = os.path.join('..','models',tag) 
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        message = f"Created new directory '{model_dir}'!"
    else: 
        message = f"Directory '{model_dir}' exists!"
    print(f"{message} \n Preparing to train model with tag '{tag}' ...")
    loss_csv = f'../logs/training_losses/losses_{tag}.csv'
    
    start = datetime.now()
    s = start    
    with open(loss_csv,'w',newline='') as output_file:
        
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(['Loss'] + NAMED_LOSSES)  
        loss_data = []
        
        for i in range(n_epochs):
            for j,samp in tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {i}"):     
                optimizer.zero_grad()
                for k,v in samp.items():
                    if torch.is_tensor(v):
                        samp[k] = v.to(device)

                salsa_latent, inter_latent, dec_out = model.forward(**samp)
                loss, run_data = get_loss_data(losses, samp, inter_latent, 
                                               dec_out, sampler.fams_per_batch)
                loss = loss.to(device).float()
                loss_data.append(run_data)
                writer.writerow([loss.item()] + run_data)
                
                loss.backward()
                optimizer.step()
                
                # Save models at specified steps.
                if j%save_step==0 or j==(len(loader)-1):
                    try:
                        state_dict = model.module.state_dict()
                    except AttributeError:
                        state_dict = model.state_dict()
                    os.path.join(model_dir,f'{i:02}_{j:0{len(loader)}}.pt')
                    torch.save(state_dict, model_path)
              
            # Report progress.
            e = datetime.now()
            lap = e - s
            s = e
            hours, minutes, seconds = get_time(lap.seconds)
            print(f"Epoch done. Runtime: {hours} hours, {minutes} mins , {seconds} secs.") 
            try: state_dict = model.module.state_dict()
            except AttributeError: state_dict = model.state_dict()

            if i+1 == n_epochs:
                lap = e - start
                # h, m, s = lap.seconds//3600, lap.seconds//60%60, lap.seconds%60
                hours, minutes, seconds = get_time(lap.seconds)
                print(f"All done. Total runtime: {hours} hr {minutes}  min {seconds} sec.")
                torch.save(state_dict, model_path)
                
    return model


def get_time(seconds):
    seconds_left = seconds
    hours = seconds_left // 3600
    seconds_left = seconds_left % 3600
    minutes = seconds_left // 60
    seconds_left = minutes % 60
    seconds = seconds_left
    return hours, minutes, seconds


