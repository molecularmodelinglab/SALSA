import random
import pandas as pd
import csv
import numpy as np
np.set_printoptions(suppress=True,formatter={'float_kind': '{:f}'.format})
import os
import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")


#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# RDKit VS Boxplots # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from matplotlib.patches import PathPatch
import seaborn as sns
import matplotlib.pyplot as plt

def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# kNN-Recall benchmark  # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import seaborn as sns
import matplotlib.pyplot as plt 

def plot_knn_recall(nns, ds_name, neigh_radii, save=True):

    points = {}
    for m,rads in nns.items():
        points[m] = {}
        for rad in rads:
            points[m][rad] = np.mean(np.asarray(nns[m][rad]))

    sns.set_theme(style='ticks',font_scale=1.25,palette='muted')

    date = datetime.today().strftime('%Y%m%d%H%M')

    plt.figure(figsize=(8,7))

    sns.lineplot(points,markers=False,legend=False, linewidth=2.5)
    sns.scatterplot(points, s=200)

    sns.despine()
    ax = plt.gca()
    ax.grid(which='major', axis='y', linestyle=':')
    plt.legend(markerscale=2)
    plt.ylabel(f"Recall of {neigh_radii[0]} mutants")
    plt.xlabel("k neighbors")
    plt.title(f"Dataset: {ds_name}")
    plt.xticks(neigh_radii)
    # plt.legend(font_scale=.75)
    plt.ylim((.4,1.03))

    if save:
        t = f'{date}_manySalsas_{ds_name}'
        plt.savefig(f'../figs/knns/{t}.png',bbox_inches='tight') 

    plt.show()




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MAHALANOBIS PROP DISTANCES  # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
sns.set_theme(style='ticks',font_scale=1.5,palette='muted')
def get_mahalanobis_histplot(ds_name, mode, xlims=None, bins=None, out_name=None, save=True):  
    ''' 
    This function plots the distribution of anc-mut Mahalanobis distances. 
        Can be furthered specified by atom_type or action of each mutation. 
        Args:
            ds_path: complete path to MUTATED dataset.  
            mode: 'vanilla', 'atoms', 'actions'
            out_name: name of output, if not provided saves to fig dir.
    '''
    
    df = pd.read_csv(ds_name)    
    dff = df[df.Is_anc==False]
    
    if xlims:
        dff = dff[dff.Prop_dist <= xlims[1]]
    if bins is None:
        bins = 'auto'
    
    plt.figure(figsize=(20,5))
    if mode=='vanilla':
        sns.histplot(dff, 
                     x="Prop_dist", 
                     element="step", 
                     bins=bins,
                     fill=False) 
    elif mode=='actions':
        sns.histplot(dff, 
                     x="Prop_dist", 
                     hue="Action",
                     element="step", 
                     bins=bins,
                     multiple="stack",
                     palette='rainbow',
                     hue_order=['Add','Replace','Remove']) 
    elif mode=='atoms':
        dff = dff[dff.Action.isin(['Replace','Add'])]
        sns.histplot(dff, 
                     x="Prop_dist", 
                     hue="To_atom",
                     element="step", 
                     bins=bins,
                     multiple="stack",
                     palette='turbo',
                     hue_order=['C','O','N','S','F','Cl','Br','I','P'])         
        
    sns.despine()
    ax = plt.gca()
    ax.grid(which='major', axis='x', linestyle='--')
    plt.xlabel('Mahalanobis distance',labelpad=15)
    plt.ylabel('Count',labelpad=15)
    if xlims:
        plt.xlim(xlims[0], xlims[1])
        plt.xticks(np.arange(xlims[0], xlims[1], 0.5))

    if out_name is None:
        tag = ds_name[ds_name.rfind('/')+1:ds_name.find('.csv')]
        dt = datetime.today().strftime('%Y%m%d-%Hh%Mm')
        out_name = f"{tag}_{mode}_{dt}"
        out_path = f"../figs/mahala_distplots/{out_name}.png"
        
    if save:    
        print(f"Saving plot to path {out_path}")
        plt.savefig(out_path, bbox_inches='tight')
    else:
        print("Not saving plot!")
        
    plt.show()


    
# import matplotlib.font_manager
# flist = matplotlib.font_manager.get_fontconfig_fonts()
# names = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]
from PIL import Image
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height)).convert('RGBA')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

from rdkit import Chem
from rdkit.Chem import AllChem,Draw

def draw_mol_pic(df, anc_id):
    from molecule_slide_generator import SlideGenerator
    dft = df[df.Anc_id==anc_id]
    all_props = []
    mols = []
    for row in dft.itertuples():
        mol = Chem.MolFromSmiles(row.Smiles)
        AllChem.Compute2DCoords(mol)
        prop = row.Prop_dist
        if prop == -1: color = '#FFFFFF'
        elif prop > 11.0: color = '#FF0000'
        elif prop > 8.0: color = '#FE4500'
        elif prop > 7.0: color = '#FB7860B'
        elif prop > 6.0: color = '#808000'
        elif prop > 5.0: color = '#4682B4'
        elif prop > 4.5: color = '#008080'
        elif prop < 2.0: color = '#33CD32'    
        elif prop >= 2.0: color = '#228B21'
        prop = f'{prop}'
        props = [TextProperty('Dist', prop, color=color)]
        all_props.append(props)
        mols.append(mol)

    ft = '/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf'
    ## Anchor
    sg = SlideGenerator(mols_per_row=1, rows=1, font_size=18,font = ft,
                        number_of_properties=1, slide_width=1000, slide_height=200)
    png1 = sg.generate_slide(mols[:1], all_props[:1], f'../figs/scratch/{anc_id}_1.png') 
    im1 = Image.open(f'../figs/scratch/{anc_id}_1.png').convert('RGBA')
    os.system(f'rm ../figs/scratch/{anc_id}_1.png')
    
    ## Mutants
    sg = SlideGenerator(mols_per_row=5, rows=2, font_size=12, font = ft,
                        number_of_properties=1, slide_width=1000, slide_height=400)
    png2 = sg.generate_slide(mols[1:], all_props[1:], f'../figs/scratch/{anc_id}_2.png') 
    im2 = Image.open(f'../figs/scratch/{anc_id}_2.png').convert('RGBA')
    os.system(f'rm ../figs/scratch/{anc_id}_2.png')
    
    time_tag = datetime.today().strftime('%Y%m%d%H%M%S')
    get_concat_v(im1, im2).save(f'../figs/rdkit_outs/{time_tag}_{anc_id}.png')
    im3 = Image.open(f'../figs/rdkit_outs/{time_tag}_{anc_id}.png').convert('RGBA')
    display(im3) 
    
    
    
    
#< ---- 20 char ---->< ---- 20 char ---->< ---- 20 char ---->< --- 18 char --->

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# TRAINING  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns 
import matplotlib.pyplot as plt
from datetime import datetime

LOSS_TO_C = {'Recon':'peru', 
             'Contrastive':'steelblue', 
             'Aligniform':'forestgreen',
             'Total':'black'}    
    
def show_losses(df, 
                t='SALSA',
                dpi=75,
                save=False, 
                y_lim=None, 
                y_log=False, 
                leg=True, 
                lim=None):
    
    sns.set_theme(style='ticks',font_scale=1.25)
    df = df.rename({'Loss':'Total'},axis=1)
    df = df.rename({'SupCon':'Contrastive'},axis=1)

    df = df.loc[:, (df != False).any(axis=0)]
    if lim: df = df[:lim]
    loss_names = df.columns.tolist()
    
    figsize=(12.5,5)
    
    plt.figure(figsize=figsize,dpi=dpi)
    for label,data in df.iteritems():
        plt.plot(data, label=label, color=LOSS_TO_C[label],alpha=1)
        
    if y_log: 
        plt.yscale('log')
        plt.title(f'{t} (Log Space)')
#     print(y_lim)
    if y_lim: plt.ylim(*y_lim)
        
    plt.grid(True)
    plt.xlabel('Batch run')
    
    if leg: plt.legend(loc='upper right')
    if not y_log: plt.title(f'{t}')
        
    if save:
        time_tag = datetime.today().strftime('%Y-%m-%d-%Hh%Mm%Ss'  )
        fname = f'{tag}_{len(df)}_losses_{time_tag}.png'
        png_out = os.path.join(HOME,'plots','loss_curves',fname)
        plt.savefig(png_out, bbox_inches='tight')
        
    plt.show()