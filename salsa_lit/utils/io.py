"""
I/O utility functions for SALSA.
"""

import time
import random
import pandas as pd


def timestamp():
    """
    Generate a timestamp string for file naming.
    
    Returns
    -------
    ts : str
        Timestamp string with no decimal point
    """
    return str(time.time()).replace(".", "")


def read_ds(dataset):
    """
    Read a dataset from file or return as-is if already a DataFrame.
    
    Parameters
    ----------
    dataset : str or pd.DataFrame
        Path to CSV file or DataFrame object
    
    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset
    else:
        return pd.read_csv(dataset)


def capped_sample(pop, samp_size, seed=None):
    """
    Sample from population, returning original if smaller than sample size.
    
    Parameters
    ----------
    pop : pd.DataFrame or list or array
        Population to sample from
    samp_size : int
        Desired sample size
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    samp : same type as pop
        Sampled subset or original population
    """
    if seed is not None:
        random.seed(seed)
    
    if isinstance(pop, pd.DataFrame):
        if len(pop) > samp_size:
            return pop.sample(samp_size)
        else:
            return pop
    else:
        if len(pop) > samp_size:
            return random.sample(list(pop), k=samp_size)
        else:
            return pop
