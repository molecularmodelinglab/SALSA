"""
I/O utility functions for SALSA.
"""

import os
import glob
import time
import random
import numpy as np
import pandas as pd


DEFAULT_FORMAT = 'parquet'
COMPRESSION = 'zstd'
FORMAT_EXTS = {
    'parquet': ('.parquet', '.pq'),
    'csv': ('.csv',),
}

# Placeholder strings the mutation writers put in otherwise-empty cells
NA_VALUES = ('None',)

# Rows buffered before a ChunkWriter flushes a chunk to disk.
CHUNK_ROWS = 50_000


def timestamp():
    """
    Generate a timestamp string for file naming.

    Returns
    -------
    ts : str
        Timestamp string with no decimal point
    """
    return str(time.time()).replace(".", "")


def infer_format(path, default=DEFAULT_FORMAT):
    """
    Determine a dataset's format from its file extension.

    Parameters
    ----------
    path : str
        Path to a dataset file
    default : str or None
        Format to return when the extension is not recognised

    Returns
    -------
    fmt : str or None
        'parquet', 'csv', or `default`
    """
    ext = os.path.splitext(str(path))[1].lower()
    for fmt, exts in FORMAT_EXTS.items():
        if ext in exts:
            return fmt
    return default


def with_ext(stem, fmt=DEFAULT_FORMAT):
    """
    Append the canonical extension for a format to a filename stem.

    Parameters
    ----------
    stem : str
        Filename without extension
    fmt : str
        'parquet' or 'csv'

    Returns
    -------
    name : str
        Filename with the format's extension

    Raises
    ------
    ValueError
        If the format is not recognised.
    """
    if fmt not in FORMAT_EXTS:
        raise ValueError(f"Unknown format: {fmt!r} (valid: {list(FORMAT_EXTS)})")
    return stem + FORMAT_EXTS[fmt][0]


def dataset_paths(directory):
    """
    List dataset files in a directory, in any supported format.

    Parameters
    ----------
    directory : str
        Directory to search

    Returns
    -------
    paths : list of str
        Sorted paths of all parquet and CSV files found
    """
    paths = []
    for exts in FORMAT_EXTS.values():
        for ext in exts:
            paths.extend(glob.glob(os.path.join(directory, '*' + ext)))
    return sorted(paths)


def read_ds(dataset, columns=None):
    """
    Read a dataset from file or return as-is if already a DataFrame.

    Parameters
    ----------
    dataset : str or pd.DataFrame
        Path to a parquet or CSV file, or a DataFrame object
    columns : list of str, optional
        Read only these columns. Much cheaper than reading everything and
        subsetting afterwards, especially for parquet.

    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset if columns is None else dataset[list(columns)]

    cols = list(columns) if columns is not None else None

    if infer_format(dataset) == 'parquet':
        df = pd.read_parquet(dataset, columns=cols)
    else:
        df = pd.read_csv(dataset, usecols=cols)

    if cols is not None:
        df = df[cols]

    return df


def write_ds(df, path, fmt=None, na_values=NA_VALUES):
    """
    Write a dataset, choosing the format from the path or `fmt`.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write
    path : str
        Output path. Its extension selects the format when recognised.
    fmt : str, optional
        Format to use when `path` has no recognised extension.
    na_values : tuple of str or None
        Placeholder strings to store as real nulls in parquet. CSV keeps
        writing them literally, since reading a CSV turns them back into NaN
        anyway. Pass None to write values through untouched.

    Returns
    -------
    path : str
        The path written
    """
    fmt = infer_format(path, default=fmt or DEFAULT_FORMAT)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if fmt == 'parquet':
        frame = _nullify(df, na_values)
        frame.to_parquet(path, engine='pyarrow', compression=COMPRESSION,
                         index=False)
    else:
        df.to_csv(path, index=False, na_rep='None')

    return path


def write_embeddings(path, smiles, embeddings, fmt=None):
    """
    Write SALSA embeddings as a SMILES column plus one column per dimension.

    Parameters
    ----------
    path : str
        Output path. Its extension selects the format when recognised.
    smiles : sequence of str
        SMILES strings, one per embedding row
    embeddings : np.ndarray
        Embeddings of shape (n_molecules, dim_salsa)
    fmt : str, optional
        Format to use when `path` has no recognised extension.

    Returns
    -------
    path : str
        The path written
    """
    # pinned to float32
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got shape {emb.shape}")

    df = pd.DataFrame(emb, columns=[f'dim_{i}' for i in range(emb.shape[1])])
    df.insert(0, 'SMILES', list(smiles))

    return write_ds(df, path, fmt=fmt, na_values=None)


def _nullify(df, na_values):
    """
    Replace placeholder strings with real nulls.
    """
    if not na_values:
        return df

    hits = df.isin(list(na_values))
    if not hits.to_numpy().any():
        return df

    return df.mask(hits)


class ChunkWriter:
    """
    Incremental dataset writer that flushes buffered rows in chunks.

    Parameters
    ----------
    path : str
        Output path. Its extension selects the format when recognised.
    columns : list of str
        Column names for the rows being written
    fmt : str, optional
        Format to use when `path` has no recognised extension.
    dtypes : dict, optional
        Column dtypes applied to every chunk. Needed for parquet: the schema
        is fixed by the first chunk, so a later chunk whose values happen to
        infer differently would otherwise be cast to the wrong type.
    chunk_rows : int
        Rows to buffer before flushing
    na_values : tuple of str or None
        Placeholder strings to store as real nulls in parquet.

    Notes
    -----
    Use as a context manager, or call close() when done. An instance that is
    never written to still produces an empty file with the right columns.
    """

    def __init__(self, path, columns, fmt=None, dtypes=None,
                 chunk_rows=CHUNK_ROWS, na_values=NA_VALUES):
        self.path = path
        self.columns = list(columns)
        self.fmt = infer_format(path, default=fmt or DEFAULT_FORMAT)
        self.dtypes = dtypes
        self.chunk_rows = chunk_rows
        self.na_values = na_values

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self._buf = []
        self._handle = None
        self._writer = None
        self._schema = None
        self._wrote_any = False

    def write(self, rows):
        """
        Buffer rows, flushing once the buffer reaches chunk_rows.

        Parameters
        ----------
        rows : list of list
            Rows to write, each matching `columns`
        """
        self._buf.extend(rows)
        if len(self._buf) >= self.chunk_rows:
            self.flush()

    def flush(self):
        """Write any buffered rows to disk."""
        if not self._buf:
            return

        frame = pd.DataFrame(self._buf, columns=self.columns)
        if self.dtypes:
            frame = frame.astype(self.dtypes)
        self._buf = []

        if self.fmt == 'parquet':
            self._write_parquet(frame)
        else:
            self._write_csv(frame)

        self._wrote_any = True

    def close(self):
        """Flush and release the underlying handle."""
        self.flush()

        if not self._wrote_any:
            empty = pd.DataFrame(columns=self.columns)
            if self.dtypes:
                empty = empty.astype(self.dtypes)
            write_ds(empty, self.path, fmt=self.fmt, na_values=self.na_values)

        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _write_parquet(self, frame):
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(_nullify(frame, self.na_values),
                                     preserve_index=False)

        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema,
                                            compression=COMPRESSION)
        elif table.schema != self._schema:
            table = table.cast(self._schema)

        self._writer.write_table(table)

    def _write_csv(self, frame):
        if self._handle is None:
            self._handle = open(self.path, 'w')
            frame.to_csv(self._handle, index=False, na_rep='None')
        else:
            frame.to_csv(self._handle, index=False, header=False, na_rep='None')
        self._handle.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


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
