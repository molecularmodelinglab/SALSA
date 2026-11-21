"""
Pretrained SALSA models published on the Hugging Face Hub.
"""

import os
from typing import Optional

from huggingface_hub import hf_hub_download, try_to_load_from_cache


HF_REPO_ID = 'molecularmodelinglab/SALSA'

# Pinned so a later upload to the repo cannot change the weights under
# installs that already exist. Bump deliberately when publishing models.
HF_REVISION = 'e3919d5d9d371ea04a01b1ac904f7c2ec46edfec'

PRETRAINED = {
    'salsa8': {'filename': 'salsa8_step2300000_weights.ckpt', 'dim_salsa': 8},
    'salsa16': {'filename': 'salsa16_step3400000_weights.ckpt', 'dim_salsa': 16},
    'salsa24': {'filename': 'salsa24_step1600000_weights.ckpt', 'dim_salsa': 24},
    'salsa32': {'filename': 'salsa32_step1200000_weights.ckpt', 'dim_salsa': 32},
}

DEFAULT_MODEL = 'salsa16'


def model_names() -> list:
    """
    List the pretrained model names, ordered by embedding dimension.

    Returns
    -------
    names : list of str
        Registry names usable wherever a model is accepted
    """
    return sorted(PRETRAINED, key=lambda n: PRETRAINED[n]['dim_salsa'])


def list_pretrained() -> dict:
    """
    Describe the pretrained models.

    Returns
    -------
    models : dict
        Name to a copy of its registry entry, ordered by dimension
    """
    return {name: dict(PRETRAINED[name]) for name in model_names()}


def cached_path(name: str, revision: str = None, cache_dir: str = None) -> Optional[str]:
    """
    Return a model's path if it is already cached, without downloading.

    Parameters
    ----------
    name : str
        Pretrained model name
    revision : str, optional
        Repository revision. Defaults to the pinned HF_REVISION.
    cache_dir : str, optional
        Cache directory. Defaults to the Hugging Face cache.

    Returns
    -------
    path : str or None
        Path to the cached file, or None if it has not been downloaded
    """
    if name not in PRETRAINED:
        raise ValueError(_unknown_model_message(name))

    result = try_to_load_from_cache(
        repo_id=HF_REPO_ID,
        filename=PRETRAINED[name]['filename'],
        revision=revision or HF_REVISION,
        cache_dir=cache_dir,
    )
    return result if isinstance(result, str) else None


def download_pretrained(name: str = None, revision: str = None,
                        cache_dir: str = None) -> str:
    """
    Download a pretrained model, returning the path to the cached file.

    Already-cached files are returned without re-downloading.

    Parameters
    ----------
    name : str, optional
        Pretrained model name. Defaults to DEFAULT_MODEL.
    revision : str, optional
        Repository revision. Defaults to the pinned HF_REVISION.
    cache_dir : str, optional
        Cache directory. Defaults to the Hugging Face cache.

    Returns
    -------
    path : str
        Path to the downloaded checkpoint

    Raises
    ------
    ValueError
        If the name is not in the registry.
    """
    name = name or DEFAULT_MODEL
    if name not in PRETRAINED:
        raise ValueError(_unknown_model_message(name))

    return _hf_download(PRETRAINED[name]['filename'], revision=revision,
                        cache_dir=cache_dir)


def resolve_model_path(spec: str = None, revision: str = None,
                       cache_dir: str = None) -> str:
    """
    Turn a model specification into a local path, downloading if needed.

    A path that exists is returned unchanged, so a local checkpoint always
    takes precedence over a registry name.

    Parameters
    ----------
    spec : str, optional
        Pretrained model name, or path to a checkpoint or model directory.
        Defaults to DEFAULT_MODEL.
    revision : str, optional
        Repository revision. Defaults to the pinned HF_REVISION.
    cache_dir : str, optional
        Cache directory. Defaults to the Hugging Face cache.

    Returns
    -------
    path : str
        Local path to a checkpoint or model directory

    Raises
    ------
    ValueError
        If `spec` is neither an existing path nor a registry name.
    """
    if spec is None:
        return download_pretrained(DEFAULT_MODEL, revision=revision,
                                   cache_dir=cache_dir)

    if os.path.exists(spec):
        return spec

    if spec in PRETRAINED:
        return download_pretrained(spec, revision=revision, cache_dir=cache_dir)

    raise ValueError(_unknown_model_message(spec))


def _hf_download(filename: str, revision: str = None, cache_dir: str = None) -> str:
    """Fetch one file from the SALSA repo, returning its cached path."""
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        revision=revision or HF_REVISION,
        cache_dir=cache_dir,
    )


def _unknown_model_message(spec: str) -> str:
    """Build the error text for a specification that resolves to nothing."""
    return (f"{spec!r} is not a pretrained model name, and no file or "
            f"directory exists at that path. "
            f"Available models: {', '.join(model_names())}")
