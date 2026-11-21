"""Tests for salsa_lit.pretrained: the registry and model resolution."""

import os
import re

import pytest

from salsa_lit import pretrained
from salsa_lit.cli.main import main
from salsa_lit.pretrained import (
    DEFAULT_MODEL, HF_REPO_ID, HF_REVISION, PRETRAINED, download_pretrained,
    list_pretrained, model_names, resolve_model_path
)


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the hub call with a recorder, so no test touches the network."""
    calls = []

    def record(filename, revision=None, cache_dir=None):
        calls.append({'filename': filename, 'revision': revision,
                      'cache_dir': cache_dir})
        return f'/fake/cache/{filename}'

    monkeypatch.setattr(pretrained, '_hf_download', record)
    return calls


def test_default_model_is_in_the_registry():
    """DEFAULT_MODEL names a real entry."""
    assert DEFAULT_MODEL in PRETRAINED


def test_every_entry_has_the_fields_the_resolver_reads():
    """Each registry entry carries a filename and a dimension."""
    for name, spec in PRETRAINED.items():
        assert isinstance(spec['filename'], str) and spec['filename']
        assert isinstance(spec['dim_salsa'], int) and spec['dim_salsa'] > 0


def test_filenames_are_unique():
    """No two entries point at the same file."""
    filenames = [spec['filename'] for spec in PRETRAINED.values()]
    assert len(filenames) == len(set(filenames))


def test_entry_dimension_matches_its_name_and_filename():
    """A name like salsa24 carries dim_salsa 24, and so does its filename."""
    for name, spec in PRETRAINED.items():
        from_name = int(re.fullmatch(r'salsa(\d+)', name).group(1))
        from_file = int(re.match(r'salsa(\d+)_', spec['filename']).group(1))
        assert from_name == spec['dim_salsa']
        assert from_file == spec['dim_salsa']


def test_revision_is_pinned_to_a_commit():
    """HF_REVISION is a full commit sha, not a mutable branch name."""
    assert re.fullmatch(r'[0-9a-f]{40}', HF_REVISION)


def test_model_names_are_ordered_by_dimension():
    """model_names sorts by embedding dimension, not alphabetically."""
    dims = [PRETRAINED[n]['dim_salsa'] for n in model_names()]
    assert dims == sorted(dims)


def test_list_pretrained_returns_copies():
    """Mutating the result does not corrupt the registry."""
    listed = list_pretrained()
    listed[DEFAULT_MODEL]['filename'] = 'tampered'
    assert PRETRAINED[DEFAULT_MODEL]['filename'] != 'tampered'


def test_existing_path_is_returned_unchanged(tmp_path, fake_download):
    """A real path wins over the registry and triggers no download."""
    ckpt = tmp_path / 'local.ckpt'
    ckpt.write_bytes(b'')
    assert resolve_model_path(str(ckpt)) == str(ckpt)
    assert fake_download == []


def test_existing_directory_is_returned_unchanged(tmp_path, fake_download):
    """A model directory resolves to itself."""
    assert resolve_model_path(str(tmp_path)) == str(tmp_path)
    assert fake_download == []


def test_none_resolves_to_the_default_model(fake_download):
    """Omitting the model downloads DEFAULT_MODEL."""
    resolve_model_path(None)
    assert len(fake_download) == 1
    assert fake_download[0]['filename'] == PRETRAINED[DEFAULT_MODEL]['filename']


@pytest.mark.parametrize('name', sorted(PRETRAINED))
def test_registry_name_downloads_the_right_file(name, fake_download):
    """Each name fetches its own file at the pinned revision."""
    path = resolve_model_path(name)
    assert fake_download[0]['filename'] == PRETRAINED[name]['filename']
    assert fake_download[0]['revision'] is None
    assert path.endswith(PRETRAINED[name]['filename'])


def test_revision_and_cache_dir_are_forwarded(fake_download):
    """Explicit revision and cache_dir reach the hub call."""
    resolve_model_path(DEFAULT_MODEL, revision='abc123', cache_dir='/tmp/x')
    assert fake_download[0]['revision'] == 'abc123'
    assert fake_download[0]['cache_dir'] == '/tmp/x'


def test_unknown_spec_raises_listing_the_options(fake_download):
    """An unusable spec names the valid models and says it is not a path."""
    with pytest.raises(ValueError) as exc:
        resolve_model_path('salsa999')
    message = str(exc.value)
    assert 'salsa999' in message
    assert 'path' in message
    for name in model_names():
        assert name in message
    assert fake_download == []


def test_download_pretrained_rejects_unknown_names(fake_download):
    """download_pretrained validates before reaching the network."""
    with pytest.raises(ValueError):
        download_pretrained('nope')
    assert fake_download == []


def test_load_model_rejects_unknown_names(fake_download):
    """load_model routes through the resolver, so bad names fail early."""
    from salsa_lit.api.inference import load_model

    with pytest.raises(ValueError):
        load_model('not_a_model')
    assert fake_download == []


def test_serve_salsa_accepts_a_model_object_without_downloading(small_model,
                                                                short_smiles,
                                                                fake_download):
    """Passing a model object bypasses resolution entirely."""
    from salsa_lit.api.inference import serve_salsa

    embeddings, _ = serve_salsa(short_smiles, small_model, device='cpu')
    assert embeddings.shape[0] == len(short_smiles)
    assert fake_download == []


def test_cli_download_list_runs(capsys):
    """download --list prints every model and the default marker."""
    main(['download', '--list'])
    out = capsys.readouterr().out
    for name in model_names():
        assert name in out
    assert 'default' in out


def test_cli_download_fetches_named_model(fake_download, capsys):
    """download NAME fetches just that model."""
    main(['download', 'salsa32'])
    assert len(fake_download) == 1
    assert fake_download[0]['filename'] == PRETRAINED['salsa32']['filename']
    assert 'salsa32' in capsys.readouterr().out


def test_cli_download_all_fetches_everything(fake_download):
    """download --all fetches one file per registry entry."""
    main(['download', '--all'])
    fetched = {call['filename'] for call in fake_download}
    assert fetched == {spec['filename'] for spec in PRETRAINED.values()}


def test_cli_download_defaults_to_the_default_model(fake_download):
    """download with no name fetches DEFAULT_MODEL."""
    main(['download'])
    assert len(fake_download) == 1
    assert fake_download[0]['filename'] == PRETRAINED[DEFAULT_MODEL]['filename']


@pytest.mark.skipif(os.environ.get('SALSA_TEST_NETWORK') != '1',
                    reason='set SALSA_TEST_NETWORK=1 to download from the hub')
def test_real_download_and_embed(tmp_path, short_smiles):
    """End to end against the published repo, including the model dimension."""
    from salsa_lit.api.inference import serve_salsa

    embeddings, smiles = serve_salsa(short_smiles, DEFAULT_MODEL, device='cpu',
                                     cache_dir=str(tmp_path))
    assert smiles == short_smiles
    assert embeddings.shape == (len(short_smiles),
                                PRETRAINED[DEFAULT_MODEL]['dim_salsa'])
