"""Tests for salsa_lit.api.inference: batch trimming and canonicalization."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import default_collate

from salsa_lit.api import inference
from salsa_lit.api.inference import load_model, serve_salsa, trim_collate
from salsa_lit.constants import MAX_VEC_LEN
from salsa_lit.data.datasets import InferenceDataset


EMB_ATOL = 1e-6

ORDER_SMILES = ['CCO', 'C' * 60, 'CC', 'C' * 40, 'O', 'C' * 80]


def assert_embeddings_agree(a, b, atol=EMB_ATOL):
    """Embeddings agree within float32 kernel noise, in value and direction."""
    a, b = np.asarray(a), np.asarray(b)
    np.testing.assert_allclose(a, b, rtol=0, atol=atol)
    cos = ((a * b).sum(axis=-1)
           / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)))
    assert cos.min() > 1 - 1e-6


def embed_both_ways(model, smiles, batch_size, monkeypatch):
    """Embed the same input with trimming on and off."""
    trimmed, smis_a = serve_salsa(smiles, model, batch_size=batch_size,
                                  device='cpu')
    monkeypatch.setattr(inference, 'trim_collate', default_collate)
    untrimmed, smis_b = serve_salsa(smiles, model, batch_size=batch_size,
                                    device='cpu')
    assert smis_a == smis_b
    return trimmed, untrimmed


def test_trim_collate_keeps_the_non_pad_prefix(tokenizer, mixed_smiles):
    """Trimmed width equals the longest unpadded sequence in the batch."""
    ds = InferenceDataset(pd.DataFrame({'SMILES': mixed_smiles}), tokenizer=tokenizer)
    items = [ds[i] for i in range(len(ds))]

    batch = trim_collate(items)

    expected = max(int((~item['pad_mask']).sum()) for item in items)
    assert batch['seq'].shape[1] == expected
    assert batch['pad_mask'].shape[1] == expected
    assert batch['avg_mask'].shape[1] == expected


def test_trim_collate_drops_only_padding(tokenizer, mixed_smiles):
    """No non-pad token is lost, and the kept columns are unchanged."""
    ds = InferenceDataset(pd.DataFrame({'SMILES': mixed_smiles}), tokenizer=tokenizer)
    items = [ds[i] for i in range(len(ds))]

    full = default_collate(items)
    trimmed = trim_collate(items)
    keep = trimmed['seq'].shape[1]

    assert torch.equal(trimmed['seq'], full['seq'][:, :keep])
    assert bool(full['pad_mask'][:, keep:].all())


def test_trim_collate_is_a_noop_at_full_width(tokenizer, max_len_smiles):
    """A batch holding a max-length SMILES is not trimmed at all."""
    ds = InferenceDataset(pd.DataFrame({'SMILES': [max_len_smiles, 'CCO']}),
                          tokenizer=tokenizer)
    batch = trim_collate([ds[i] for i in range(len(ds))])
    assert batch['seq'].shape[1] == MAX_VEC_LEN


def test_trim_collate_leaves_out_mask_alone(tokenizer, short_smiles):
    """out_mask is per-token, not per-position, so trimming must not touch it."""
    ds = InferenceDataset(pd.DataFrame({'SMILES': short_smiles}), tokenizer=tokenizer)
    items = [ds[i] for i in range(len(ds))]
    assert torch.equal(trim_collate(items)['out_mask'], default_collate(items)['out_mask'])


def test_trim_actually_shortens_a_short_batch(tokenizer, short_smiles):
    """A batch of short SMILES comes out far narrower than the pad width."""
    ds = InferenceDataset(pd.DataFrame({'SMILES': short_smiles}), tokenizer=tokenizer)
    batch = trim_collate([ds[i] for i in range(len(ds))])
    assert batch['seq'].shape[1] < MAX_VEC_LEN


@pytest.mark.parametrize('smiles_fixture', ['short_smiles', 'mixed_smiles'])
def test_embed_is_invariant_to_trailing_pad(small_model, tokenizer, smiles_fixture,
                                            request):
    """Embedding a trimmed batch matches embedding it full-width."""
    smiles = request.getfixturevalue(smiles_fixture)
    ds = InferenceDataset(pd.DataFrame({'SMILES': smiles}), tokenizer=tokenizer)
    items = [ds[i] for i in range(len(ds))]

    full = default_collate(items)
    trimmed = trim_collate(items)

    with torch.inference_mode():
        out_full, _ = small_model.embed(full['seq'], full['pad_mask'],
                                        full['avg_mask'])
        out_trim, _ = small_model.embed(trimmed['seq'], trimmed['pad_mask'],
                                        trimmed['avg_mask'])

    assert_embeddings_agree(out_full, out_trim)


@pytest.mark.parametrize('batch_size', [1, 2, 3, 8, 50])
def test_serve_salsa_trimmed_matches_untrimmed(small_model, mixed_smiles,
                                               monkeypatch, batch_size):
    """serve_salsa agrees with and without trimming, at every batch size."""
    trimmed, untrimmed = embed_both_ways(small_model, mixed_smiles, batch_size,
                                         monkeypatch)
    assert_embeddings_agree(trimmed, untrimmed)


@pytest.mark.parametrize('batch_size', [2, 8, 50])
def test_batch_size_perturbs_results_without_trimming_too(small_model, mixed_smiles,
                                                          monkeypatch, batch_size):
    """Batch size alone perturbs results even with trimming disabled."""
    monkeypatch.setattr(inference, 'trim_collate', default_collate)
    ref, _ = serve_salsa(mixed_smiles, small_model, batch_size=1, device='cpu')
    other, _ = serve_salsa(mixed_smiles, small_model, batch_size=batch_size,
                           device='cpu')
    assert_embeddings_agree(ref, other)


def test_serve_salsa_preserves_input_order(small_model):
    """Row i of the output corresponds to row i of the input."""
    embeddings, returned = serve_salsa(ORDER_SMILES, small_model, batch_size=2,
                                       device='cpu')

    assert returned == ORDER_SMILES
    assert embeddings.shape[0] == len(ORDER_SMILES)

    for i, smi in enumerate(ORDER_SMILES):
        one, _ = serve_salsa([smi], small_model, batch_size=1, device='cpu')
        assert_embeddings_agree(embeddings[i], one[0])


def test_order_check_would_catch_a_swapped_row(small_model):
    """Distinct molecules sit far enough apart for the order check to bite."""
    embeddings, _ = serve_salsa(ORDER_SMILES, small_model, batch_size=2,
                                device='cpu')
    for i in range(len(ORDER_SMILES)):
        for j in range(i + 1, len(ORDER_SMILES)):
            assert not np.allclose(embeddings[i], embeddings[j],
                                   rtol=0, atol=EMB_ATOL)


def count_canonicalizations(monkeypatch):
    """Patch get_cansmiles everywhere it is used and return a call log."""
    from salsa_lit.utils import chemistry

    calls = []
    real = chemistry.get_cansmiles

    def counting(smiles, *args, **kwargs):
        calls.append(smiles)
        return real(smiles, *args, **kwargs)

    monkeypatch.setattr('salsa_lit.api.inference.get_cansmiles', counting)
    monkeypatch.setattr('salsa_lit.data.datasets.get_cansmiles', counting)
    return calls


def test_filtering_canonicalizes_each_input_once(small_model, short_smiles,
                                                 monkeypatch):
    """With filter_input, SMILES are canonicalized once rather than twice."""
    calls = count_canonicalizations(monkeypatch)
    serve_salsa(short_smiles, small_model, device='cpu', filter_input=True)
    assert len(calls) == len(short_smiles)


def test_without_filtering_canonicalizes_each_input_once(small_model, short_smiles,
                                                         monkeypatch):
    """Without filter_input, the dataset still canonicalizes, exactly once."""
    calls = count_canonicalizations(monkeypatch)
    serve_salsa(short_smiles, small_model, device='cpu', filter_input=False)
    assert len(calls) == len(short_smiles)


def test_filtering_still_canonicalizes(small_model):
    """Skipping the dataset's pass does not skip canonicalization itself."""
    _, returned = serve_salsa(['OCC'], small_model, device='cpu', filter_input=True)
    assert returned == ['CCO']


@pytest.mark.parametrize('filter_input', [True, False])
def test_invalid_smiles_never_reach_the_tokenizer(small_model, filter_input):
    """Unparseable input is dropped in both filter modes."""
    smiles = ['CCO', 'not_a_molecule', 'CCCO', '[[[']
    embeddings, returned = serve_salsa(smiles, small_model, device='cpu',
                                       filter_input=filter_input)

    assert returned == ['CCO', 'CCCO']
    assert embeddings.shape[0] == 2


def test_filtering_reports_unparseable_input(small_model, capsys):
    """The drop message counts unparseable SMILES, not just long ones."""
    serve_salsa(['CCO', 'not_a_molecule'], small_model, device='cpu',
                filter_input=True)
    assert 'unparseable' in capsys.readouterr().out


def test_filtering_drops_multifragment_and_overlong(small_model):
    """Multi-fragment and over-length SMILES are still filtered out."""
    smiles = ['CCO', 'CCO.CCO', 'C' * 300]
    _, returned = serve_salsa(smiles, small_model, device='cpu', filter_input=True)
    assert returned == ['CCO']


def test_trim_agrees_for_the_trained_model(trained_model_path, mixed_smiles,
                                           monkeypatch):
    """Trimming holds for the real model, not just a small random one."""
    model = load_model(trained_model_path, map_location='cpu')
    trimmed, untrimmed = embed_both_ways(model, mixed_smiles, 4, monkeypatch)
    assert_embeddings_agree(trimmed, untrimmed)
