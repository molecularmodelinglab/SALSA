"""Tests for salsa_lit.data.datasets.InferenceDataset."""

import pandas as pd
import pytest

from salsa_lit.data.datasets import InferenceDataset


def test_canonicalizes_by_default(tokenizer):
    """Non-canonical input is canonicalized when canonicalize is left alone."""
    ds = InferenceDataset(pd.DataFrame({'SMILES': ['OCC']}), tokenizer=tokenizer)
    assert ds.dataframe['SMILES'].tolist() == ['CCO']


def test_canonicalize_false_leaves_smiles_untouched(tokenizer):
    """With canonicalize=False the SMILES are used exactly as supplied."""
    ds = InferenceDataset(pd.DataFrame({'SMILES': ['OCC']}), tokenizer=tokenizer,
                          canonicalize=False)
    assert ds.dataframe['SMILES'].tolist() == ['OCC']


def test_canonicalize_false_skips_rdkit(tokenizer, monkeypatch):
    """canonicalize=False does no RDKit work at all."""
    calls = []
    monkeypatch.setattr('salsa_lit.data.datasets.get_cansmiles',
                        lambda s, *a, **k: calls.append(s) or s)
    InferenceDataset(pd.DataFrame({'SMILES': ['CCO', 'CCCO']}),
                     tokenizer=tokenizer, canonicalize=False)
    assert calls == []


@pytest.mark.parametrize('canonicalize', [True, False])
def test_empty_smiles_are_dropped_either_way(tokenizer, canonicalize):
    """Empty SMILES are removed whether or not this dataset canonicalizes."""
    df = pd.DataFrame({'SMILES': ['CCO', '', 'CCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer, canonicalize=canonicalize)
    assert ds.dataframe['SMILES'].tolist() == ['CCO', 'CCCO']


def test_unparseable_smiles_dropped_when_canonicalizing(tokenizer):
    """Canonicalizing maps unparseable input to '', which is then dropped."""
    df = pd.DataFrame({'SMILES': ['CCO', 'not_a_molecule']})
    ds = InferenceDataset(df, tokenizer=tokenizer, canonicalize=True)
    assert ds.dataframe['SMILES'].tolist() == ['CCO']


def test_origin_id_tracks_input_position(tokenizer):
    """Origin_id keeps the original row position across dropped rows."""
    df = pd.DataFrame({'SMILES': ['CCO', '', 'CCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    assert ds.dataframe['Origin_id'].tolist() == [0, 2]


def test_items_are_tokenized_consistently(tokenizer, short_smiles):
    """Each item's seq matches encoding its SMILES directly."""
    ds = InferenceDataset(pd.DataFrame({'SMILES': short_smiles}), tokenizer=tokenizer)
    for i in range(len(ds)):
        item = ds[i]
        assert item['seq'].equal(tokenizer.encode(item['smiles']))


def test_input_dataframe_is_not_mutated(tokenizer):
    """Constructing a dataset leaves the caller's frame alone."""
    df = pd.DataFrame({'SMILES': ['OCC']})
    InferenceDataset(df, tokenizer=tokenizer)
    assert df['SMILES'].tolist() == ['OCC']
    assert 'Origin_id' not in df.columns
