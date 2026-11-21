"""Tests for carrying identifiers from embedding input to output."""

import pandas as pd
import pytest

from salsa_lit.api.inference import serve_salsa, trim_collate
from salsa_lit.data.datasets import ID_COLUMNS, InferenceDataset
from salsa_lit.utils.io import dataset_columns, read_ds, write_ds


def test_dataset_generates_ids_from_row_position(tokenizer):
    """Without an ID column, IDs are the input row positions."""
    df = pd.DataFrame({'SMILES': ['CCO', 'CCCO', 'CCCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    assert [ds[i]['id'] for i in range(len(ds))] == [0, 1, 2]


def test_generated_ids_skip_dropped_rows(tokenizer):
    """Generated IDs stay the input position when rows are dropped."""
    df = pd.DataFrame({'SMILES': ['CCO', 'not_a_molecule', 'CCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    assert [ds[i]['id'] for i in range(len(ds))] == [0, 2]


def test_generated_ids_survive_filtering(tokenizer):
    """With filter_=True the generated ID is still the input position."""
    df = pd.DataFrame({'SMILES': ['CCO.CCO', 'CCO', 'CCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer, filter_=True)
    assert ds.dataframe['ID'].tolist() == [1, 2]


@pytest.mark.parametrize('column', ID_COLUMNS)
def test_supplied_ids_are_used(tokenizer, column):
    """ID, Id and id are all recognised."""
    df = pd.DataFrame({column: ['a', 'b'], 'SMILES': ['CCO', 'CCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    assert [ds[i]['id'] for i in range(len(ds))] == ['a', 'b']


def test_supplied_ids_pass_through_verbatim(tokenizer):
    """Duplicates and odd values are carried as given, not corrected."""
    df = pd.DataFrame({'ID': ['x', 'x', ''], 'SMILES': ['CCO', 'CCCO', 'CCCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    assert [ds[i]['id'] for i in range(len(ds))] == ['x', 'x', '']


def test_integer_ids_stay_integers(tokenizer):
    """An integer ID column is not widened to float."""
    df = pd.DataFrame({'ID': [10, 20], 'SMILES': ['CCO', 'CCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    assert [ds[i]['id'] for i in range(len(ds))] == [10, 20]
    assert all(isinstance(ds[i]['id'], int) for i in range(len(ds)))


def test_id_column_takes_precedence_over_aliases(tokenizer):
    """When both ID and id exist, ID wins and id is left alone."""
    df = pd.DataFrame({'ID': ['right'], 'id': ['wrong'], 'SMILES': ['CCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    assert ds[0]['id'] == 'right'


def test_collate_keeps_ids_as_a_list(tokenizer):
    """Identifiers are not tensorised by the collate."""
    df = pd.DataFrame({'ID': [7, 8], 'SMILES': ['CCO', 'CCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    batch = trim_collate([ds[0], ds[1]])
    assert batch['id'] == [7, 8]


def test_collate_without_ids_still_works(tokenizer):
    """A batch lacking 'id' collates as before."""
    df = pd.DataFrame({'SMILES': ['CCO', 'CCCO']})
    ds = InferenceDataset(df, tokenizer=tokenizer)
    items = [{k: v for k, v in ds[i].items() if k != 'id'} for i in range(2)]
    assert 'id' not in trim_collate(items)


@pytest.mark.parametrize('ext', ['.parquet', '.csv'])
def test_output_carries_supplied_ids(small_model, tmp_path, ext):
    """Written embeddings keep the input IDs, in input order."""
    src = tmp_path / f'input{ext}'
    write_ds(pd.DataFrame({'ID': ['c1', 'c2', 'c3'],
                           'SMILES': ['CCO', 'CCCO', 'c1ccccc1']}), str(src))
    out = tmp_path / f'out{ext}'

    serve_salsa(str(src), small_model, output_file=str(out), device='cpu')

    written = read_ds(str(out))
    assert written['ID'].tolist() == ['c1', 'c2', 'c3']
    assert list(written.columns)[:2] == ['ID', 'SMILES']


@pytest.mark.parametrize('ext', ['.parquet', '.csv'])
def test_output_carries_generated_ids(small_model, tmp_path, ext):
    """Without input IDs, the output is keyed by row position."""
    src = tmp_path / f'input{ext}'
    write_ds(pd.DataFrame({'SMILES': ['CCO', 'CCCO', 'c1ccccc1']}), str(src))
    out = tmp_path / f'out{ext}'

    serve_salsa(str(src), small_model, output_file=str(out), device='cpu')

    assert read_ds(str(out))['ID'].tolist() == [0, 1, 2]


@pytest.mark.parametrize('filter_input', [True, False])
def test_output_ids_identify_rows_after_drops(small_model, tmp_path, filter_input):
    """Dropped molecules leave a gap in the IDs rather than shifting them."""
    src = tmp_path / 'input.parquet'
    write_ds(pd.DataFrame({'SMILES': ['CCO', 'not_a_molecule', 'CCCO']}), str(src))
    out = tmp_path / 'out.parquet'

    serve_salsa(str(src), small_model, output_file=str(out), device='cpu',
                filter_input=filter_input)

    assert read_ds(str(out))['ID'].tolist() == [0, 2]


def test_serve_salsa_does_not_mutate_the_input_frame(small_model):
    """Stamping generated IDs leaves the caller's frame untouched."""
    df = pd.DataFrame({'SMILES': ['CCO', 'CCCO']})
    serve_salsa(df, small_model, device='cpu')
    assert list(df.columns) == ['SMILES']


def test_output_ids_align_with_returned_embeddings(small_model, tmp_path):
    """Row i of the file is row i of the returned array."""
    src = tmp_path / 'input.parquet'
    write_ds(pd.DataFrame({'ID': [5, 6, 7],
                           'SMILES': ['CCO', 'CCCO', 'c1ccccc1']}), str(src))
    out = tmp_path / 'out.parquet'

    embeddings, smiles = serve_salsa(str(src), small_model,
                                     output_file=str(out), device='cpu')

    written = read_ds(str(out))
    assert written['ID'].tolist() == [5, 6, 7]
    assert written['SMILES'].tolist() == smiles
    dims = [c for c in written.columns if c.startswith('dim_')]
    assert written[dims].to_numpy().shape == embeddings.shape


def test_smiles_list_input_gets_generated_ids(small_model, tmp_path):
    """A plain list of SMILES is keyed by position."""
    out = tmp_path / 'out.parquet'
    serve_salsa(['CCO', 'CCCO'], small_model, output_file=str(out), device='cpu')
    assert read_ds(str(out))['ID'].tolist() == [0, 1]


def test_dataframe_input_keeps_its_ids(small_model, tmp_path):
    """An in-memory frame's ID column is carried through."""
    out = tmp_path / 'out.parquet'
    df = pd.DataFrame({'ID': ['a', 'b'], 'SMILES': ['CCO', 'CCCO']})
    serve_salsa(df, small_model, output_file=str(out), device='cpu')
    assert read_ds(str(out))['ID'].tolist() == ['a', 'b']


def test_other_input_columns_are_not_carried(small_model, tmp_path):
    """Only ID, SMILES and the dimensions are written."""
    src = tmp_path / 'input.parquet'
    write_ds(pd.DataFrame({'ID': [1], 'SMILES': ['CCO'], 'Assay': [3.5]}), str(src))
    out = tmp_path / 'out.parquet'

    serve_salsa(str(src), small_model, output_file=str(out), device='cpu')

    assert 'Assay' not in read_ds(str(out)).columns


@pytest.mark.parametrize('ext', ['.parquet', '.csv'])
def test_dataset_columns_lists_names_without_rows(tmp_path, ext):
    """dataset_columns reports the schema for both formats."""
    path = tmp_path / f'f{ext}'
    write_ds(pd.DataFrame({'ID': [1], 'SMILES': ['CCO'], 'X': [2]}), str(path))
    assert dataset_columns(str(path)) == ['ID', 'SMILES', 'X']
