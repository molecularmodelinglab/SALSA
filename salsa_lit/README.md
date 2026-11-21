# salsa_lit

Python API and command line reference. See the [main README](../README.md) for
installation and a quick start.

## Pretrained models

| name | dim | training steps |
|---|---|---|
| `salsa8` | 8 | 2,300,000 |
| `salsa16` | 16 | 3,400,000 |
| `salsa24` | 24 | 1,600,000 |
| `salsa32` | 32 | 1,200,000 |

`salsa16` is the default. Weights live in the
[molecularmodelinglab/SALSA](https://huggingface.co/molecularmodelinglab/SALSA)
repository, pinned to a fixed revision so results stay reproducible, and are
downloaded into the Hugging Face cache on first use. Set `HF_HOME` or pass
`cache_dir` to put them elsewhere.

```python
from salsa_lit import download_pretrained, list_pretrained, load_model

list_pretrained()                  # name -> filename and dimension
download_pretrained('salsa32')     # pre-fetch, returns the cached path
model = load_model('salsa32')      # downloads if needed
model = load_model()               # the default model
model = load_model('run/')         # a local directory or checkpoint
```

Anywhere a model is accepted, an existing path takes precedence over a
registry name.

## Python API

### Training

```python
from salsa_lit import cook_salsa, ModelConfig, TrainingConfig

model, trainer = cook_salsa("training_data.parquet", save_dir="run/")
```

```python
model_config = ModelConfig(
    dim_salsa=64,      # Embedding dimension
    dim_emb=512,       # Transformer hidden dim
    n_heads=8,         # Attention heads
    L_enc=8,           # Encoder layers
    L_dec=8,           # Decoder layers
)

training_config = TrainingConfig(
    n_epochs=30,
    batch_size=50,     # Families per batch
    lr=0.00001,
    devices=4,         # DDP across 4 GPUs
    accelerator='gpu',
    precision='bf16-mixed',
)

model, trainer = cook_salsa(
    "training_data.parquet",
    save_dir="run/",
    model_config=model_config,
    training_config=training_config,
    val_csv="validation_data.parquet",
)
```

Multi-GPU training runs through Lightning's DDP; set `devices` to the number of
GPUs and leave the rest alone.

### Generating embeddings

```python
from salsa_lit import serve_salsa, load_model

embeddings, smiles = serve_salsa(["CCO", "CC(=O)O", "c1ccccc1"])

embeddings, smiles = serve_salsa(
    "molecules.parquet",
    model="salsa32",
    output_file="embeddings.parquet",
    batch_size=50,
    filter_input=False,  # keep input the model cannot embed, and error on it
)
```

`serve_salsa` returns embeddings in input order, one row per molecule, and
writes `SMILES` plus one `dim_*` column per dimension. A model may be a path, a
`SalsaNet`, or a `SalsaLightningModule`; `load_model` loads one explicitly.

### Creating training data

```python
from salsa_lit import mutate_dataset, DataConfig

df, path = mutate_dataset(
    "source_molecules.parquet",
    output_file="training_data.parquet",
    data_config=DataConfig(n_muts=5, mahala_thresh=6.0, weight='chembl'),
)
```

### Creating evaluation data

```python
from salsa_lit import supermutate_dataset, SuperMutantConfig

df, path = supermutate_dataset(
    "anchors.parquet",
    output_file="supermutants.parquet",
    config=SuperMutantConfig(depth=5, n_branches=1),
    n_jobs=8,
)
```

Supermutants are chains of successive single edits, so the compound at depth
*n* is *n* graph edits from its anchor.

## Configuration

### ModelConfig
- `dim_salsa`: SALSA embedding dimension (default: 16)
- `dim_emb`: Transformer hidden dimension (default: 512)
- `n_heads`: Number of attention heads (default: 8)
- `L_enc`: Encoder layers (default: 8)
- `L_dec`: Decoder layers (default: 8)
- `dim_ff`: Feed-forward dimension (default: 2048)
- `dropout`: Dropout rate (default: 0.1)
- `pool_func`: 'avg', 'max' or 'set' (default: 'avg')
- `proj_head`: Use a projection head (default: False)

### TrainingConfig
- `n_epochs`: Number of training epochs (default: 1)
- `batch_size`: Families per batch (default: 50)
- `lr`: Learning rate (default: 0.00001)
- `losses`: Loss functions to use (default: ['Recon', 'SupCon'])
- `save_step`: Checkpoint every N steps (default: 10000)
- `devices`: Number of GPUs (default: 1)
- `accelerator`: 'auto', 'gpu', 'cpu' (default: 'auto')
- `precision`: '32', '16-mixed', 'bf16-mixed' (default: 'bf16-mixed')
- `num_workers`: Dataloader workers (default: 4)

### DataConfig
- `n_muts`: Mutations per anchor (default: 5)
- `mahala_thresh`: Mahalanobis distance threshold (default: 6.0)
- `weight`: 'chembl' or 'uniform' (default: 'chembl')
- `actions`: Mutation types (default: ['Add', 'Replace', 'Remove'])

### SuperMutantConfig
- `depth`: Maximum edit depth per chain (default: 5)
- `n_branches`: Chains grown per anchor (default: 1)
- `mahala_thresh`: Mahalanobis distance threshold (default: 6.0)
- `weight`: 'chembl' or 'uniform' (default: 'chembl')
- `actions`: Mutation types (default: ['Add', 'Replace', 'Remove'])

## Command Line Interface

```bash
# Pretrained models
salsa-lit download --list
salsa-lit download salsa32
salsa-lit download --all

# Create training data
salsa-lit mutate source_molecules.parquet -o data/ --n-muts 5

# Create evaluation data
salsa-lit supermutate anchors.parquet -o data/ --depth 5 --n-jobs 8

# Train a model
salsa-lit train training_data.parquet -o run/ --epochs 30 --devices 4

# Generate embeddings, with a pretrained model or your own
salsa-lit embed molecules.parquet -o embeddings.parquet
salsa-lit embed molecules.parquet -m salsa32 -o embeddings.parquet
salsa-lit embed molecules.parquet -m run/ -o embeddings.parquet
```

Every command accepts `--format {parquet,csv}`, defaulting to parquet; an
explicit `.csv` output path takes precedence over the flag. Run
`salsa-lit <command> --help` for the full option list.
