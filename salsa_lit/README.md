# SALSA-Lit: Structurally-Aware Latent Space Autoencoder

A refactored, modular implementation of SALSA with PyTorch Lightning support for multi-GPU training.

## Installation

### From source (recommended)

```bash
# Clone the repository
git clone https://github.com/molecularmodelinglab/SALSA.git
cd SALSA

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Training a Model

```python
from salsa_lit import cook_salsa, ModelConfig, TrainingConfig

# Train with default settings
model, trainer = cook_salsa("training_data.csv", save_dir="./checkpoints/model_default")

# Train with custom config
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
    devices=4,         # Multi-GPU support!
    accelerator='gpu',
)

model, trainer = cook_salsa(
    "training_data.csv",
    save_dir="./checkpoints/model_custom",
    model_config=model_config,
    training_config=training_config,
)
```

### Generating Embeddings

```python
from salsa_lit import serve_salsa

# From SMILES list
embeddings, smiles = serve_salsa(
    ["CCO", "CC(=O)O", "c1ccccc1"],
    model="./checkpoints/model_default/",
)

# From CSV file
embeddings, smiles = serve_salsa(
    "molecules.csv",
    model="./checkpoints/model_default/",
    output_file="embeddings.csv",
)
```

### Creating Training Data (Mutation)

```python
from salsa_lit import mutate_dataset, DataConfig

data_config = DataConfig(
    n_muts=5,           # Mutations per anchor
    mahala_thresh=6.0,  # Mahalanobis threshold
    weight='chembl',    # Weighting scheme
)

df, path = mutate_dataset(
    "source_molecules.csv",
    output_file="training_data.csv",
    data_config=data_config,
)
```

## Command Line Interface

After installation, the `salsa-lit` command is available:

```bash
# Train a model
salsa-lit train training_data.csv -o ./checkpoints/model_default --epochs 30 --devices 4

# Generate embeddings
salsa-lit embed molecules.csv -m ./checkpoints/model_default -o embeddings.csv

# Create training data (mutation)
salsa-lit mutate source_molecules.csv -o ./data --n-muts 5
```

## Package Structure

```
salsa_lit/
├── __init__.py           # Main exports
├── constants.py          # Static values (vocab, tokens, atom counts)
├── config/               # Configuration dataclasses
│   ├── model_config.py   # ModelConfig (architecture)
│   ├── training_config.py # TrainingConfig (training hyperparams)
│   └── data_config.py    # DataConfig (mutation settings)
├── utils/                # Utility modules
│   ├── io.py             # File I/O utilities
│   ├── chemistry.py      # Chemistry/RDKit utilities
│   └── fingerprints.py   # Fingerprint generation
├── data/                 # Data handling
│   ├── tokenizer.py      # SMILESTokenizer
│   ├── datasets.py       # PyTorch Datasets
│   ├── samplers.py       # Family-based batch sampling
│   └── mutation.py       # Molecule mutation
├── models/               # Neural network components
│   ├── components.py     # Transformer components (PMA, MAB, etc.)
│   ├── losses.py         # SupConLoss, reconstruction loss
│   └── salsa_net.py      # Main SalsaNet model
├── lightning/            # Lightning integration
│   ├── salsa_module.py   # LightningModule wrapper
│   └── data_module.py    # DataModule for data loading
├── api/                  # High-level API
│   ├── train.py          # cook_salsa()
│   ├── inference.py      # serve_salsa(), load_model()
│   └── mutate.py         # mutate_dataset()
└── cli/                  # Command-line interface
    └── main.py           # CLI entry point
```

## Multi-GPU Training

SALSA-Lit uses PyTorch Lightning for seamless multi-GPU training:

```python
# Automatic multi-GPU with DDP
training_config = TrainingConfig(
    devices=4,              # Number of GPUs
    accelerator='gpu',
    precision='16-mixed',   # Mixed precision for speed
)
```

## Configuration

### ModelConfig
- `dim_salsa`: SALSA embedding dimension (default: 32)
- `dim_emb`: Transformer hidden dimension (default: 512)
- `n_heads`: Number of attention heads (default: 8)
- `L_enc`: Encoder layers (default: 8)
- `L_dec`: Decoder layers (default: 8)
- `dim_ff`: Feed-forward dimension (default: 2048)
- `dropout`: Dropout rate (default: 0.1)

### TrainingConfig
- `n_epochs`: Number of training epochs (default: 1)
- `batch_size`: Families per batch (default: 50)
- `lr`: Learning rate (default: 0.00001)
- `losses`: Loss functions to use (default: ['Recon', 'SupCon'])
- `devices`: Number of GPUs (default: 1)
- `accelerator`: 'auto', 'gpu', 'cpu' (default: 'auto')
- `precision`: '32', '16-mixed', 'bf16-mixed' (default: '32')

### DataConfig
- `n_muts`: Mutations per anchor (default: 5)
- `mahala_thresh`: Mahalanobis distance threshold (default: 6.0)
- `weight`: 'chembl' or 'uniform' (default: 'chembl')
- `actions`: Mutation types (default: ['Add', 'Remove', 'Replace'])

## Backward Compatibility

The original `salsa/` package remains unchanged. Models trained with the old code can be loaded:

```python
from salsa_lit import SalsaNet

# Load old model weights
model = SalsaNet.from_pretrained("old_model_dir/")
```

## Requirements

Core dependencies (installed automatically with `pip install -e .`):

- Python
- PyTorch
- Lightning
- RDKit
- pandas
- numpy
- scipy
- tqdm
- PyYAML

## License

MIT
