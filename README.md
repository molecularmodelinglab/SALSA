[SALSA: Semantically-Aware Latent Space Autoencoder](https://doi.org/10.1609/aaai.v38i12.29221) (AAAI 2024)

## Installation

Requires Python 3.11 or newer.

```bash
pip install git+https://github.com/molecularmodelinglab/SALSA.git
```

To work on the code instead:

```bash
git clone https://github.com/molecularmodelinglab/SALSA.git
cd SALSA
pip install -e ".[dev]"
```

Optional extras: `viz` for plotting and UMAP, `dev` for tests and linters.

## Quick Start

Installing provides the `salsa-lit` command:

```bash
salsa-lit mutate molecules.parquet -o mutants/
salsa-lit train mutants/<generated>.parquet -o run/
salsa-lit embed molecules.parquet -m run/ -o embeddings.parquet
```

The same three steps from Python:

```python
from salsa_lit import mutate_dataset, cook_salsa, serve_salsa

df, train_path = mutate_dataset("molecules.parquet",
                                output_file="mutants.parquet")

model, trainer = cook_salsa(train_path, save_dir="run/")

embeddings, smiles = serve_salsa("molecules.parquet", model="run/",
                                 output_file="embeddings.parquet")
```

Datasets are read and written as parquet by default; pass a `.csv` path or
`--format csv` to use CSV instead.

The Python API and command line reference are in
[salsa_lit/README.md](salsa_lit/README.md).

## Citation

Kirchoff, K. E., Maxfield, T., Tropsha, A., & Gomez, S. M. (2024). SALSA:
Semantically-Aware Latent Space Autoencoder. Proceedings of the AAAI Conference
on Artificial Intelligence, 38(12), 13211–13219.
https://doi.org/10.1609/aaai.v38i12.29221
