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

## Usage

Installing provides the `salsa-lit` command:

```bash
salsa-lit mutate molecules.parquet -o mutants/
salsa-lit train mutants/<generated>.parquet -o run/
salsa-lit embed molecules.parquet -m run/ -o embeddings.parquet
```

Run `salsa-lit <command> --help` for options. The Python API is documented in
[salsa_lit/README.md](salsa_lit/README.md).
