"""
Command-line interface for SALSA.

Provides commands for:
- train: Train a SALSA model
- embed: Generate embeddings for molecules
- mutate: Create mutated dataset for training
"""

import argparse
import sys


def resolve_output_format(output, fmt_flag):
    """
    Settle the output format from an explicit output path and --format.

    Parameters
    ----------
    output : str or None
        Explicit output path, if one was given
    fmt_flag : str or None
        Value of --format, or None if it was not passed

    Returns
    -------
    fmt : str
        'parquet' or 'csv'
    """
    from salsa_lit.utils.io import DEFAULT_FORMAT, infer_format

    if output:
        by_ext = infer_format(output, default=None)
        if by_ext is not None:
            if fmt_flag is not None and by_ext != fmt_flag:
                raise SystemExit(
                    f"salsa-lit: error: --format {fmt_flag} conflicts with the "
                    f"extension of {output} (reads as {by_ext})"
                )
            return by_ext

    return fmt_flag or DEFAULT_FORMAT


def train_command(args):
    """Train a SALSA model."""
    from salsa_lit.config import ModelConfig, TrainingConfig
    from salsa_lit.api.train import cook_salsa

    # Build configs from args
    model_config = ModelConfig(
        dim_salsa=args.dim_salsa,
        dim_emb=args.dim_emb,
        n_heads=args.n_heads,
        L_enc=args.n_enc_layers,
        L_dec=args.n_dec_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    )

    training_config = TrainingConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        save_step=args.save_step,
        devices=args.devices,
        accelerator=args.accelerator,
        precision=args.precision,
        num_workers=args.num_workers,
    )

    # Train
    model, trainer = cook_salsa(
        train_csv=args.train_data,
        save_dir=args.output_dir,
        val_csv=args.val_data,
        model_config=model_config,
        training_config=training_config,
        n_samp=args.n_samp if args.n_samp > 0 else None,
        resume_ckpt=args.resume,
    )

    print(f"Training complete. Model saved to {args.output_dir}")


def embed_command(args):
    """Generate embeddings for molecules."""
    from salsa_lit.api.inference import serve_salsa

    embeddings, smiles = serve_salsa(
        smiles=args.input,
        model=args.model,
        output_file=args.output,
        batch_size=args.batch_size,
        device=args.device,
        filter_input=args.filter,
        fmt=resolve_output_format(args.output, args.format),
    )

    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    if args.output:
        print(f"Saved to {args.output}")


def mutate_command(args):
    """Create mutated dataset for training."""
    from salsa_lit.config import DataConfig
    from salsa_lit.api.mutate import prepare_training_data

    data_config = DataConfig(
        n_muts=args.n_muts,
        mahala_thresh=args.mahala_thresh,
        weight=args.weight,
    )

    df, output_path = prepare_training_data(
        smiles_file=args.input,
        output_dir=args.output_dir,
        data_config=data_config,
        n_samp=args.n_samp,
        filter_data=not args.no_filter,
        max_len=args.max_len,
        max_atoms=args.max_atoms,
        fmt=resolve_output_format(None, args.format),
    )

    print(f"Created mutated dataset with {len(df)} samples")
    print(f"Saved to {output_path}")


def supermutate_command(args):
    """Create supermutant dataset for evaluation."""
    from salsa_lit.config import SuperMutantConfig
    from salsa_lit.api.supermutate import prepare_supermutant_data

    config = SuperMutantConfig(
        depth=args.depth,
        n_branches=args.n_branches,
        mahala_thresh=args.mahala_thresh,
        weight=args.weight,
    )

    df, output_path = prepare_supermutant_data(
        smiles_file=args.input,
        output_dir=args.output_dir,
        config=config,
        n_samp=args.n_samp,
        filter_data=not args.no_filter,
        max_len=args.max_len,
        max_atoms=args.max_atoms,
        n_jobs=args.n_jobs,
        fmt=resolve_output_format(None, args.format),
    )

    print(f"Created supermutant dataset with {len(df)} compounds")
    print(f"Saved to {output_path}")


def main(argv=None):
    """Main entry point for SALSA CLI."""
    parser = argparse.ArgumentParser(
        prog='salsa-lit',
        description='SALSA: Structurally-Aware Latent Space Autoencoder for molecules'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    ### Train command
    train_parser = subparsers.add_parser('train', help='Train a SALSA model')
    train_parser.add_argument('train_data',
                              help='Path to training parquet/CSV with mutated molecules')
    train_parser.add_argument('-o', '--output-dir', help='Output directory for model')
    train_parser.add_argument('--val-data', help='Path to validation parquet/CSV')
    train_parser.add_argument('--n-samp', type=int, default=-1, help='Number of families to sample')

    # Model architecture
    train_parser.add_argument('--dim-salsa', type=int, default=32, help='SALSA embedding dimension')
    train_parser.add_argument('--dim-emb', type=int, default=512, help='Transformer hidden dimension')
    train_parser.add_argument('--n-heads', type=int, default=8, help='Number of attention heads')
    train_parser.add_argument('--n-enc-layers', type=int, default=8, help='Encoder layers')
    train_parser.add_argument('--n-dec-layers', type=int, default=8, help='Decoder layers')
    train_parser.add_argument('--dim-ff', type=int, default=2048, help='Feed-forward dimension')
    train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training hyperparameters
    train_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=50, help='Batch size (families per batch)')
    train_parser.add_argument('--learning-rate', type=float, default=0.00001, help='Learning rate')
    train_parser.add_argument('--save-step', type=int, default=10000, help='Checkpoint save interval')

    # Hardware
    train_parser.add_argument('--devices', type=int, default=1, help='Number of GPUs')
    train_parser.add_argument('--accelerator', default='auto', help='Accelerator type')
    train_parser.add_argument('--precision', default='bf16-mixed',
                               help="Training precision: 'bf16-mixed' (default), "
                                    "'16-mixed', or '32'")
    train_parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    train_parser.add_argument('--resume', metavar='CKPT_PATH', default=None,
                               help='Path to checkpoint to resume training from (e.g. last.ckpt)')

    ### Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings for molecules')
    embed_parser.add_argument('input', help='Input SMILES (parquet or CSV file)')
    embed_parser.add_argument('-m', '--model', required=True, help='Path to trained model')
    embed_parser.add_argument('-o', '--output', help='Output file for embeddings')
    embed_parser.add_argument('--format', choices=['parquet', 'csv'], default=None,
                              help='Output format (default: parquet). An extension '
                                   'on --output takes precedence.')
    embed_parser.add_argument('--batch-size', type=int, default=50, help='Inference batch size')
    embed_parser.add_argument('--device', help='Device (cpu, cuda, cuda:0, etc.)')
    embed_parser.add_argument('--filter', action='store_true',
                              help='Drop multi-fragment or over-length SMILES instead of '
                                   'erroring on them (costs a canonicalization pass)')

    ### Mutate command
    mutate_parser = subparsers.add_parser('mutate', help='Create mutated dataset for training')
    mutate_parser.add_argument('input', help='Input parquet or CSV with SMILES and ID columns')
    mutate_parser.add_argument('-o', '--output-dir', help='Output directory')
    mutate_parser.add_argument('--format', choices=['parquet', 'csv'], default=None,
                               help='Output format (default: parquet)')
    mutate_parser.add_argument('--n-samp', type=int, default=-1, help='Number of molecules to sample')
    mutate_parser.add_argument('--n-muts', type=int, default=5, help='Mutations per molecule')
    mutate_parser.add_argument('--mahala-thresh', type=float, default=6.0, help='Mahalanobis threshold')
    mutate_parser.add_argument('--weight', choices=['chembl', 'uniform'], default='chembl',
                               help='Mutation weighting scheme')
    mutate_parser.add_argument('--no-filter', action='store_true', help='Skip input filtering')
    mutate_parser.add_argument('--max-len', type=int, default=120, help='Max SMILES length')
    mutate_parser.add_argument('--max-atoms', type=int, default=35, help='Max number of atoms')

    ### Supermutate command
    sup_parser = subparsers.add_parser(
        'supermutate',
        help='Create supermutant dataset (chains of successive edits) for evaluation')
    sup_parser.add_argument('input', help='Input parquet or CSV with SMILES and ID columns')
    sup_parser.add_argument('-o', '--output-dir', help='Output directory')
    sup_parser.add_argument('--format', choices=['parquet', 'csv'], default=None,
                            help='Output format (default: parquet)')
    sup_parser.add_argument('--n-samp', type=int, default=-1, help='Number of anchors to sample')
    sup_parser.add_argument('--depth', type=int, default=5,
                            help='Max GED depth; one compound per depth 1..N')
    sup_parser.add_argument('--n-branches', type=int, default=1,
                            help='Independent chains grown per anchor')
    sup_parser.add_argument('--mahala-thresh', type=float, default=6.0,
                            help='Mahalanobis threshold, applied at every step')
    sup_parser.add_argument('--weight', choices=['chembl', 'uniform'], default='chembl',
                            help='Mutation weighting scheme')
    sup_parser.add_argument('--n-jobs', type=int, default=1, help='Worker processes')
    sup_parser.add_argument('--no-filter', action='store_true', help='Skip input filtering')
    sup_parser.add_argument('--max-len', type=int, default=120, help='Max SMILES length')
    sup_parser.add_argument('--max-atoms', type=int, default=35, help='Max number of atoms')

    ### Parse and dispatch
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'train':
        train_command(args)
    elif args.command == 'embed':
        embed_command(args)
    elif args.command == 'mutate':
        mutate_command(args)
    elif args.command == 'supermutate':
        supermutate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
