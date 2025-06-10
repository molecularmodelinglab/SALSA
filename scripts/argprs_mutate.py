import argparse

def main():
    parser = argparse.ArgumentParser(description='Mutate dataset')
    parser.add_argument('ds', help='Anchor dataset (DataFrame or path to CSV)')
    parser.add_argument('-n', '--n_samp', type=int, default=-1, help='Number of samples to use')
    parser.add_argument('-o', '--out_name', help='Output file name')
    parser.add_argument('-m', '--num_muts', type=int, default=N_MUTS, help='Number of mutations')
    parser.add_argument('-a', '--actions', type=str, default='Add-Replace-Remove', help='Mutation actions')
    parser.add_argument('-t', '--mahala_thresh', type=float, default=MAHALA_THRESH, help='Mahalanobis threshold')
    parser.add_argument('-w', '--weight', type=float, default=WEIGHT, help='Mutation weight')
    parser.add_argument('--use_tqdm', action='store_true', help='Enable tqdm progress bar')
    parser.add_argument('--slurm', action='store_true', help='Execute as a Slurm job')
    
    args = parser.parse_args()
    
    df_out = mutate_data(args.ds, args.n_samp, args.out_name, args.num_muts, args.actions,
                         args.mahala_thresh, args.weight, args.use_tqdm, args.slurm)
    
    # Do something with the output dataframe (df_out)
    print(df_out)
    # Alternatively, you can write it to a file, etc.
    
if __name__ == '__main__':
    main()
