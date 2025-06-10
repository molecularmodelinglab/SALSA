#!/bin/bash
losses=array("Recon-SupCon" "Recon" "SupCon")

for $loss in losses:
do
    cmd="$loss"
    #sbatch -N 1 -n 1 --job-name="poop" --mem=1g -t 1-00:00:00 --wrap="$cmd" -o "_${v}_%j.out"
done
