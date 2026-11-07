#!/bin/bash
#SBATCH --job-name=jacobi_mpi_benchmark
#SBATCH --output=output_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --time=00:20:00
#SBATCH --mem=150G
#SBATCH --partition=teachingnodes
#SBATCH --account=2026COMP3270

module load openmpi

N=100000
MARGIN=1.0
RESULTS=jacobi_mpi_benchmark_results0.txt
echo "procs runtime" > $RESULTS

for p in 1 2 4 8 16 24 32 48 64
do
    echo "Running jacobi_mpi with $p processes"
    OUT=$(mpirun -n $p ./jacobi_mpi $N $MARGIN)
    echo "$OUT"
    RUNTIME=$(echo "$OUT" | grep "Total run time" | awk '{print $4}')
    echo "$p $RUNTIME" >> $RESULTS
done
