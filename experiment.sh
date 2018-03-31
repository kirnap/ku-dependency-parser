#! /bin/bash
#SBATCH -J A.R.O.G.2
#SBATCH --mem=16384
#SBATCH --nodelist=cn4
#SBATCH -p ai_gpu
#SBATCH --time 90:00:00
#SBATCH --gres=gpu:1

export JULIA_PKGDIR=/home/okirnap/.julia
source /KUFS/scratch/okirnap/deep_deneme/ku-dependency-parser/getready.sh
julia --depwarn no main_v6.jl --load $chm --datafiles $d1 $d2 --otrain 1 --btrain 0  > newbtrain16_test4.txt
