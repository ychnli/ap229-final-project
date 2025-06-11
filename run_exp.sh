# run a N=100 experiment with 25 trials per noise sample 
# python3 run_langevin_gpu.py --N 100 --M 25 --output_dir "/scratch/users/yucli/ap229_langevin/N100_second_exp"

# run a N=1000 experiment with 5 trials per noise sample
# python3 run_langevin_gpu.py --N 1000 --M 5 --output_dir "/scratch/users/yucli/ap229_langevin/N1000"

# run a N=1000 experiment with 5 trials per noise sample
python3 run_langevin_gpu.py --N 1000 --M 10 --output_dir "/scratch/users/yucli/ap229_langevin/N1000_continued"