#!/bin/bash


#SBATCH --partition=dgx
#SBATCH --account=undergrad_research
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --output=./best_channels/slurm-%j.out


####
#
# Here's the actual job code.
# Note: You need to make sure that you execute this from the directory that
# your python file is located in OR provide an absolute path.
#
####

# Path to container
container="/data/containers/msoe-tensorflow-22.06-tf2-py3.sif"

# Command to run inside container
command="python get_best_channels.py"

# Execute singularity container on node.
singularity exec --nv -B /data:/data ${container} ${command}