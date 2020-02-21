#!/bin/bash
# set -e

module load cuda/10.1
module load cudnn/7.5-cu10.1

singularity run --nv --writable 20191218_singularity_sandbox/
source /opt/ros/kinetic/setup.zsh 
source /venvs/rl_env/bin/activate
source /workspace/devel/setup.zsh

python code/myfinger2.py
