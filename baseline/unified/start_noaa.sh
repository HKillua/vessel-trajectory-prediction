#!/bin/bash
# Wrapper to run NOAANY baselines
source /home/wangguangjie/anaconda3/etc/profile.d/conda.sh
conda activate ship-traj-pred
cd /home/wangguangjie/djs/vessel-trajectory-prediction/baseline/unified
bash run_noaa_baselines.sh pred10
