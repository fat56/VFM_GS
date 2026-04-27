#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
PYTHON_BIN="${PYTHON:-python3}"

CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/bicycle -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --grad_abs_thresh 0.0012
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=flowers ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/flowers -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --dense 0.005 --grad_abs_thresh 0.0015
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=garden ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/garden -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=stump ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/stump -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --dense 0.004 --grad_abs_thresh 0.0015
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=treehill ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/treehill -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --dense 0.01 --grad_abs_thresh 0.002
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=room ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/room -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0008
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=counter ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/counter -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0008 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/kitchen -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0006
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bonsai ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/mipnerf360/bonsai -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --grad_abs_thresh 0.0006
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=truck ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/tanksandtemples/truck --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.04 --grad_abs_thresh 0.0009 --mult 0.7 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=train ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/tanksandtemples/train --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.042 --grad_abs_thresh 0.0015 --dense 0.01 --mult 0.7 
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=playroom ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/db/playroom --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0015 --dense 0.003 --mult 0.7
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=drjohnson ${PYTHON_BIN} -m vfm_gs.cli.train --variant fastgs_baseline -s ./datasets/db/drjohnson --eval --densification_interval 500  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.0025 --grad_abs_thresh 0.0012 --dense 0.013 --mult 0.7 

CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/bicycle --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/flowers --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/garden --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/stump --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/treehill --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/room --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/counter --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/kitchen --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/bonsai --skip_train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/truck --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/train --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/playroom --skip_train --mult 0.7
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.render -m output/drjohnson --skip_train --mult 0.7

CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/bicycle
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/flowers
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/garden
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/stump
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/treehill
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/room
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/counter
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/kitchen
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/bonsai
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/truck
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/train
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/playroom
CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} -m vfm_gs.cli.metrics -m output/drjohnson
