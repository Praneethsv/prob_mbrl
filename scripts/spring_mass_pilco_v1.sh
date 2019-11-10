#!/bin/bash

python examples/deep_pilco_mm.py \
--env=Cartpole \
--verbose=20 \
--port=8080 \
--include_current_state=true \
--include_current_excitations=true \
--wait_action=0.1 \
--incremental_actions=True \
--reset_step=100 \
--goal_reward=5 \
--goal_threshold=1.0 \
--save_interval=10 \
--eval_interval=10 \
--control_H=4 \
--w_r=1 \
--w_u=1 \
--w_d=0 \
--test=false \
--gui=false \


# --load_path=/home/amirabdi/artisynth-rl/results/JawEnv-v1/jaw-sac-pytorch-v1-3/trained/saved \
# --env=Point2PointEnv-v1 \