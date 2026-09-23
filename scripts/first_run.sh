#!/usr/bin/env bash
set -x
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa
export TE_PARALLEL_COMPILER=1
export MAX_COMPILE_CORE_NUMBER=8
export PYOPENGL_PLATFORM=osmesa
export PI05_CHECKPOINT_PATH=/data/rpent/checkpoints/pi05-libero-130
export SAM3_CHECKPOINT_PATH=/data/rpent/checkpoints/sam3.pt
export LIBERO_TYPE=pro
export LIBERO_PRO_ASSET_PATH=/data/rpent/liberopro-assets
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_API_KEY="${GLM_API_KEY:?请设置 GLM_API_KEY 环境变量}"
cd /data/rpent/rpent
rpent --robot libero --suite libero_object_swap --task 2 --seed 0   --planner api --model anthropic:GLM-5.3   --memory-profile local --memory-dir /data/rpent/rpent/memory/libero   --vla-endpoint http://127.0.0.1:18803 --sam3-endpoint http://127.0.0.1:18802 --max-turns 100 --planner-timeout-s 5400   --output-dir /data/rpent/logs/first_run
echo "FIRST_RUN_EXIT=$?"
