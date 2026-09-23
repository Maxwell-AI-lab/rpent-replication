#!/usr/bin/env bash
set -u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TE_PARALLEL_COMPILER=1
export PI05_CHECKPOINT_PATH=/data/rpent/checkpoints/pi05-libero-130
export SAM3_CHECKPOINT_PATH=/data/rpent/checkpoints/sam3.pt
export LIBERO_TYPE=pro
export LIBERO_PRO_ASSET_PATH=/data/rpent/liberopro-assets
export ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
export ANTHROPIC_API_KEY="${GLM_API_KEY:?请设置 GLM_API_KEY 环境变量}"
cd /data/rpent/rpent
OUT=/data/rpent/logs/eval_object_swap
mkdir -p $OUT
for t in 0 1 2 3 4 5 6 7 8 9; do
  for s in 0 1; do
    D=$OUT/t${t}_s${s}
    [ -f $D/.done ] && { echo "skip t$t s$s"; continue; }
    echo "=== [task $t seed $s] start $(date +%H:%M:%S) ==="
    rpent --robot libero --suite libero_object_swap --task $t --seed $s \
      --planner api --model anthropic:GLM-5.3 \
      --memory-profile local --memory-dir /data/rpent/rpent/memory/libero \
      --vla-endpoint http://127.0.0.1:18803 \
      --sam3-endpoint http://127.0.0.1:18802 \
      --max-turns 100 --planner-timeout-s 5400 \
      --output-dir $D > $D.out 2>&1
    echo "exit=$? $(date +%H:%M:%S)"
    touch $D/.done
  done
done
echo "== BATCH DONE =="
