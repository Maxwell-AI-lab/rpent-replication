#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export TE_PARALLEL_COMPILER=1
export MAX_COMPILE_CORE_NUMBER=8
export ASCEND_SLOG_PRINT_TO_STDOUT=0
cd /data/rpent/rpent
export PI05_CHECKPOINT_PATH=/data/rpent/checkpoints/pi05-libero-130
exec python3 rpent/robots/components/pi05_vla_server.py --embodiment libero --transport http --host 127.0.0.1 --port 18803
