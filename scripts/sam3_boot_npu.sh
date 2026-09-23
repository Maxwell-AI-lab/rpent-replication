#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
cd /data/rpent/rpent
export SAM3_CHECKPOINT_PATH=/data/rpent/checkpoints/sam3.pt
exec python3 rpent/robots/components/sam3_server.py --transport http --host 127.0.0.1 --port 18802
