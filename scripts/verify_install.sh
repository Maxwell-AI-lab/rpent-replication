#!/usr/bin/env bash
set -x
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/rpent/.venv/bin/activate
cd /data/rpent/rpent
python -c "import rpent; print('rpent ok')"
python -c "import liberopro; print('liberopro ok')"
python -c "import openpi; print('openpi ok')" 2>/dev/null || python -c "import rpent_openpi" 2>/dev/null || echo "openpi import name TBD"
python -c "import torch, torch_npu; print('torch', torch.__version__, 'npu ok', torch.npu.is_available())"
MUJOCO_GL=osmesa python -c "
import mujoco
print('mujoco', mujoco.__version__)
" 
echo "== check-llm (GLM) =="
rpent-check-llm --planner api --model anthropic:GLM-5.3 --timeout-s 60 2>&1 | tail -4
echo "== VERIFY DONE =="
