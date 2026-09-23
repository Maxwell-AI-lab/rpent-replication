#!/usr/bin/env bash
set -ex
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ROOT=/data/rpent
cd $ROOT
if [ ! -f rpent/pyproject.toml ]; then
  for a in 1 2 3 4 5; do
    rm -rf rpent.tmp
    git clone --depth 1 https://gh-proxy.com/https://github.com/RLinf/RPent.git rpent.tmp && break
    sleep 5
  done
  rm -rf rpent && mv rpent.tmp rpent
fi
cd rpent
git config url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/" || true
printf "torch==2.7.1\n" > /tmp/constraints.txt
pip install -e ".[libero-pro]" -c /tmp/constraints.txt
rc=$?
echo "PIP_RC=$rc"
python3 /data/rpent/patch_npu.py /data/rpent/rpent
echo "== ENV INSTALL3 DONE RC=$rc =="
