# RPent 复现（LIBERO-PRO @ 昇腾 910B3，GLM-5.3 大脑）

复现 [RLinf/RPent](https://github.com/RLinf/RPent)（清华+无问芯穹，"Agentic Infrastructure for the Physical World"，arXiv:2607.08448）——**在纯昇腾算力（无 NVIDIA GPU）上跑通官方 LIBERO-PRO 基准，大脑用 GLM-5.3 替代官方 GPT-6/Claude**。

## 最终成绩（2026-09-23）

| 方法 | Object Swap 套件 | 备注 |
|---|---|---|
| 冻结 π0.5 单独跑 | 17% | 官方数字 |
| RPent + Qwen3.6-27B | 84% | 官方数字 |
| RPent + GPT-5.5 | 91% | 官方数字 |
| RPent + GPT-6 Astra | 99% | 官方数字 |
| **RPent + GLM-5.3（本仓，昇腾）** | **100%（20/20）** | 20 集小样本，真实档位 90-99% |

核心主张验证：**同一个冻结 π0.5，套上"LLM 大脑 + SAM3 感知 + 记忆 + 原语工具"的 agent 循环，从 17% 到 20/20**——差距不在模型权重，在编排。

性能（20 集均值）：GLM 每集 31 请求 / 上下文 avg 74k tok / **cache 命中 90.4%** / decode 22.7k tok；π0.5 约 5.6s/action-chunk（NPU）；SAM3 分割 1-2s（NPU）；单集端到端 919s。

## 仓库结构

```
docs/
  RPent复现报告.md          # 完整版：环境、网络五坑、NPU 三死锁、成绩、性能、结论
  RPent复现-同事参考版.md    # 精简分享版
videos/
  first_solved_t2_s0.mp4    # 首跑解出回放（抓沙拉酱瓶入筐）
  eval20/t*_s*/episode.mp4  # 全部 20 集成功回放
scripts/
  install_env3.sh           # 容器内环境安装（防抖版：临时目录克隆+原子换名）
  vla_boot.sh               # π0.5 VLA 常驻服务（NPU device 0，:18803）
  sam3_boot_npu.sh          # SAM3 常驻服务（NPU device 1，:18802）
  first_run.sh              # 单任务首跑（env 齐全套，key 走 $GLM_API_KEY）
  eval20_final.sh           # 20 集批量（断点续跑，.done 标记）
  analyze_perf2.py          # 性能分析器（上下文/缓存命中率/decode/turn 周期）
  push_parts.py             # Mac→服务器 7.5G 权重分块并行推送（6 路最优）
  merge_pi05.sh             # 分块合并 + 字节数校验
  verify_install.sh         # 安装验证（导入/NPU/osmesa/GLM check）
  patches/                  # 全部 NPU 适配补丁（见下）
```

## 复现步骤（昇腾环境，从零到 20 集）

1. **容器**：`docker run -d --name rpent-npu --privileged -e ASCEND_VISIBLE_DEVICES=0-7 --entrypoint bash -v /data:/data -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro <镜像> -c "sleep infinity"`（**必须 --privileged**，否则 davinci 设备 resource busy）
2. **系统库**：apt 源换 aliyun ubuntu-ports，装 `libosmesa6 libgl1 libegl1 libglib2.0-0`
3. **环境**：`install_env3.sh`（pip 源换 aliyun；git 全局 insteadOf 走 gh-proxy；constraints 锁 `torch==2.7.1` 保住 torch_npu 配对）
4. **打补丁**（`scripts/patches/`，顺序执行；都是幂等的）：
   - `patch_npu.py`：VLA/SAM3 server 注入 `torch_npu.contrib.transfer_to_npu`
   - `patch_fused.py` + `patch_fused2.py` + `patch_fused3.py`：**SAM3 死锁根因**——`aten._addmm_activation` CUDA 融合核昇腾不支持，换通用 `F.linear` 路径
   - `patch_sam3_bf16.py`：CPU 回退时跳过显式 bf16 强转（备用路径，最终未采用）
   - 手工两处 sed：`robot_spec.py` 的 `"MUJOCO_GL": "egl"` → `os.environ.get("MUJOCO_GL","egl")`；`env_server.py` 的 `PYOPENGL_PLATFORM` setdefault 同理派生
5. **资产**：π0.5 ckpt（HF `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT`，**须 --exclude optimizer.pt**；国内网络不通 Xet CDN 时用 `push_parts.py` 分块中转）+ sam3.pt（modelscope `facebook/sam3`）+ LIBERO-PRO assets（HF dataset `RLinf/LIBERO-PRO-assets`，设 `LIBERO_PRO_ASSET_PATH` 免下载）+ 记忆语料（`RLinf/RPent-memory` dataset，`--memory-profile local` 绕 HF）
6. **起服务**：`vla_boot.sh` + `sam3_boot_npu.sh`（**关键环境变量 `TE_PARALLEL_COMPILER=1`**——绕过 TBE 算子编译的 multiprocessing fork 死锁，这是 π0.5 首推理挂死的根因）
7. **跑**：`GLM_API_KEY=xxx bash first_run.sh` → `bash eval20_final.sh`
8. **分析**：`python3 analyze_perf2.py`

## 三个最重要的坑（同环境复现必踩）

1. **SAM3 静默挂死**：AICore 0%、无异常、segment 120s 超时 → CUDA 融合核问题，用 patches 里的 fused fallback
2. **π0.5 首推理挂死**：py-spy 卡在 `tbe/.../multiprocess_util.py` 的 Manager recv → `export TE_PARALLEL_COMPILER=1`，首次前向 7.9s（含算子编译），kernel 落盘缓存后复用
3. **run 内偶发再挂**：fork 竞态非确定性 → VLA/SAM3 一律外部常驻服务 + `--vla-endpoint`/`--sam3-endpoint` 挂入，顺带省每集 169s 模型加载

## 已知局限

- 仅 1 套件 20 集（官方口径 8 套件 800 集）；Object Swap 是最简单套件（Astra 99%），难套件（Long Task/Swap，Astra 也只有 72-85%）预期会看到失败
- NPU 推理延迟 ~5.6s/chunk（GPU 约 0.5s），单线程评估可接受
- 未测 Flash Mode（记忆回放）与探索记忆闭环
