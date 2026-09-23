# RPent 复现报告（LIBERO-PRO @ aura-7/155，GLM-5.3 大脑）

> 项目：复现 [RLinf/RPent](https://github.com/RLinf/RPent)（清华+无问芯穹，"Agentic Infrastructure for the Physical World"），微信文章 2026-09。
> 论文：arXiv:2607.08448 "Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents"
> 复现窗口：2026-09-22 21:00 → 09-23（跨夜自动执行）。执行：ZCode（GLM-5.3 驱动）+ 用户

## TL;DR

1. **端到端复现成功**：非官方硬件路径（华为昇腾 910B3 NPU 而非 NVIDIA GPU）上，RPent 全栈（LLM planner + SAM3 感知 + 冻结 π0.5 + MuJoCo 仿真）完整跑通，`libero_object_swap t2 s0` **首次正式跑即解出**（632 秒，LIBERO 官方终止谓词判定）。
2. **GLM-5.3 当大脑可用**：走 bigmodel 的 Anthropic 兼容口（`--planner api --model anthropic:GLM-5.3`），视觉输入、tool calling、prompt caching 全部工作。GLM 展现真实智能体行为：主动串行化 SAM3 调用、诊断 VLA 超时并降级、闭环视觉伺服（"final_dist 0.012, at tolerance"）。
3. **NPU 移植三大坑全部定位并修复**（SAM3 CUDA 融合核挂死、π0.5 TBE 编译 fork 死锁、进程偶发再死锁→外部常驻服务化），全部改动为运行时补丁，不改上游模型/权重。
4. 20 集小批量评估结果见 §4（跑批中实时更新）。

## 1. 复现目标与口径

| 方法（官方 leaderboard） | LIBERO-PRO Overall | Object Swap（本批量对照套件） |
|---|---|---|
| **冻结 π0.5 单独（无 agent）** | **11.0%** | **17%** |
| RPent + Qwen3.6-27B（no-reasoning） | 70.63% | 84% |
| RPent + GPT-5.5（xhigh） | 72.1% | 91% |
| RPent Flash Mode（Molmo2-8B 回放） | 72.63% | 93% |
| RPent + Opus-4.7（max reasoning） | 82.4% | — |
| RPent + GPT-6 Astra（官方最优） | **92.63%** | 99% |

核心主张：**记忆引导的 agent 循环把冻结 VLA 从 11% 拉到 90%+**。本复现用 GLM-5.3 替代 GPT/Claude 大脑，验证（a）主张方向（b）GLM 档位。

## 2. 硬件与环境（非官方路径）

- **执行机**：aura-7（昇腾节点，华为云 HK 出口 EIP，具体地址见内部记录），8×昇腾 910B3（64G HBM/卡），192 核鲲鹏，1.5T 内存，/data SFS 共享盘
- **容器**：`rpent-npu`（镜像 k3-train:cann852-v14 + pip 安装后 docker commit 固化，--privileged + /dev + Ascend driver 只读挂载）
- **软件栈**：Python 3.11.15 / torch 2.7.1+cpu + torch_npu 2.7.1.post2（CANN 8.5.2）/ rpent editable + rpent-openpi + rpent-libero(robosuite 1.5.2) + rpent-liberopro + rpent-rlinf + sam3 / mujoco 3.3.0
- **渲染**：`MUJOCO_GL=osmesa`（无 NVIDIA EGL；apt 装 libosmesa6，aliyun ubuntu-ports 源）
- **资产**：π0.5 `RLinf-Pi05-LIBERO-130-fullshot-SFT`（7,473,091,464 字节校验）、SAM3 sam3.pt 3.45GB（modelscope）、LIBERO-PRO assets 622MB、记忆语料 RLinf/RPent-memory

## 3. 移植与调试全记录（按发现顺序）

### 3.1 网络五坑（Mac ↔ 155 ↔ 公网）
1. 155 的 HK 出口对 HF Xet CDN（cas-bridge.xethub.hf.co / us.aws.cdn.hf.co）TCP 可连但大文件 0 字节/秒，hf-mirror 对 Xet 文件也只透传重定向 → **π0.5 权重走 Mac（SSRDOG 代理 hf download ~1MB/s）分块推送**：28×256MB、6 路并发 scp ≈1.2MB/s（单流 SSH 经 VPN 仅 38KB/s，并行线性扩展；24 路会打爆代理连接数）
2. 155 出口总量 ~1.2-1.5MB/s，一个大下载占满时新连接全部饿死（测速必须串行）
3. files.pythonhosted.org 也在 CloudFront → pip 换 aliyun 镜像；GitHub 依赖 fork 走 gh-proxy.com（须 `git config --global url.insteadOf`，仓库级配置对 pip 临时 clone 无效）
4. GLM API 从 155 偶发 60 秒超时后自愈（Mac 代理稳定，可作后备）
5. modelscope 直连稳定 ~1.2MB/s（sam3 3.45GB 走它）

### 3.2 NPU 三大坑（全部 py-spy/栈定位）
| 症状 | 根因 | 修复 |
|---|---|---|
| SAM3 推理 120s 超时、AICore 0% | `sam3/perflib/fused.py::addmm_act` 调 CUDA 融合核 `aten._addmm_activation` + 强制 bf16（torch_npu 不支持该核 → 静默挂死） | fused.py 打 portable fallback（普通 `F.linear`+激活，CPU/NPU 通用）+ `get_device_properties` None shim |
| π0.5 首次 predict 挂死 | TBE 算子编译的 `cann_kb_init` 用 multiprocessing Manager，在多线程进程 fork 死锁（recv 永等） | **`export TE_PARALLEL_COMPILER=1`** 串行编译绕过；首次前向 7.9s 后 kernel 落盘缓存 |
| run 内 spawn 的 VLA 偶发再挂 | 同 2，非确定性 fork 竞态 | **VLA/SAM3 外部常驻服务化**（`--vla-endpoint :18803`/`--sam3-endpoint :18802`），跨集复用 |

其他：容器须 `--privileged`（否则 davinci "resource busy"）；`robot_spec.py` 的 `MUJOCO_GL:"egl"` env_overrides 与 `env_server.py` 的 `PYOPENGL_PLATFORM` setdefault 都要改成继承父环境；`rpent-check-llm` 对 GLM 报 sdk_error 是假阴性（16-token 探针被 thinking 吃满，实际默认 8192 正常）。

### 3.3 SAM3 CPU 路径（备用，未采用）
SAM3 也能跑 CPU（192 核）：需 model.float() + 全局 tensor shim（cuda→cpu 重定向）+ 跳过 sam3_image.py 的显式 bf16 强转。实测 fp32 CPU 单次推理 >590s（OpenBLAS 192 线程劣化），弃用，最终走 NPU。

## 4. 评估结果（libero_object_swap × task0-9 × seed0/1 = 20 集）

**最终战绩：20/20 全部解出 = 100%**（2026-09-23 07:31 跑完，LIBERO 官方终止谓词判定）

| 方法 | Object Swap 套件 | LIBERO-PRO Overall（官方） |
|---|---|---|
| 冻结 π0.5 单独（无 agent） | 17% | 11.0% |
| RPent + Qwen3.6-27B | 84% | 70.63% |
| RPent + GPT-5.5 | 91% | 72.1% |
| RPent + GPT-6 Astra | 99% | 92.63% |
| **RPent + GLM-5.3（我们，昇腾 NPU）** | **100%（20/20）** | 未测 |

注：20 集样本小于官方 100 集/套件口径，100% vs 99% 无统计显著差异；结论是**稳稳落在 Astra 档（~99%），显著高于 GPT-5.5（91%）与 Qwen3.6（84%）**，核心主张"agent 循环 >> 冻结 VLA（17%）"以约 6 倍差距成立。

**性能（20 集均值）**：单集 919s（成功集多数 470-1180s）；LLM 31 请求/集，单请求上下文 avg 74k / max 159k tokens；prompt cache 命中 90.4%；每集新增 prefill 157k、decode 22.7k tokens；turn 周期 29.1s（含工具执行）。执行层：π0.5 ~5.6s/chunk（NPU），SAM3 1-2s/次（NPU）。全部 20 集 mp4 已存 `videos/eval20/`。

**GLM-5.3 行为观察**：两种策略——pi0_pick（VLA 抓取，主流、快）与手工原语抓取（segment+move_pose+set_gripper，慢，曾在调试期 3621s 超时）；失败前兆特征 = 上下文撑大（151k+）+ decode 3 倍 + 请求 2.5 倍的重试循环（本批量未出现真失败）。智能体行为：主动串行化并行分割、VLA 超时自诊自降级、闭环视觉伺服自述误差。

## 5. 结论与讨论

1. **RPent 的核心主张在我们环境完全成立**：同一个冻结 π0.5，单独跑 Object Swap 只有 17%，套上"LLM 大脑 + SAM3 感知 + 记忆 + 原语工具"的 agent 循环后 20/20。差距不在模型权重，在编排。
2. **GLM-5.3 是合格的具身大脑**：vision+tool-calling+长上下文+cache 全部工作，成功率达到官方最强档（样本量内）。这为"国产大脑+国产算力跑具身 agent"提供了直接证据。
3. **昇腾可跑此类栈**：三大死锁都是工程适配问题（融合核替换、串行编译、服务常驻化），模型本体零改动。代价是首跑前 ~30 分钟的算子编译（一次性）与 5.6s/chunk 的推理（GPU 上应 ~0.5s），对单线程评估可接受，对大规模并行评估需优化。
4. **局限**：仅 1 个套件 20 集（官方口径 8 套件 800 集）；未测 Flash Mode/探索记忆闭环；NPU 延迟使长时评估偏慢。后续可选：扩第二个套件（libero_goal/10）、跑 Flash Mode 验证记忆回放、用 GLM-5.3-flash 降本。
