# RPent（清华/无问芯穹）复现进展速报 —— 2026-09-23 晨

> 可直接转发。来源：微信文章（2026-09）→ github.com/RLinf/RPent，论文 arXiv:2607.08448。
> 我们的工作：在**自有昇腾算力**上端到端跑通官方 LIBERO-PRO 基准，大脑用 **GLM-5.3** 替代官方的 GPT-6/Claude。

## 一、结论先行

1. **端到端复现成功**。RPent 全栈（LLM planner + SAM3 视觉分割 + 冻结 π0.5 VLA + MuJoCo 仿真 + 记忆语料）在**纯国产算力**（昇腾 910B3，无 NVIDIA GPU）上完整跑通，首个正式 episode 即解出 LIBERO-PRO 任务。
2. **批量评估收官：20/20 全部解出 = 100%**（libero_object_swap × 10 任务 × 2 种子，LIBERO 官方终止谓词判定，2026-09-23 07:31 完成）。落在官方 GPT-6 Astra 档（99%），高于 GPT-5.5（91%）/Qwen3.6-27B（84%），冻结 π0.5 单独跑仅 17%。
3. RPent 的核心主张——**"记忆引导的 agent 循环把冻结 VLA 从 11% 拉到 92.63%"**——在我们环境上以约 6 倍差距成立（17% → 100%）。

## 二、官方数字 vs 我们的配置

| 方法 | LIBERO-PRO Overall | Object Swap 套件 |
|---|---|---|
| 冻结 π0.5 单独跑（无 agent） | 11.0% | 17% |
| RPent + Qwen3.6-27B 大脑 | 70.63% | 84% |
| RPent + GPT-5.5 大脑 | 72.1% | 91% |
| RPent + GPT-6 Astra（官方最优） | **92.63%** | 99% |
| **RPent + GLM-5.3 大脑（我们，昇腾）** | 未测 | **100%（20/20）** |

硬件：aura-7（8×910B3，64G HBM/卡）+ 192 核鲲鹏 CPU 跑 MuJoCo（osmesa 软渲染）。π0.5 前向 ~8 秒/chunk（含编译，之后更快）、SAM3 分割 ~1 秒/次，均在 NPU 上。

## 三、仿真视频（已解出任务的回放）

- `first_solved_t2_s0.mp4`（20 秒，首跑：抓沙拉酱瓶放入筐）
- `eval_t0_s0.mp4` / `eval_t0_s1.mp4`（批量前两集）
（位置：`rpent-repro/videos/`，每集官方自动录制 `episode.mp4`）

## 四、性能数据（前 10 集，9 胜 1 负）

**GLM-5.3 大脑（api planner）**：

| 指标 | 值 |
|---|---|
| 单请求上下文长度 | 平均 **75k tokens**（44k-96k；失败集重试撑到 151k） |
| Prompt cache 命中率 | **90.3%**（GLM Anthropic 兼容口的 caching 有效） |
| 每集新增 prefill（未命中 token） | ~169k |
| 每集 decode 输出 | ~23.6k tokens（成功集 13-36k） |
| 每集 LLM 请求数 | 平均 32 |
| 平均 turn 周期（LLM+工具混合） | ~30 秒 |

**执行层（NPU 实测）**：

| 指标 | 值 |
|---|---|
| π0.5 前向 | 首次 7.9s（含算子编译），稳态 ~5.6s/action-chunk（pi0_pick 整技能 67s/12 chunks） |
| SAM3 分割 | 1-2s/次（NPU 热缓存） |
| 单集总时长 | 成功集均值 **716s**（最快 469s） |

## 五、主要技术坑（供复用）

**NPU 移植三大死锁**（py-spy 定位，均为运行时补丁、不改模型权重）：
1. SAM3 静默挂死：官方代码用 CUDA 专用融合核 `aten._addmm_activation`，昇腾不支持 → 换通用 `F.linear` 路径后 1 秒/次。
2. π0.5 首推理挂死：CANN 算子编译器（TBE）在多线程进程里 fork 死锁 → `TE_PARALLEL_COMPILER=1` 串行编译解决，kernel 落盘缓存后复用。
3. 长跑稳定性：VLA/SAM3 改外部常驻服务（HTTP RPC），跨 episode 复用，避免偶发再挂 + 省每集 169 秒模型加载。

**网络**（华为云 HK 出口对主流 CDN 几乎全断）：HF 的 Xet CDN、PyPI 的 CloudFront、GitHub Release 都不通或每秒 0 字节 → 7.5G 权重走"Mac 经 VPN 下载 + 28 分块×6 路并行 scp"（单流 38KB/s，并行 1.2MB/s）；pip 换阿里源、GitHub 走 gh-proxy。

**GLM-5.3 当 planner 大脑**：Anthropic 兼容口直插（官方本来就支持 `anthropic:` 前缀），视觉输入/tool calling/prompt caching 全部工作。有意思的行为：它会主动把并行分割请求改串行、VLA 超时后自己降级策略、闭环伺服时自报"距离 0.012，在容差内"。

## 六、后续可选

- 扩展第二个套件（libero_goal / libero_10 系列）验证泛化
- 跑 RPent Flash Mode（记忆回放免 LLM 规划，官方 72.63%）
- 完整报告：`RPent复现报告.md`（环境/调试全记录 + 最终成绩表 + 性能数据）；20 集全部回放视频：`videos/eval20/`
