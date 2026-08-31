# IsaacLab（火山引擎版）Release Note · v1

本版本是 Isaac Lab 的火山引擎发行版，基于上游 `isaac-sim/IsaacLab` 的 v2.3.2 稳定线（上游最新的正式版本），镜像内使用 Isaac Sim 5.1，对应分支 `release_v1`。上游的仿真环境、内置任务与 PPO 训练能力原样保留，可直接使用；以下是在此基础上补充的部分。

## 一、SO-ARM101 机械臂支持

上游未收录该机型。本版本集成开源扩展
[`isaac_so_arm101`](https://github.com/MuammerBay/isaac_so_arm101)（BSD-3），它提供 reach 与
lift 两个任务；在其之上，本版本补充了面向 sim2real 的域随机化配置。

## 二、off-policy 算法（SAC）

上游的内置任务只随附 PPO 训练配置，使用其他算法需要自行集成。SAC 属于 off-policy 算法，采集到的数据可反复利用，样本效率更高。本版本集成的是 ETH 的 SAC 实现（[算法](https://github.com/leggedrobotics/rsl_rl_sac)、
[集成与 9 款足式/人形机器人的配置](https://github.com/sabagian/isaaclab-sac)、
[论文](https://arxiv.org/abs/2605.24975)，BSD 3-Clause 许可，原实现基于 rsl-rl 4.0.1），并将其移植到 Isaac Lab v2.3.2 所用的 rsl-rl 5.0.1，使 SAC 与 PPO 运行在同一版 rsl-rl 上，随镜像交付，默认超参为论文原版配置。实测该实现在 G1 人形机器人的崎岖地形任务上，最终表现明显优于 PPO（见「实测数据」）。

训练命令加一个参数即可从 PPO 切到 SAC：`--agent rsl_rl_sac_cfg_entry_point`。

## 三、训练画面实时回传

上游的画面串流依赖显卡内置的视频编码硬件（NVENC），而 A100、A30、H20 等主流训练卡并不具备该硬件。本版本采用一条不依赖该硬件的推流链路，任意训练卡都可以在浏览器中实时查看训练画面：训练命令加 `--stream` 即可，移动类与机械臂类任务均支持。相机自动跟随机器人群，不需要设置机位；多个训练同时进行时各占独立房间，互不覆盖。观看页受用户名密码保护。

## 四、训练产物

每次训练自动生成一个目录：

```
logs/rsl_rl/<任务名>/<启动时刻>/
├── model_50.pt  model_100.pt  ...     每 50 轮一个 checkpoint
├── params/env.yaml  params/agent.yaml  本次实际生效的完整配置
└── events.out.tfevents.*               TensorBoard 曲线
```

以下三项按需执行，各对应一条命令（见 `EXAMPLE_COMMANDS.md`）：

- **训练诊断报告** —— 汇总本次实际生效的全部超参，以及平均回合奖励、平均回合长度、每步奖励的全程走势与末段值、UTD（每采集一条数据平均做多少次梯度更新）等训练指标；
- **评估视频** —— 加载指定 checkpoint（PPO 与 SAC 均可）查看策略的实际表现，同时导出可部署模型（`policy.onnx` / `policy.pt`）；
- **上传 TOS** —— 将训练目录按原结构上传，走内网端点。

## 实测数据

### SAC / PPO 对照

A100 单卡、seed 42、各算法推荐配置（PPO 4096 env / SAC 8192 env）：

| 任务 | PPO | SAC |
|---|---|---|
| G1 平地 | +27.83 / 26 min（1000 轮） | +29.77 / 49 min（800 轮） |
| G1 崎岖 | +11.54 / 47 min（1500 轮） | **+21.07 / 55 min（800 轮）** |

**崎岖地形上 SAC 明显占优**：G1 崎岖地形上，SAC 训练 28 分钟即达到 PPO 47 分钟的最终水平，继续训练最终 reward 接近 PPO 的两倍；平地等简单任务则是 PPO 更快，质量相当。同轮数下 SAC 的墙钟约为 PPO 的 1.9 倍，优势在样本效率与最终表现，不在单轮速度。

两点实测均与论文一致：论文报告 SAC 在人形机器人任务上超过 PPO（"on humanoid tasks，
SAC surpasses PPO"），也指出其墙钟开销显著高于 PPO 且在结构上难以消除。

### 并行环境数

| 场景 | 推荐 `--num_envs` | 依据 |
|---|---|---|
| 移动类 PPO | 4096 | 上游移动类任务的默认值 |
| 移动类 SAC | 8192（配 800 轮） | ETH 论文所用配置，实测结果与论文相当 |
| 机械臂操作 PPO | 4096 | 上游与扩展操作类任务的默认值 |
| 推流确认场景 | 16 | 推流相机按环境创建，不能配大并行度 |

并行度需要调得更高时，注意：

- **SAC 的显存由 replay buffer 决定**（默认 500 万条，全额预分配），环境数的影响反而小。显存不足时优先下调 `agent.algorithm.replay_buffer_size`，而不是环境数；
- **机械臂操作任务受物理引擎的材质数量上限约束**，这个上限与显存无关，换更大的卡不会提高；
- **限制并行度的可能是机器内存而非显存**：选卡时应将机器内存一并纳入评估，尤其是图形类机型，其内存配比通常远低于训练机型。

## 注意事项

- **SAC 仅支持移动类任务**：操作类只有 PPO 配置，上游与论文均未提供可参照的操作类 SAC 超参。
- **SAC 不支持断点续训**：replay buffer 不落盘，恢复即冷启动退化。
- **`--stream` 不配大并行度**：推流相机按并行环境创建，数量等于 `--num_envs`，用于小规模确认场景，正式训练不加。
- **Sim 工作站必须用带 NVENC 的显卡**：L4 / L20 / RTX 4090 可以，A100 / A30 / H20 不行。无编码器的卡上服务能起、连接能建，但画面永远不来，且无有效报错。
- **工作站同时只接一个客户端**：Isaac Sim 自身限制，多人同时使用需部署独立实例。

## 参考

- 命令速查见仓库根目录的 `EXAMPLE_COMMANDS.md`。
- 本版本包含上游 IsaacLab v2.3.2 的全部功能，上游完整变更请参考其官方 release notes。上游 3.0 系列目前最新为 `v3.0.0-beta2.patch1`，尚无正式版本，本版本暂不跟进。
- 性能数字来自特定硬件与单 seed 的实测，环境数、训练预算与地形难度都会影响结果，请以自身场景的实测为准。
- 第三方组件与许可：[`rsl_rl_sac`](https://github.com/leggedrobotics/rsl_rl_sac)
  （ETH Zurich / NVIDIA，BSD 3-Clause）、
  [`isaac_so_arm101`](https://github.com/MuammerBay/isaac_so_arm101)
  （Muammer Bay、Louis Le Lay，BSD 3-Clause）。二者的许可全文随镜像分发，分别位于 `rsl_rl_port/LICENSE` 与 `soarm101/LICENSE`。
