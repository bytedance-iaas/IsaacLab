# Isaac Lab 昇腾训练支持 v2.0 发布说明

**发布日期：2026-10-27**  ·  分支 `newton-v2`  ·  基线 Isaac Lab v3.0.0-EA

v2.0 让 Isaac Lab 的强化学习训练可以用昇腾 NPU 做策略学习。训练拆成物理仿真和策略学习两部分，两者经 gRPC
连接，可以分开部署：仿真可以放在 NVIDIA 显卡上，也可以用 Newton 的 CPU 后端放在昇腾服务器自己的 CPU 上。
算法支持 PPO 和 SAC，共用一个训练入口，产出的检查点与上游 rsl-rl 格式相同。已验证任务为 Unitree G1
人形机器人速度跟踪（平地 `Isaac-Velocity-Flat-G1`、粗糙地形 `Isaac-Velocity-Rough-G1`），物理后端为 Newton / MJWarp。

## 主要功能

- 纯 NPU 训练：一台昇腾服务器即可完成训练，不需要 NVIDIA 硬件。
  - 物理仿真拆成多个 Newton CPU 进程（默认 96 个）并绑核，铺满整机 CPU；策略学习用一张 NPU。
- 异构训练：仿真留在 NVIDIA 显卡上，策略学习放到另一台机器的 NPU 上，两台机器可以不在同一机房；学习端不需要装 Isaac Sim 或 NVIDIA 驱动。
- 纯 NVIDIA 训练：照常在一张 NVIDIA 显卡上同时跑仿真和学习。
- SAC 算法：与 PPO 共用训练入口，`--agent` 一个参数切换，三种部署形态都可用；粗糙地形上 SAC 的 reward 约为 PPO 的两倍。

## 完整流程：起仿真 → 训练 → 取检查点

```
  ① 起仿真(serve_shards.sh / serve_persist.sh)
        │  纯 NPU: 本机起多个 CPU 仿真进程      异构: N 卡机器起一个常驻仿真服务
        │  就绪后输出仿真地址
        ▼
  ② 训练(train_remote.py --target <仿真地址> --device npu:0)
        │  每步: 学习端发动作 ──gRPC──▶ 仿真端推进一步, 回传观测 / 奖励
        │  每次迭代: 学习端用采到的数据更新策略(PPO 或 SAC)
        ▼
  ③ 产物: logs/rsl_rl/<实验名>/<时间戳>_<run_name>/
           检查点 model_<迭代>.pt + TensorBoard 日志
```

1. **起仿真**：纯 NPU 用 `serve_shards.sh` 在本机起多个 CPU 仿真进程；异构用 `serve_persist.sh` 在 NVIDIA 机器上起一个常驻服务，学习端断开后不退出，下次训练直接重连。
2. **训练**：`train_remote.py` 连上仿真地址，把远端仿真当作本地环境使用，训练循环与上游 rsl-rl 相同。
3. **取检查点**：检查点按 rsl-rl 的格式定期保存在日志目录里，PPO 可以用 `--resume` 从检查点续训。纯 NVIDIA 形态不需要第 1 步，直接用上游的训练脚本。

## 三种部署形态

```
① 纯 NVIDIA              ② 异构                              ③ 纯 NPU
┌────────────┐          ┌──────────┐       ┌──────────┐     ┌────────────────┐
│ 一张 N 卡  │          │ NPU      │ gRPC  │ N 卡     │     │ 一台昇腾服务器 │
│ 物理仿真   │          │ 策略学习 │◀─────▶│ 物理仿真 │     │ CPU 多进程仿真 │
│ 策略学习   │          └──────────┘       └──────────┘     │ NPU 策略学习   │
└────────────┘                  可跨机房                     └────────────────┘
```

| | ① 纯 NVIDIA | ② 异构 | ③ 纯 NPU |
|---|---|---|---|
| 硬件 | 一张 NVIDIA 显卡 | NVIDIA 显卡（仿真）+ 昇腾 NPU（学习） | 一台昇腾服务器 |
| 学习端需装 Isaac Sim | 是 | 否 | 否 |
| 主要开销 | — | 每步一次网络往返 | CPU 物理仿真 |
| 适用场景 | 有 N 卡，要最快出结果 | 已有 N 卡，要把学习卸到 NPU | 只有昇腾机器 |

## 训练效果与耗时

### 平地 G1、4096 个环境

| 算法 | 部署形态 | 墙钟 | 末 50 次迭代平均 reward |
|---|---|---|---|
| PPO（1500 次迭代） | 纯 NVIDIA（4090） | 18.8 min | 26.96 |
| | 异构（NPU 学习 + 4090 仿真，经公网网关） | 222 min | 27.65 |
| | 纯 NPU（昇腾 950PR） | 47.6 min | 27.91 |
| SAC（800 次迭代） | 纯 NVIDIA（4090） | 27.9 min | 27.79 |
| | 异构（NPU 学习 + 4090 仿真，经公网网关） | 118.1 min | 24.71 |
| | 纯 NPU（昇腾 950PR） | 42.5 min | 26.69 |

### 同一台机器上 PPO 与 SAC

PPO 用 4096 个环境、1500 次迭代；SAC 用 8192 个环境、800 次迭代。

| 任务 | 机器 | PPO | SAC |
|---|---|---|---|
| 粗糙地形 | 4090 | 24.4 min，reward 15.33 | 36.9 min，reward **28.45** |
| 粗糙地形 | 昇腾 950PR | 89.1 min，reward 11.33 | 117.7 min，reward **22.23** |
| 平地 | 4090 | 18.8 min，reward 26.96 | 31.1 min，reward 27.35 |
| 平地 | 昇腾 950PR | 47.6 min，reward 27.91 | 65.3 min，reward 27.41 |

- **粗糙地形用 SAC**：reward 约为 PPO 的两倍，墙钟多 30%～50%
- **平地用 PPO**：两者 reward 接近，PPO 更快

## 纯 NPU 训练

- **多进程仿真**：环境平均分给多个 Newton CPU 仿真进程；Warp 的 CPU 后端单进程只用得上一个核，拆开才能铺满整机
- **绑核**：每个仿真进程固定在一个物理核及其超线程上，学习进程单独占几个核，互不抢占
- **异步采集**：各仿真进程独立推进，不逐步同步，只在每次迭代结束时汇合
- **CPU 图回放**：一个控制步约 1630 次计算启动合并为 12 次回放；同一组动作下的仿真结果与不回放时一致

## 异构训练

- **环境即服务**：仿真端把 Isaac Lab 环境包成 gRPC 服务，学习端把它当成普通的本地环境使用
- **常驻服务**：仿真服务在学习端断开后不退出，免去每次几分钟的启动
- **安全的数据格式**：张量用 safetensors 传输，不用 pickle，对端数据不会被当作代码执行；每次调用带令牌校验
- PPO 与 SAC 共用同一个仿真服务

## SAC

- 基于 rsl-rl-lib 5.4.1 的补丁（`sac_port/apply.sh`），加入 SAC 算法、actor/critic 模型、经验回放缓冲区和 `OffPolicyRunner`
- G1 平地与粗糙地形自带 SAC 任务配置（`rsl_rl_sac_cfg_entry_point`）
- 三种部署形态都可用

## 命令行

`remote_env/serve_shards.sh` 纯 NPU 起 CPU 仿真进程 · `remote_env/serve_persist.sh` 异构起常驻仿真服务 ·
`remote_env/train_remote.py` 连接仿真地址训练（`--agent` 选 PPO / SAC，`--device` 选 NPU / GPU，`--resume` 续训）。
所有仿真端与学习端之间的调用都要求同一个 `ENV_TOKEN`。

## 已知限制

- 异构形态每步一次网络往返，延迟直接累加到训练时间；经公网网关时，PPO 1500 次迭代的 222 分钟里约 192 分钟花在网络上
- 异构形态当前为明文传输（h2c，无 TLS），只适合受信网络
- 一个仿真服务同时只服务一个学习端
- SAC 不能从检查点续训（经验回放缓冲区不落盘），中断后只能从头训练；粗糙地形 SAC 在纯 NPU 上出现过一次训练发散，同配置重跑未复现
- Newton 与 PhysX 两个物理后端不等价，在一个后端上训出的策略不能直接用于另一个
- 已验证任务仅为 G1 速度跟踪（平地与粗糙地形）

## 快速开始

纯 NPU，平地 G1，PPO：

```bash
# 1. 起仿真: 96 个 CPU 仿真进程并绑核, 就绪后输出所有进程的地址
export ENV_TOKEN=<令牌>
T=$(SHARD_CPUS=8-31,40-127 TASK=Isaac-Velocity-Flat-G1 \
    bash remote_env/serve_shards.sh 96 4096 19000)

# 2. 训练: 在 NPU 上训练(换成 SAC: --agent rsl_rl_sac_cfg_entry_point)
ASCEND_RT_VISIBLE_DEVICES=0 taskset -c 0-7,128-135 \
python remote_env/train_remote.py --task Isaac-Velocity-Flat-G1 --target $T \
  --device npu:0 --agent rsl_rl_cfg_entry_point --iters 1500

# 3. 取检查点: logs/rsl_rl/<实验名>/<时间戳>_remote/model_<迭代>.pt
```

异构与纯 NVIDIA 的用法、全部参数、调优建议见 `USER_GUIDE_v2.md`。
