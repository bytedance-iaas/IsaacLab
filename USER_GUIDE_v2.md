# Isaac Lab 昇腾训练支持 v2.0 使用文档

在昇腾 NPU 上训练 Isaac Lab 强化学习策略。基线为 Isaac Lab v3.0.0-EA，代码在 `newton-v2` 分支。

- 版本与变更：[RELEASE_NOTE_v2.md](RELEASE_NOTE_v2.md)
- 三种部署形态：纯 NVIDIA、异构（NPU 学习 + NVIDIA 仿真）、纯 NPU
- 两种算法：PPO 与 SAC，训练入口相同，`--agent` 切换

## 选哪种形态

| | ① 纯 NVIDIA | ② 异构 | ③ 纯 NPU |
|---|---|---|---|
| 硬件 | 一张 NVIDIA 显卡 | NVIDIA 显卡（仿真）+ 昇腾 NPU（学习） | 一台昇腾服务器 |
| 平地 PPO 1500 次迭代 | 18.8 min | 222 min（经公网网关） | 47.6 min |
| 瓶颈 | — | 每步一次网络往返 | CPU 物理仿真 |
| 什么时候用 | 有 N 卡，要最快出结果 | 已有 N 卡，要把学习卸到 NPU | 只有昇腾机器 |

只有昇腾机器时用 ③。② 的速度取决于两端之间的网络延迟，同机房内网会比上表快得多。

## 安装

### 学习端（昇腾机器，② 和 ③ 都需要）

先装好与 CANN 版本匹配的 `torch` 和 `torch_npu`，再在仓库根目录：

```bash
pip install -c npu/constraints.txt --extra-index-url https://download.pytorch.org/whl/cpu \
  -e "source/isaaclab_newton[all]" -e source/isaaclab -e source/isaaclab_assets -e source/isaaclab_tasks \
  -e "source/isaaclab_rl[rsl-rl]" -e source/isaaclab_contrib -e source/isaaclab_experimental \
  -e source/isaaclab_visualizers
pip install rsl-rl-lib==5.4.1 && ./sac_port/apply.sh      # 用 SAC 时需要
```

学习端不需要 Isaac Sim、Kit 或 NVIDIA 驱动。纯 NPU 形态的仿真进程也在这台机器上，用的是同一套安装。

检查：

```bash
python -c "import warp; warp.init(); print(warp.get_devices())"          # 应包含 cpu
python -c "import torch, torch_npu; print(torch.npu.is_available())"     # 应为 True
```

### 仿真端（NVIDIA 机器，① 和 ② 需要）

按上游 Isaac Lab v3.0.0-EA 的方式安装，再装 SAC 补丁（同上最后一行）。

## 纯 NPU（③）

### 跑起来

```bash
export ENV_TOKEN=<自定令牌>

# 1. 起仿真进程。参数：进程数、总环境数、起始端口。全部就绪后输出地址列表
T=$(SHARD_CPUS=8-31,40-127 TASK=Isaac-Velocity-Flat-G1 \
    bash remote_env/serve_shards.sh 96 4096 19000)

# 2. 训练
ASCEND_RT_VISIBLE_DEVICES=0 taskset -c 0-7,128-135 env OMP_NUM_THREADS=8 \
python remote_env/train_remote.py --task Isaac-Velocity-Flat-G1 --target $T \
  --device npu:0 --seed 42 --agent rsl_rl_cfg_entry_point --iters 1500
```

SAC：`--agent rsl_rl_sac_cfg_entry_point --iters 800`，第 1 步的总环境数改为 8192。
粗糙地形：两处 `Isaac-Velocity-Flat-G1` 都换成 `Isaac-Velocity-Rough-G1`。

仿真进程是常驻的，训练结束后不会退出，同一组进程可以接着跑下一次训练（任务和环境数要相同）。不再使用时自行结束这些进程。

### 工作方式

```
┌──────── 昇腾服务器 ─────────────────────────────────────────────┐
│                                                                 │
│  学习进程 train_remote.py      绑核 0-7,128-135     用 1 张 NPU │
│    把 96 个仿真进程合成一个 4096 环境的环境对象                 │
│            │  96 条 gRPC 连接（本机回环）                       │
│            ▼                                                    │
│  仿真进程 server.py × 96       每个绑 1 个物理核及其超线程      │
│    每个跑 Newton / MJWarp CPU 后端，约 43 个环境                │
└─────────────────────────────────────────────────────────────────┘
```

- **为什么拆成多个进程**：Warp 的 CPU 后端一次计算只用一个线程，单进程无论多少环境都只能用满一个核。
- **异步采集**：每个仿真进程由一个线程驱动，连续推进一次迭代所需的步数，进程之间不逐步同步，只在迭代结束时汇合。采集期间策略固定不变，所以先后顺序不影响结果。
- **CPU 图回放**：Warp 把一个控制步内的全部计算录下来，之后整段回放。G1 一个控制步的计算启动次数从约 1630 次降到 12 次。同一组动作下，开关图回放的观测校验和到小数点后 6 位一致。

| 图回放的效果 | 逐次启动 | 图回放 | 加速 |
|---|---|---|---|
| 单进程每步，1 个环境 | 31.5 ms | 3.6 ms | 8.7× |
| 单进程每步，43 个环境 | 87.5 ms | 50.8 ms | 1.72× |
| 单进程每步，256 个环境 | 330.5 ms | 293.7 ms | 1.13× |
| 整机每次迭代（96 进程，4096 环境） | 2.91 s | 1.78 s | 1.63× |

每个进程的环境越少，加速越明显：启动开销是固定的，环境越多越被真实计算摊薄。

### 调优

以下数据来自 2 路 AMD EPYC 9A75（128 物理核 / 256 硬件线程，2 个 NUMA 节点）。

**进程数**：固定 4096 个环境、平地 PPO，96 个进程最快：

| 仿真进程数 | 每进程环境数 | 每次迭代 | 全机 CPU 繁忙 |
|---|---|---|---|
| 48 | 85 | 3.18 s | 18% |
| **96** | 43 | **1.86 s** | 31% |
| 120 | 34 | 2.53 s | 39% |
| 224 | 18 | 3.24 s | 27% |

进程再多反而变慢：每次迭代要等最慢的那个进程，进程越多快慢差得越大（120 个进程时最慢比最快高 39%，96 个时只高 12%）。超过物理核数后，两个进程挤一个核，总时间更长。换机器时建议按物理核数的 3/4 左右起扫。

**绑核**：`SHARD_CPUS` 指定仿真进程可用的物理核范围，每个进程固定在其中一个核及其超线程上。同样 96 个进程，绑核比不绑快 11%（每次迭代 3.87 s → 3.46 s，未开图回放时测得）。学习进程用 `taskset` 绑到 `SHARD_CPUS` 以外的核上，两边不抢占。

**CPU 利用率**：最优配置下全机 CPU 繁忙约 31%，靠加进程填不满，瓶颈在每次迭代的汇合等待。

### 参考耗时（纯 NPU）

| 任务 | 算法 | 环境数 / 迭代数 | 墙钟 | 每次迭代 |
|---|---|---|---|---|
| 平地 | PPO | 4096 / 1500 | 47.6 min | 1.91 s（采集 1.80 s，更新 0.06 s） |
| 平地 | SAC | 4096 / 800 | 42.5 min | 3.2 s |
| 平地 | SAC | 8192 / 800 | 65.3 min | 4.9 s |
| 粗糙地形 | PPO | 4096 / 1500 | 89.1 min | 3.6 s |
| 粗糙地形 | SAC | 8192 / 800 | 117.7 min | 8.8 s |

粗糙地形每步多了地形高度扫描和更复杂的接触计算，CPU 上的代价比 GPU 上大。

## 异构（②）

### 跑起来

仿真端（NVIDIA 机器）起一个常驻服务：

```bash
ENV_TOKEN=<自定令牌> ENV_PORT=18990 TASK=Isaac-Velocity-Flat-G1 NUM_ENVS=4096 \
  bash remote_env/serve_persist.sh
```

学习端（昇腾机器）：

```bash
ENV_TOKEN=<同一令牌> python remote_env/train_remote.py \
  --task Isaac-Velocity-Flat-G1 --target <仿真端地址>:18990 \
  --device npu:0 --seed 42 --agent rsl_rl_cfg_entry_point --iters 1500
```

经网关转发时，`--target` 填网关地址，`--authority` 填网关用来匹配路由的域名。

### 工作方式

- 仿真端启动一个完整的 Isaac Lab 训练进程，再把其中的训练循环换成服务循环：不自己训练，只按学习端的请求执行 `reset` / `step`。环境创建与配置解析全部沿用 Isaac Lab 原有流程。
- 学习端拿到的是一个本地环境对象，每个方法内部是一次 gRPC 调用。rsl-rl 的训练循环不改。
- PPO 每次迭代采集 24 步，每步一次往返；网络更新在 NPU 本地完成，不走网络。
- 同一个仿真服务既能服务 PPO，也能服务 SAC。

### 网络是主要开销

平地 PPO 每次迭代采集约 7.5 秒，折合每步约 310 毫秒，其中仿真本身只占约 40 毫秒，其余是网络往返和序列化。所以 ② 的速度几乎由两端之间的往返延迟决定：经公网网关时 1500 次迭代要 222 分钟，同机房内网直连会快得多。

## 纯 NVIDIA（①）

与上游 Isaac Lab 用法相同，SAC 多一个入口：

```bash
python scripts/reinforcement_learning/train.py --rl_library rsl_rl \
  --task=Isaac-Velocity-Rough-G1 --agent rsl_rl_sac_cfg_entry_point \
  --num_envs 8192 --max_iterations 800 physics=newton_mjwarp
```

## 选哪个算法

粗糙地形用 SAC：4090 上 SAC 末 50 次迭代平均 reward 28.45，PPO 15.33；昇腾上 SAC 22.23，PPO 11.33。平地上两者接近，PPO 更快。

## 参数速查

### `remote_env/train_remote.py`（学习端）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--task` | `Isaac-Velocity-Flat-G1` | 任务名，须与仿真端一致 |
| `--agent` | `rsl_rl_cfg_entry_point` | PPO；SAC 用 `rsl_rl_sac_cfg_entry_point` |
| `--target` | 必填 | 仿真端地址 `host:port`，多个用逗号分隔 |
| `--authority` | 无 | 经网关时用于路由匹配的域名 |
| `--async_collect` | `1` | 多个仿真进程时各自推进、不逐步同步 |
| `--iters` | `1500` | 迭代数 |
| `--seed` | `42` | 随机种子 |
| `--device` | `cuda:0` | 学习设备，昇腾上用 `npu:0` |
| `--run_name` | `remote` | 日志目录名后缀 |
| `--resume` | 无 | 从检查点续训，仅 PPO |

令牌通过环境变量 `ENV_TOKEN` 传入。日志与检查点在 `logs/rsl_rl/<实验名>/<时间>_<run_name>/`。

### `remote_env/serve_shards.sh`（纯 NPU 的仿真进程）

用法：`serve_shards.sh <进程数> <总环境数> [起始端口，默认 19000]`，全部就绪后输出 `--target` 用的地址列表。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ENV_TOKEN` | 必填 | 令牌 |
| `TASK` | `Isaac-Velocity-Flat-G1` | 任务名 |
| `SHARD_CPUS` | 不绑核 | 可用物理核范围，如 `8-31,40-127` |
| `SHARD_LOGDIR` | `/tmp/env_shards` | 各进程日志目录 |
| `PHYSICS` | `newton_mjwarp` | 物理后端 |

### `remote_env/serve_persist.sh`（异构的仿真服务）

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ENV_TOKEN` | 必填 | 令牌 |
| `ENV_PORT` | `18990` | 监听端口 |
| `TASK` | `Isaac-Velocity-Flat-G1` | 任务名 |
| `NUM_ENVS` | `4096` | 环境数 |
| `PHYSICS` | `newton_mjwarp` | 物理后端，也可用 `isaacsim_physx` |

## 安全约定

- 令牌只通过环境变量 `ENV_TOKEN` 传入，不写进文件、命令行参数或镜像
- 每次调用都校验令牌，不匹配直接拒绝
- 张量用 safetensors 传输，不用 pickle，对端数据不会被当作代码执行
- 当前为明文传输（h2c，无 TLS），令牌和数据都未加密：② 只在受信网络中使用，测试结束后更换令牌

## 已知限制

- ② 每步一次网络往返，延迟直接累加到训练时间
- ② 当前为明文传输，见「安全约定」
- 一个仿真服务同时只服务一个学习端
- SAC 不能从检查点续训：经验回放缓冲区不落盘，中断后只能从头训练。粗糙地形 SAC 在纯 NPU 上出现过一次训练中途发散（损失变为 NaN），同配置重跑未复现；长时间训练建议留意日志中的损失
- Newton 后端的结果不是逐位可重复的，同一动作序列两次运行之间存在微小差异
- Newton 与 PhysX 不等价：在一个后端上训出的策略放到另一个后端上会摔倒，部署时要用训练时的后端
- 已验证任务仅为 G1 速度跟踪（平地与粗糙地形），其他任务未做系统验证
