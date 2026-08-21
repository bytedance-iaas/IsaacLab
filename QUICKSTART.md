# Quick Start · 从拿到 Pod 到训练产物

面向使用者:假设管理员已经交付给你一个跑着本镜像的 GPU pod(下文以 `isaaclab-a100` 为例)。
所有命令均为单行,可直接复制。

## 本仓库是什么

基于 **isaac-sim/IsaacLab `main` 分支(v2.3.2 稳定线,对应 Isaac Sim 5.1)** 的训练发行版,自研增强:

- **off-policy SAC**(vendored ETH RSL-RL-SAC 5.0.1 port,[论文](https://arxiv.org/abs/2605.24975)),与官方 PPO 并存,一个参数切换
- **配置驱动训练服务**(`training-service/`):机器人/任务/算法/预算写 yaml 提交,免代码
- **训练画面回传**(`--stream`,LiveKit)与诊断报告(`make_report.py`)
- 域随机化(sim2real)、TOS 产物上传、SO-ARM101 集成

> 上游 3.0 系列(develop 分支、Isaac Sim 6.0)尚为 beta,本仓库暂不跟进。

---

# 一、训练

## 1.1 进入 pod 并确认空闲

```bash
kubectl exec -it isaaclab-a100 -- bash
```

```bash
cd /workspace/isaaclab
```

确认没有别人的训练在跑(应输出 0;进程名是 python3,`-f` 不能省):

```bash
pgrep -cf train.py
```

## 1.2 起训练:PPO(默认)

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --num_envs 4096 --max_iterations 1500 --headless > /workspace/train.log 2>&1 < /dev/null &
```

- `setsid` 不能省:断开 kubectl exec 会杀进程组,nohup 挡不住
- 训练必须从 `/workspace/isaaclab` 启动(日志写到"当前目录/logs")

## 1.3 起训练:SAC(本仓库新增)

同一条命令,加 **`--agent rsl_rl_sac_cfg_entry_point`**:

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --agent rsl_rl_sac_cfg_entry_point --num_envs 8192 --max_iterations 800 --headless > /workspace/sac.log 2>&1 < /dev/null &
```

SAC 推荐 `--num_envs 8192 --max_iterations 800`(论文协议,G1 rough 实测复现 +21~+23)。

## 1.4 看进度

```bash
grep -E "Learning iteration|Mean reward|Time elapsed" /workspace/train.log | tail -6
```

前 1-2 分钟是 Isaac Sim 启动,没输出是正常的。

## 1.5 停止

先查 PID 再杀(**不要 pkill -f train.py**,会误杀含该字样的自身命令串;SIGTERM 杀不净时对残留 PID 用 `kill -9`,再 `ps` 确认):

```bash
pgrep -f train.py
```

```bash
kill <PID>
```

---

# 二、训练产物

依次执行(全程留在 `/workspace/isaaclab`):

**① 诊断报告**(TensorBoard 曲线 + 实际生效参数 → 一份 report.md,适合整体粘给 AI 分析):

```bash
./isaaclab.sh -p training-service/make_report.py --latest
```

**② 找 run 目录**:

```bash
ls -dt /workspace/isaaclab/logs/rsl_rl/*/*/ | head -1
```

**③ 评估视频 + 导出部署模型**(`--agent` 必须与训练一致;SAC 加 `--agent rsl_rl_sac_cfg_entry_point`):

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-G1-v0 --num_envs 32 --headless --video --video_length 400 --checkpoint <run目录>/model_799.pt > /workspace/eval.log 2>&1 < /dev/null &
```

产物:`videos/play/*.mp4`(评估视频)、`exported/policy.onnx` + `policy.pt`(**部署用**;run 目录里的 `model_*.pt` 是续训用的,别拿去部署)。

**④ 上传 TOS**(凭据从 pod 环境变量自动读;`--local-dir` 必须指定,否则会把全部历史 run 传上去):

```bash
./isaaclab.sh -p training-service/upload_tos.py --local-dir <run目录> --prefix my-run
```

---

# 三、改参数(全部免代码)

两种写法,按参数性质选:

| 写法 | 用在哪 | 写错了会怎样 |
|---|---|---|
| Hydra override | 命令行尾部,如 `agent.algorithm.gamma=0.99` | 启动时报错,指出是哪个键,不会静默 |
| 命令行旗标 | `--num_envs` / `--max_iterations` / `--seed` | 优先级最高,写成 override 会被静默覆盖 |

**参数名从哪查**:每次训练的 run 目录自动保存 `params/agent.yaml`(算法超参)和 `params/env.yaml`(环境参数),**里面的树形路径逐级用点连起来就是 override 语法**:

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --num_envs 4096 --headless agent.algorithm.gamma=0.99 env.rewards.track_lin_vel_xy_exp.weight=2.0 > /workspace/train.log 2>&1 < /dev/null &
```

规则:Hydra 覆盖放所有 `--` 旗标之后;列表值方括号内不能有空格(`[-1.5,1.5]`)。

⚠️ **唯一会静默失效的坑**:`num_envs` / `max_iterations` / `seed` 是命令行旗标,写进 override 会被旗标值悄悄覆盖 —— 这三个只能走旗标。

## SAC 专属超参(agent.algorithm.*,默认值 = 论文配置)

| override | 默认 | 说明 |
|---|---|---|
| `agent.algorithm.gamma=0.99` | 0.97 | 折扣因子 |
| `agent.algorithm.num_mini_batches=100` | 200 | 更新强度(决定 UTD) |
| `agent.algorithm.replay_buffer_size=3000000` | 5000000 | buffer 容量(显存大头,见下) |
| `agent.algorithm.n_steps=3` | 5 | n-step 回报 |
| `agent.algorithm.actor_learning_rate=1e-4` | 2e-4 | actor 学习率 |
| `agent.algorithm.target_entropy_scale=0.2` | 0.167 | 熵目标系数 |
| `agent.algorithm.tau=0.005` | 0.003 | target 软更新速率 |

⚠️ 默认值是 ETH 论文原版,已多次复现 —— **改前想清楚**。实测把 `num_mini_batches` 改成 12(UTD 8.3→0.5)训练直接崩到 −43。

---

# 四、SAC 适用场景(什么时候选它)

依据:[ETH RSL-RL-SAC 论文](https://arxiv.org/abs/2605.24975) + 本仓库 A100 实测(G1, seed 42, 各自推荐配置)。

**选 SAC 的场景:**

- **困难地形 / 人形机器人**。论文结论"人形任务上 SAC 超越 PPO",实测印证:G1 崎岖地形 SAC 28 分钟即达到 PPO 47 分钟的最终水平,继续训练到 +21.07(PPO 1500 轮止步 +11.54)。
- **需要真机微调的流程**。off-policy 意味着仿真预训练的策略可以在真机上用同一算法直接微调,不需要大规模并行采样 —— on-policy 的 PPO 做不到。
- **样本昂贵的任务**。SAC 的 replay buffer 反复利用数据,单位数据的学习量远高于 PPO。

**不选 SAC 的场景:**

- **简单任务(平地等)**:实测 PPO 26 分钟收敛且质量相当,SAC 反而慢(单轮计算更重:网络更大、三个 optimizer、UTD 高)。
- **操作类任务(机械臂 reach/lift)**:只有 PPO 配置。上游 Isaac Lab 和 ETH 论文都没有操作类的
  SAC 超参可参照,自行调一组也无从验证对错,因此不提供。
- **需要断点续训的长任务**:off-policy 续训尚不支持(replay buffer 不落盘,恢复即冷启动退化)。

**资源注意**:replay buffer 在 GPU 上全额预分配(观测存两份)。参考:8192 env + 5M buffer ≈ 占满 A30(24G)的 80%;显存小的卡按比例降 `replay_buffer_size` 或 `num_envs`。

**墙钟如实说**:SAC 的优势是样本效率和最终高度,**不是单轮速度**(同轮数墙钟约为 PPO 的 1.9×)。宣传语境里"SAC 20 分钟训会走路"出自 FlashSAC(闭源,本仓库未集成),与本仓库的 ETH RSL SAC 不是一回事。

---

# 五、训练画面回传(可选)

演示用(与正式训练互斥:渲染拖垮吞吐,环境数调小):

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 16 --max_iterations 5000 --headless --stream > /workspace/stream.log 2>&1 < /dev/null &
```

约 60 秒后确认(应看到 `publishing training video to room ...`):

```bash
grep -aF "[livekit]" /workspace/stream.log | tail -2
```

LiveKit 凭据由集群 Secret `livekit-creds` 自动注入,无需配置;若日志报 `are not set`,说明 pod 建于 Secret 之前 —— 找管理员重建 pod。观看端用「Isaac 训练画面」桌面 app(地址/口令找管理员)。

---

# 六、Sim 工作站(交互式 3D 编辑器,可选)

**硬门槛:GPU 必须带 NVENC 编码器**(L4/L20/4090 可以;A30/A100/H20 不行 —— 症状是一切正常但画面永远不来)。

启动(pod 需 hostNetwork + EIP,见 `~/isaac-k8s/` 清单;`<公网IP>` 换成节点 EIP):

```bash
kubectl exec <pod> -- bash -c 'export OMNI_KIT_ALLOW_ROOT=1; cd /workspace/isaacsim && nohup ./isaac-sim.sh --no-window --allow-root --no-ros-env --/app/livestream/publicEndpointAddress=<公网IP> --/app/livestream/port=49100 --/app/livestream/allowDynamicResize=true --enable omni.services.livestream.nvcf > /workspace/workstation.log 2>&1 &'
```

客户端:NVIDIA 官方 **Isaac Sim WebRTC Streaming Client**(推荐)或「Isaac Sim 工作站」app,Server 填节点公网 IP。

要点速记:每次服务重启 ≈ 一个会话,换人先重启;安全组要放行 49100/TCP + 47998/UDP(和集群 common 安全组**并存**,不能替换);漏 UDP 的症状是连上但黑屏。

---

# 七、常见问题

| 症状 | 原因 |
|---|---|
| `pgrep train.py` 查不到但训练在跑 | 少了 `-f`(进程名是 python3) |
| override 写了没生效、也不报错 | 写的是 num_envs/max_iterations/seed —— 只能走旗标 |
| `Could not override 'xxx'. Key not in struct` | override 路径打错,对照 run 目录 `params/*.yaml` |
| SAC 起手 OOM | 降 `replay_buffer_size` 或 `num_envs`(buffer 全额预分配) |
| play.py 加载失败 | `--agent` 与训练不一致 |
| 断开 ssh 后训练消失 | 启动时没带 `setsid` |
| 推流日志 `credentials are not set` | pod 建于 livekit-creds Secret 之前,重建 pod |
