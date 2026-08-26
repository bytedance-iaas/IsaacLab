# 命令速查

可直接复制的命令,每条单行。参数含义与背景见 `QUICKSTART.md`。

标 **[本机]** 的在自己电脑上执行,其余在训练容器内执行。容器名以实际部署为准,下文用
`isaac-train-0`(训练)与 `isaac-render-0`(工作站)。

## 可调整参数

下表列出命令中各可替换部分,后文所有命令均按此调整。

| 调整项 | 写法 |
|---|---|
| 机器人与任务 | `--task <任务ID>`;机械臂类任务需同时更换入口,见末行 |
| 算法 | 加 `--agent rsl_rl_sac_cfg_entry_point` 为 SAC,不加为 PPO(SAC 仅移动类可用) |
| 训练规模 | `--num_envs`(上限见 QUICKSTART 的显存表)、`--max_iterations` |
| 随机种子 | `--seed 42` |
| 环境参数(奖励权重、指令范围等) | 命令**末尾**追加 `路径=值`,如 `env.rewards.track_lin_vel_xy_exp.weight=2.0` |
| 算法超参 | 同上,如 `agent.algorithm.gamma=0.99`;参数名取自训练目录下 `params/*.yaml` 中的树形路径 |
| 日志路径 | `> /workspace/<名称>.log` |
| 训练脚本入口 | 移动类:`-p scripts/reinforcement_learning/rsl_rl/train.py`;机械臂类:`-p -m isaac_so_arm101.scripts.rsl_rl.train`。两者都支持 `--stream` |

⚠️ `--num_envs`、`--max_iterations`、`--seed` 三项只能以旗标形式给出;写成末尾覆盖会被静默忽略。

---

## 进入容器与状态检查

**[本机]**

```bash
kubectl exec -it isaac-train-0 -- bash
```

```bash
cd /workspace/isaaclab && pgrep -f train.py
```

有输出表示已有训练在运行。执行 `kill <PID>` 后**需再次确认输出为空**,仍有残留则执行 `kill -9 <PID>`。

## 发起训练

移动类任务,SAC:

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --agent rsl_rl_sac_cfg_entry_point --num_envs 8192 --max_iterations 800 --headless > /workspace/sac_g1.log 2>&1 < /dev/null &
```

机械臂类任务,PPO:

```bash
setsid nohup ./isaaclab.sh -p -m isaac_so_arm101.scripts.rsl_rl.train --task Isaac-SO-ARM101-PnP-Bin-v0 --num_envs 4096 --max_iterations 1500 --headless > /workspace/pnp.log 2>&1 < /dev/null &
```

带参数覆盖:

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-G1-v0 --num_envs 4096 --headless agent.algorithm.gamma=0.99 env.rewards.track_lin_vel_xy_exp.weight=2.0 > /workspace/train.log 2>&1 < /dev/null &
```

`setsid` 不可省略,否则断开连接后训练随即终止。

## 查看进度

```bash
grep -aE "Learning iteration|Mean reward|Time elapsed" /workspace/sac_g1.log | tail -6
```

## 推流查看训练画面

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-G1-v0 --num_envs 16 --max_iterations 5000 --headless --stream > /workspace/stream.log 2>&1 < /dev/null &
```

约 60 秒后确认,出现 `publishing training video to room ...` 即为成功,随后在浏览器打开观看页面:

```bash
grep -aF "[livekit]" /workspace/stream.log | tail -2
```

`--stream` 会开启相机渲染,`--num_envs` 应保持在几十以内。

## 训练产物

诊断报告:

```bash
./isaaclab.sh -p training-service/make_report.py --latest
```

定位本次训练目录:

```bash
ls -dt /workspace/isaaclab/logs/rsl_rl/*/*/ | head -1
```

录制评估视频并导出 ONNX(`--agent` 需与训练时一致;checkpoint 文件名为 `model_<轮数-1>.pt`):

```bash
setsid nohup ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-G1-v0 --agent rsl_rl_sac_cfg_entry_point --num_envs 32 --headless --video --video_length 400 --checkpoint <训练目录>/model_799.pt > /workspace/eval.log 2>&1 < /dev/null &
```

机械臂类任务同样更换入口:

```bash
setsid nohup ./isaaclab.sh -p -m isaac_so_arm101.scripts.rsl_rl.play --task Isaac-SO-ARM101-PnP-Bin-Play-v0 --num_envs 32 --headless --video --video_length 400 --checkpoint <训练目录>/model_1499.pt > /workspace/eval.log 2>&1 < /dev/null &
```

## 量取资产尺寸

**[本机]**

```bash
kubectl cp ./my_object.usd isaac-train-0:/tmp/my_object.usd
```

```bash
/workspace/isaaclab/_isaac_sim/kit/python/bin/python3 training-service/inspect_usd.py /tmp/my_object.usd
```

## 归档产物:上传 TOS

**[本机]** 首次使用前创建 Secret,键名需与下方一致(桶不在华北区域时另需提供 `TOS_ENDPOINT`、
`TOS_REGION`),并在部署时以 `envFromSecrets: [tos-creds]` 挂载:

```bash
kubectl create secret generic tos-creds -n <命名空间> --from-literal=TOS_ACCESS_KEY=<AK> --from-literal=TOS_SECRET_KEY=<SK> --from-literal=TOS_BUCKET=<桶名>
```

确认凭据已注入(输出应为 3),随后上传:

```bash
env | grep -c TOS_
```

```bash
./isaaclab.sh -p training-service/upload_tos.py --local-dir <训练目录> --prefix <前缀>
```

未指定 `--local-dir` 时会上传整个 `logs/` 目录,即该容器上的全部历史训练。

## 归档产物:拷回本地

**[本机]** 整个目录(含各轮 checkpoint,通常数百 MB 起):

```bash
kubectl cp isaac-train-0:<训练目录> ./run_archive
```

仅需结果时可按需单独取 `report.md`、`exported`、`videos`:

```bash
kubectl cp isaac-train-0:<训练目录>/exported ./exported
```

## 取回工作站中的场景文件

**[本机]**

```bash
kubectl cp isaac-render-0:/root/so101_pnp_example.usd ./so101_pnp_example.usd
```
