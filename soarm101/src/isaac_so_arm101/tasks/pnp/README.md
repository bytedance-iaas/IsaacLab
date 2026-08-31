# SO-ARM101 PnP-Bin：手写任务的范例

本目录是一个**自定义任务的完整范例**：把一个在 Isaac Sim 工作站里搭出来的场景，翻译成可训练的任务。

任务 ID：`Isaac-SO-ARM101-PnP-Bin-v0`（评估用 `Isaac-SO-ARM101-PnP-Bin-Play-v0`）任务内容：SO-101 机械臂把桌面上的方块放进料箱。

## 从场景到任务配置

场景在工作站的图形界面里搭建（机械臂、光学平台、方块、料箱），随后手工翻译成本目录下的配置。
`pnp_env_cfg.py` 的文件头逐条记录了翻译过程中的判断，包括：

- **所有位姿都保持场景原样**，机器人、桌子、方块、料箱的相对关系与原文件一致，训练回传画面跟在工作站里看到的是同一个场景；
- 每个尺寸是量出来的，不是估的（平台包围盒、台面高度、料箱脚印）；
- 方块从"一个位姿"改成"一片区域"：场景只能描述一次摆放，任务需要的是分布，否则策略会把唯一的那次伸手背下来。抖动范围由可达距离和料箱间距推出；
- 方块直接引用场景中的原件（DexCube），料箱则按实测脚印用图元重建：原件内置光源会被复制进每个并行环境，且它过浅的箱底会让奖励在方块入箱时静默归零，不适合直接引用。细节见
  `pnp_env_cfg.py` 文件头。

这一步无法从场景文件自动生成：场景只承载几何，而奖励、观测与成功判据这些任务语义并不在其中，需要另行定义；上面这类物理细节的判断，也要在理解任务之后才能做出。奖励与终止条件复用 lift 任务的 mdp 库。

---

## 训练与评估本任务

必须使用包自带入口 `-m isaac_so_arm101.scripts.rsl_rl.train`，核心训练脚本不认识这些任务 ID。

```bash
setsid nohup ./isaaclab.sh -p -m isaac_so_arm101.scripts.rsl_rl.train --task Isaac-SO-ARM101-PnP-Bin-v0 --num_envs 4096 --max_iterations 1500 --headless > /workspace/pnp.log 2>&1 < /dev/null &
```

```bash
grep -aE "Learning iteration|Mean reward" /workspace/pnp.log | tail -4
```

录制评估视频并导出可部署模型（入口与训练时一致）：

```bash
setsid nohup ./isaaclab.sh -p -m isaac_so_arm101.scripts.rsl_rl.play --task Isaac-SO-ARM101-PnP-Bin-Play-v0 --num_envs 32 --headless --video --video_length 400 --checkpoint <训练目录>/model_1499.pt > /workspace/eval.log 2>&1 < /dev/null &
```

诊断报告、产物归档等与任务无关的步骤，见仓库根目录的 `EXAMPLE_COMMANDS.md`。

## 说明

- **只有 PPO 配置**。上游 Isaac Lab 与 ETH 论文均未提供可参照的操作类 SAC 超参。
- **并行环境数上限 32768**，与显卡无关：每个环境的物体各自带材质，更高的并行数会触发物理引擎
  6.5 万材质的硬上限。
