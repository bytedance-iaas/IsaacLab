# SO-ARM101 PnP-Bin:手写任务的范例

本目录是一个**自定义任务的完整范例**:把一个在 Isaac Sim 工作站里搭出来的场景,翻译成可训练的任务。

任务 ID:`Isaac-SO-ARM101-PnP-Bin-v0`(评估用 `Isaac-SO-ARM101-PnP-Bin-Play-v0`)
任务内容:SO-101 机械臂把桌面上的方块放进料箱。

## 这个任务是怎么来的

场景在工作站的图形界面里搭建(机械臂、光学平台、方块、料箱),随后手工翻译成本目录下的配置。
`pnp_env_cfg.py` 的文件头逐条记录了翻译过程中的判断,包括:

- **所有位姿都保持场景原样**,机器人、桌子、方块、料箱的相对关系与原文件一致,训练回传画面
  跟在工作站里看到的是同一个场景;
- 每个尺寸是量出来的,不是估的(平台包围盒、台面高度、料箱脚印);
- 方块从"一个位姿"改成"一片区域":场景只能描述一次摆放,任务需要的是分布,否则策略会把唯一的
  那次伸手背下来。抖动范围由可达距离和料箱间距推出;
- 方块直接引用场景中的原件(DexCube),料箱则按实测脚印用图元重建。重建的原因有两条,第二条更
  值得注意:原件内置了一盏灯,会被复制到每个并行环境;而且原件缩放后只有 5.9cm 高、内底贴近箱
  底,方块放进去中心约在 z=0.02,**低于"算抬起来"的 0.025 阈值** —— 而这个阈值同时 gate 了
  目标奖励,直接引用会让 37 分奖励里的 36 分在任务成功那一刻归零,且不报任何错。

这类判断场景文件本身无法表达,也是这一步必须由人完成的原因。奖励与终止条件复用 lift 任务的 mdp 库。

---

## 训练与评估本任务

必须使用包自带入口 `-m isaac_so_arm101.scripts.rsl_rl.train`,核心训练脚本不认识这些任务 ID。

```bash
setsid nohup ./isaaclab.sh -p -m isaac_so_arm101.scripts.rsl_rl.train --task Isaac-SO-ARM101-PnP-Bin-v0 --num_envs 4096 --max_iterations 1500 --headless > /workspace/pnp.log 2>&1 < /dev/null &
```

```bash
grep -aE "Learning iteration|Mean reward" /workspace/pnp.log | tail -4
```

录制评估视频并导出可部署模型(入口与训练时一致):

```bash
setsid nohup ./isaaclab.sh -p -m isaac_so_arm101.scripts.rsl_rl.play --task Isaac-SO-ARM101-PnP-Bin-Play-v0 --num_envs 32 --headless --video --video_length 400 --checkpoint <训练目录>/model_1499.pt > /workspace/eval.log 2>&1 < /dev/null &
```

诊断报告、产物归档等与任务无关的步骤,见仓库根目录的 `EXAMPLE_COMMANDS.md`。

## 已知边界

- **只有 PPO 配置**。上游 Isaac Lab 与 ETH 论文均未提供可参照的操作类 SAC 超参,自行调一组也无从
  验证正确性,因此不提供。
- **并行环境数上限 32768**,与显卡无关:每个环境的物体各自带材质,更高的并行数会触发物理引擎
  6.5 万材质的硬上限。
