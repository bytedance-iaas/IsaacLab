"""Turn a finished run's TensorBoard log into a single markdown diagnostic sheet.

The sheet reports *measurements only*. It states what each quantity is and what this run's numbers
were; it does not say whether they are good, nor what to change. That judgement needs to know the
user's robot, their tolerance for wall-clock, and what they are actually trying to achieve -- none
of which we have. Users who want an opinion can paste this file into an AI along with their goal,
which is the workflow it is shaped for: dense, labelled, self-contained, and small enough to fit in
a prompt.

The quantities picked here are the ones that were actually load-bearing when debugging the G1 SAC
runs -- termination composition, entropy-target tracking, and reward-term decomposition each
localized a real problem faster than the reward curve did.

Usage:
  python make_report.py --run <log-dir> [--out <log-dir>/report.md]
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import yaml
from tensorboard.backend.event_processing import event_accumulator

# Tail fraction used for the trend measurement at the end of training
TAIL_FRAC = 0.2


class _TolerantLoader(yaml.SafeLoader):
    """env.yaml is dumped with python-specific tags (tuples, enums, class references). unsafe_load
    would resolve them by importing and calling arbitrary constructors, which is not acceptable for
    a file that can originate from a user-supplied config. Reduce unknown tags to plain containers
    instead -- the report only ever reads scalars out of this tree."""


def _plain(loader: yaml.Loader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_TolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _plain)
_TolerantLoader.add_multi_constructor("!", _plain)


def _read_yaml(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_TolerantLoader) or {}


def _load(run: str) -> dict[str, np.ndarray]:
    """Read every scalar series from the run's event file, keyed by tag."""
    ea = event_accumulator.EventAccumulator(run, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return {t: np.array([s.value for s in ea.Scalars(t)]) for t in ea.Tags()["scalars"]}


def _trend(y: np.ndarray) -> tuple[float, float, int]:
    """Least-squares slope over the last TAIL_FRAC of training, in units per 100 iterations, plus
    the scatter of the points about that line. Comparing the two is what separates a curve that is
    still climbing from one that is only wandering inside its own noise."""
    n = max(int(len(y) * TAIL_FRAC), 2)
    tail = y[-n:]
    x = np.arange(n)
    slope, intercept = np.polyfit(x, tail, 1)
    residual_sd = float(np.std(tail - (slope * x + intercept)))
    return float(slope) * 100.0, residual_sd, n


def _fmt(v: float, width: int = 12) -> str:
    return f"{v:>{width}.4f}"


def _table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _section_headline(L: list[str], s: dict[str, np.ndarray], agent: dict[str, Any]) -> None:
    L.append("## 1. 结果")
    L.append("")
    rows = []
    for tag, label in (
        ("Train/mean_reward", "平均回合奖励(整段求和)"),
        ("Train/mean_episode_length", "平均回合长度(步)"),
    ):
        if tag in s:
            y = s[tag]
            rows.append((label, _fmt(y[0]), _fmt(y.max()), _fmt(y[-1])))

    # Episode reward is a sum over the episode, so it moves with episode length as well as with
    # policy quality. When length changes by a large factor over training the two effects are
    # confounded, and the per-step form is the one that isolates policy quality.
    rw, ln = s.get("Train/mean_reward"), s.get("Train/mean_episode_length")
    per_step = None
    if rw is not None and ln is not None:
        per_step = np.divide(rw, ln, out=np.zeros_like(rw), where=ln > 0)
        rows.append(("平均每步奖励(= 上面两项相除)", _fmt(per_step[0]), _fmt(per_step.max()), _fmt(per_step[-1])))

    L += _table(rows, ("量", "首轮", "训练中最大", "末轮"))
    L.append("")
    if per_step is not None and ln is not None and ln[0] > 0:
        L.append(
            f"> 回合长度在训练中变化了 {ln[-1] / ln[0]:.1f} 倍。回合奖励是整段求和,"
            "因此它同时受回合长度和策略质量影响;每步奖励只受后者影响。两行需要一起读。"
        )
        L.append("")
    L.append(f"总迭代数 {len(s.get('Train/mean_reward', []))},配置的 `max_iterations` = {agent.get('max_iterations', '?')}。")
    L.append("")


def _section_scale(L: list[str], s: dict[str, np.ndarray], agent: dict[str, Any], env: dict[str, Any], algo: str) -> None:
    """Sample budget: how many transitions were collected and how many were actually used for
    gradient updates. For off-policy runs these two numbers are set independently, so their ratio
    is a configured quantity rather than a derived one, and it is not visible anywhere else."""
    scene = env.get("scene") or {}
    n_envs = scene.get("num_envs")
    n_steps = agent.get("num_steps_per_env")
    iters = len(s.get("Train/mean_reward", []))
    if not (n_envs and n_steps and iters):
        return

    alg = agent.get("algorithm") or {}
    per_iter = int(n_envs) * int(n_steps)
    rows = [
        ("并行环境数 `num_envs`", f"{int(n_envs):,}"),
        ("每环境每轮步数 `num_steps_per_env`", f"{int(n_steps):,}"),
        ("每轮采集的 transition 数", f"{per_iter:,}"),
        ("全程采集的 transition 数", f"{per_iter * iters:,}"),
    ]

    epochs = alg.get("num_learning_epochs")
    if algo == "SAC":
        nmb, mbs = alg.get("num_mini_batches"), alg.get("mini_batch_size")
        if nmb and mbs and epochs:
            used = int(epochs) * int(nmb) * int(mbs)
            rows += [
                ("每轮用于梯度更新的样本数", f"{used:,}  (= {epochs} × {nmb} × {mbs:,})"),
                ("**UTD**(更新数 ÷ 采集数)", f"**{used / per_iter:.2f}**"),
                ("全程梯度步数", f"{int(epochs) * int(nmb) * iters:,}"),
            ]
        buf = alg.get("replay_buffer_size")
        if buf:
            rows.append(("replay buffer 深度", f"{int(buf):,} 条 = {int(buf) / per_iter:.1f} 轮的数据"))
    elif epochs:
        rows += [
            ("每轮用于梯度更新的样本数", f"{per_iter * int(epochs):,}  (= 采集数 × {epochs} 个 epoch)"),
            ("**UTD**(更新数 ÷ 采集数)", f"**{float(epochs):.2f}**"),
        ]

    L.append("## 2. 样本预算")
    L.append("")
    L += _table(rows, ("量", "值"))
    L.append("")
    L.append("> UTD(update-to-data)= 每采集一条 transition 平均做了多少次梯度更新。")
    L.append("> on-policy 算法采完即用即弃,UTD 等于 epoch 数;off-policy 从 replay buffer 反复采样,")
    L.append("> UTD 由 `num_mini_batches` × `mini_batch_size` 与采集量独立决定。")
    L.append("")


def _section_trend(L: list[str], s: dict[str, np.ndarray]) -> None:
    if "Train/mean_reward" not in s:
        return
    slope, noise, n = _trend(s["Train/mean_reward"])
    L.append("## 3. 末段趋势")
    L.append("")
    L.append(f"对最后 {n} 轮({int(TAIL_FRAC * 100)}%)的 `Train/mean_reward` 做最小二乘直线拟合:")
    L.append("")
    L += _table(
        [
            ("拟合斜率", f"{slope:+.4f} / 100 轮"),
            ("点相对该直线的标准差", f"{noise:.4f}"),
            ("斜率 ÷ 标准差", f"{slope / noise:+.3f}" if noise > 0 else "n/a"),
        ],
        ("量", "值"),
    )
    L.append("")
    L.append("> 斜率是末段每 100 轮的奖励变化量;标准差是同一段里点偏离该直线的幅度。")
    L.append("")


def _section_termination(L: list[str], s: dict[str, np.ndarray]) -> None:
    tags = sorted(t for t in s if t.startswith("Episode_Termination/"))
    if not tags:
        return
    # The categories partition every episode, so they must sum to 1. Printing the sum makes any
    # iteration where they do not -- the first one is typically still warming up -- visible rather
    # than silently distorting the comparison.
    total = np.sum([s[t] for t in tags], axis=0)
    first = 1 if len(total) > 1 and abs(total[0] - 1.0) > 0.02 else 0

    L.append("## 4. 回合终止原因构成")
    L.append("")
    rows = [(t.split("/", 1)[1], _fmt(s[t][first]), _fmt(s[t][-1]), _fmt(s[t][-1] - s[t][first])) for t in tags]
    rows.append(("**合计**", _fmt(float(total[first])), _fmt(float(total[-1])), ""))
    L += _table(rows, ("终止原因", f"第 {first + 1} 轮", "末轮", "变化"))
    L.append("")
    L.append("> 每个回合按其结束原因归入一类,数值是该原因在当轮所有回合中的占比,各项合计应为 1。")
    L.append("> `time_out` 表示回合跑满了最大步数才结束,其余各项都是提前终止。")
    if first:
        L.append(f">")
        L.append(f"> 起始列取的是第 2 轮而非第 1 轮:第 1 轮各项合计仅 {total[0]:.4f},")
        L.append(f"> 回合统计缓冲尚未填满,该轮数值不可用于比较。")
    L.append("")


def _section_rewards(L: list[str], s: dict[str, np.ndarray], env: dict[str, Any]) -> None:
    tags = [t for t in s if t.startswith("Episode_Reward/")]
    if not tags:
        return
    tags.sort(key=lambda t: -abs(s[t][-1]))

    # The logged series is weight x term, so the weight is needed to tell "this penalty is
    # over-weighted" apart from "this quantity is genuinely large".
    weights = {k: v.get("weight") for k, v in (env.get("rewards") or {}).items() if isinstance(v, dict)}

    L.append("## 5. 奖励项分解")
    L.append("")
    L.append("按末轮绝对值排序 —— 排在前面的项主导了总奖励。")
    L.append("")
    rows = []
    for t in tags:
        name = t.split("/", 1)[1]
        w = weights.get(name)
        rows.append((name, "n/a" if w is None else f"{float(w):g}", _fmt(s[t][0]), _fmt(s[t][-1]), _fmt(s[t][-1] - s[t][0])))
    L += _table(rows, ("奖励项", "配置权重", "首轮", "末轮", "变化"))
    L.append("")
    L.append("> 各项之和即总奖励。正项为激励、负项为惩罚。")
    L.append("> 表中数值是**已乘过权重的结果**,所以某项数值大可能是权重高,也可能是该物理量本身大 ——")
    L.append("> 两者要靠权重列区分。完整奖励配置见同目录 `params/env.yaml`。")
    L.append("")


def _section_optimization(L: list[str], s: dict[str, np.ndarray], algo: str) -> None:
    """Loss and policy-entropy series. The tags differ between PPO and SAC, so emit whatever the
    run actually recorded rather than assuming one algorithm's set."""
    tags = sorted(t for t in s if t.startswith(("Loss/", "Policy/")))
    if not tags:
        return
    L.append("## 6. 优化过程")
    L.append("")
    rows = [(t, _fmt(s[t][0]), _fmt(s[t][-1]), _fmt(float(s[t].min())), _fmt(float(s[t].max()))) for t in tags]
    L += _table(rows, ("量", "首轮", "末轮", "最小", "最大"))
    L.append("")

    glossary = {
        "Loss/alpha": "SAC 的温度系数损失。SAC 把「策略熵达到目标熵」作为一个约束来优化,这一项就是该约束的残差:接近 0 表示当前熵与目标熵一致,绝对值大表示两者相差很远。目标熵由 `target_entropy_scale` × 动作维度决定。",
        "Policy/alpha": "SAC 的温度系数本身,即熵项在目标函数中的权重,由上面那个损失自动调节。",
        "Policy/mean_std": "策略输出动作分布的平均标准差,即探索幅度。初值由 `init_noise_std` 设定。",
        "Loss/critic1": "Q 网络的时序差分误差。",
        "Loss/critic2": "第二个 Q 网络的时序差分误差(SAC 用两个 Q 网取较小值)。",
        "Loss/actor": "策略网络损失。",
        "Loss/value_function": "PPO 的价值网络损失。",
        "Loss/surrogate": "PPO 的策略替代损失。",
        "Loss/entropy": "PPO 的策略熵。",
        "Loss/learning_rate": "当前学习率(可能被自适应调度改动)。",
    }
    present = [t for t in tags if t in glossary]
    if present:
        L.append("各项含义:")
        L.append("")
        for t in present:
            L.append(f"- **`{t}`** — {glossary[t]}")
        L.append("")


def _section_throughput(L: list[str], s: dict[str, np.ndarray], agent: dict[str, Any]) -> None:
    if not any(t.startswith("Perf/") for t in s):
        return
    L.append("## 7. 吞吐")
    L.append("")
    rows = []
    for tag, label in (
        ("Perf/total_fps", "总吞吐(环境步/秒)"),
        ("Perf/collection_time", "每轮采样耗时(秒)"),
        ("Perf/learning_time", "每轮梯度更新耗时(秒)"),
    ):
        if tag in s:
            y = s[tag]
            rows.append((label, _fmt(y[0]), _fmt(float(y.mean())), _fmt(y[-1])))
    L += _table(rows, ("量", "首轮", "全程均值", "末轮"))
    L.append("")

    col, lea = s.get("Perf/collection_time"), s.get("Perf/learning_time")
    if col is not None and lea is not None:
        per_iter = float(col.mean() + lea.mean())
        share = float(lea.mean()) / per_iter * 100 if per_iter else 0.0
        L.append(f"单轮平均 {per_iter:.2f} 秒,其中梯度更新占 {share:.0f}%;")
        L.append(f"全程 {len(col)} 轮合计约 {per_iter * len(col) / 60:.0f} 分钟(不含 Isaac Sim 启动时间)。")
        L.append("")


def _section_curriculum(L: list[str], s: dict[str, np.ndarray]) -> None:
    tags = sorted(t for t in s if t.startswith(("Curriculum/", "Metrics/")))
    if not tags:
        return
    L.append("## 8. 课程与跟踪误差")
    L.append("")
    rows = [(t, _fmt(s[t][0]), _fmt(s[t][-1]), _fmt(s[t][-1] - s[t][0])) for t in tags]
    L += _table(rows, ("量", "首轮", "末轮", "变化"))
    L.append("")
    L.append("> `Curriculum/*` 是当前难度等级;课程开启时难度随表现自动提升,")
    L.append("> 因此后期的跟踪误差是在更难的条件下测得的,与前期不可直接比较。")
    L.append("")


def _section_curve(L: list[str], s: dict[str, np.ndarray]) -> None:
    if "Train/mean_reward" not in s:
        return
    y = s["Train/mean_reward"]
    L.append("## 9. 学习曲线采样")
    L.append("")
    idx = [int(len(y) * f) for f in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)] + [len(y) - 1]
    rows = [(str(i + 1), _fmt(float(y[i]))) for i in idx]
    L += _table(rows, ("迭代", "平均回合奖励"))
    L.append("")


def _section_config(L: list[str], agent: dict[str, Any]) -> None:
    L.append("## 10. 超参")
    L.append("")
    L.append("以下为本次训练实际生效的值,完整配置见同目录 `params/agent.yaml` 与 `params/env.yaml`。")
    L.append("")
    L.append("```yaml")
    L.append(yaml.safe_dump(agent.get("algorithm", {}), allow_unicode=True, sort_keys=False).rstrip())
    L.append("```")
    L.append("")
    top = {k: agent[k] for k in ("seed", "num_steps_per_env", "max_iterations", "class_name") if k in agent}
    if top:
        L.append("```yaml")
        L.append(yaml.safe_dump(top, allow_unicode=True, sort_keys=False).rstrip())
        L.append("```")
        L.append("")


def build_report(run: str) -> str:
    s = _load(run)
    if not s:
        raise SystemExit(f"[report] no scalar data found in {run}")

    agent = _read_yaml(os.path.join(run, "params", "agent.yaml"))
    env = _read_yaml(os.path.join(run, "params", "env.yaml"))

    runner = str(agent.get("class_name", ""))
    algo = "SAC" if "OffPolicy" in runner or "Policy/alpha" in s else "PPO"
    experiment = os.path.basename(os.path.dirname(run.rstrip("/")))

    L: list[str] = []
    L.append(f"# 训练诊断 — {experiment}")
    L.append("")
    L += _table(
        [
            ("实验", experiment),
            ("运行目录", os.path.basename(run.rstrip("/"))),
            ("算法", f"{algo}(runner `{runner or 'n/a'}`)"),
            ("随机种子", str(agent.get("seed", "?"))),
        ],
        ("", ""),
    )
    L.append("")
    L.append("本文件只记录测量值,不含结论或调参建议。")
    L.append("")
    L.append("---")
    L.append("")

    _section_headline(L, s, agent)
    _section_scale(L, s, agent, env, algo)
    _section_trend(L, s)
    _section_termination(L, s)
    _section_rewards(L, s, env)
    _section_optimization(L, s, algo)
    _section_throughput(L, s, agent)
    _section_curriculum(L, s)
    _section_curve(L, s)
    _section_config(L, agent)

    return "\n".join(L) + "\n"


def _latest_run(logs_root: str) -> str:
    """Newest directory that actually contains an event file. A training job does not know the
    timestamped path the runner chose, so it asks for the newest one instead of passing a path."""
    candidates = []
    for dirpath, _, files in os.walk(logs_root):
        if any(f.startswith("events.out.tfevents.") for f in files):
            candidates.append((os.path.getmtime(dirpath), dirpath))
    if not candidates:
        raise SystemExit(f"[report] no run with an event file found under {logs_root}")
    return max(candidates)[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a markdown diagnostic sheet for a finished run")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run directory (the one containing events.out.tfevents.*)")
    g.add_argument("--latest", action="store_true", help="use the most recent run under --logs-root")
    ap.add_argument("--logs-root", default="/workspace/isaaclab/logs", help="where to search when --latest is given")
    ap.add_argument("--out", default=None, help="output path (default: <run>/report.md)")
    args = ap.parse_args()

    run = args.run or _latest_run(args.logs_root)
    if args.latest:
        print(f"[report] latest run: {run}")

    out = args.out or os.path.join(run, "report.md")
    text = build_report(run)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[report] wrote {out} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
