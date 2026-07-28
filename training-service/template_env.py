"""Template -> env cfg generator (phases 1 and 2).

Approach: take the env cfg of a registered base_task and add, remove or modify parts of it
according to the template:
  - scene.assets: swap the robot or object USD, add obstacles and objects (phase 1)
  - rewards / events: change weights and params, or add new terms from the task's mdp package (phase 2)

Hydra is enough for value-only overrides; this module handles adding and removing structural
items, which Hydra cannot do.
It imports isaaclab, so it only runs under the container's isaac python.
"""
from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg  # noqa: F401  (referenced from template params)

# Built-in asset library: friendly ref -> USD path (extend as needed)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

BUILTIN_USD = {
    "dex_cube": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    "seattle_lab_table": f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
}
# Where uploaded assets land inside the job (downloaded from TOS)
UPLOAD_DIR = "/assets"

# Primitive shape -> spawn cfg class
PRIMITIVES = {
    "box": sim_utils.CuboidCfg,
    "sphere": sim_utils.SphereCfg,
    "cylinder": sim_utils.CylinderCfg,
    "capsule": sim_utils.CapsuleCfg,
    "cone": sim_utils.ConeCfg,
}


def _spawn_from_spec(spec: dict[str, Any], is_rigid: bool):
    """Turn an asset spawn spec into an isaaclab spawn cfg. Supports builtin, upload (USD) and
    primitive sources.

    is_rigid decides whether rigid-body physics is attached:
      - True (kind=rigid, a movable object): adds RigidBodyProperties, Mass and Collision
      - False (kind=static, a fixed obstacle or table): adds Collision only
    """
    sp = spec.get("spawn", {})
    src = sp.get("source", "builtin")
    phys = spec.get("physics", {})
    if phys.get("from_usd"):
        # The USD already carries full physics (inspect_usd reports self_describing) -> keep it as is
        rigid = mass = collision = None
    else:
        # Geometry-only USD or a primitive -> fill in default physics (mass, collision, rigid body)
        # so the user never has to specify things like an inertia tensor
        rigid = sim_utils.RigidBodyPropertiesCfg() if is_rigid else None
        mass = sim_utils.MassPropertiesCfg(mass=phys.get("mass", 0.1)) if is_rigid else None
        collision = sim_utils.CollisionPropertiesCfg()

    if src in ("builtin", "upload"):
        usd_path = BUILTIN_USD[sp["ref"]] if src == "builtin" else f"{UPLOAD_DIR}/{sp['ref']}"
        scale = tuple(sp.get("scale", (1.0, 1.0, 1.0)))
        return sim_utils.UsdFileCfg(
            usd_path=usd_path, scale=scale,
            rigid_props=rigid, mass_props=mass, collision_props=collision,
        )
    if src == "primitive":
        cls = PRIMITIVES[sp["shape"]]
        kw = {"rigid_props": rigid, "mass_props": mass, "collision_props": collision,
              "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(sp.get("color", (0.6, 0.6, 0.6))))}
        if sp["shape"] == "box":
            kw["size"] = tuple(sp.get("size", (0.05, 0.05, 0.05)))
        else:
            kw["radius"] = sp.get("radius", 0.03)
            if sp["shape"] in ("cylinder", "capsule", "cone"):
                kw["height"] = sp.get("height", 0.06)
        return cls(**kw)
    raise ValueError(f"unknown spawn source: {src}")


def _build_asset(spec: dict[str, Any]):
    """Asset spec -> isaaclab asset cfg. kind is rigid (movable) or static (fixed, collision only)."""
    is_rigid = spec["kind"] == "rigid"
    spawn = _spawn_from_spec(spec, is_rigid)
    prim = f"{{ENV_REGEX_NS}}/{spec['name'].capitalize()}"
    init = None
    if "pos" in spec or "rot" in spec:
        init_kw = {}
        if "pos" in spec:
            init_kw["pos"] = tuple(spec["pos"])
        if "rot" in spec:
            init_kw["rot"] = tuple(spec["rot"])
        init = (RigidObjectCfg if spec["kind"] == "rigid" else AssetBaseCfg).InitialStateCfg(**init_kw)
    Cfg = RigidObjectCfg if spec["kind"] == "rigid" else AssetBaseCfg
    kw = {"prim_path": prim, "spawn": spawn}
    if init is not None:
        kw["init_state"] = init
    return Cfg(**kw)


def _fix_entity_refs(env_cfg, asset_name: str) -> list[str]:
    """After replacing a scene entity, rewrite any SceneEntityCfg in events/rewards/terminations/
    curriculum that references it with a hard-coded body_names to body_names=None (i.e. the root),
    so a new USD with different internal body names does not break resolution."""
    from isaaclab.managers import SceneEntityCfg as _SEC

    fixed = []
    for mgr_name in ("events", "rewards", "terminations", "curriculum"):
        mgr = getattr(env_cfg, mgr_name, None)
        if mgr is None:
            continue
        for term_name, term in vars(mgr).items():
            params = getattr(term, "params", None)
            if not isinstance(params, dict):
                continue
            for v in params.values():
                if isinstance(v, _SEC) and v.name == asset_name and v.body_names:
                    v.body_names = None
                    fixed.append(f"{mgr_name}.{term_name}")
    return fixed


def apply_scene(env_cfg, scene_spec: dict[str, Any]) -> list[str]:
    """Apply the template to the scene: add or replace rigid/static assets. Returns a list of notes."""
    notes = []
    if "num_envs" in scene_spec:
        env_cfg.scene.num_envs = scene_spec["num_envs"]
    if "env_spacing" in scene_spec:
        env_cfg.scene.env_spacing = scene_spec["env_spacing"]
    for spec in scene_spec.get("assets", []):
        name = spec["name"]
        if spec["kind"] in ("rigid", "static"):
            replacing = hasattr(env_cfg.scene, name) and getattr(env_cfg.scene, name) is not None
            setattr(env_cfg.scene, name, _build_asset(spec))
            notes.append(f"scene.{name} ← {spec['kind']} ({spec.get('spawn', {}).get('source')})")
            if replacing:  # replacing an existing entity -> fix body_names that reference it
                refs = _fix_entity_refs(env_cfg, name)
                if refs:
                    notes.append(f"  fixed body_names referencing {name}: {', '.join(refs)}")
    return notes


def apply_rewards(env_cfg, rewards_spec: list[dict], mdp) -> list[str]:
    """Modify or add reward terms. An existing term has its weight/params updated; a new one is
    created from a func in the task's mdp package."""
    notes = []
    for r in rewards_spec:
        name = r["name"]
        term = getattr(env_cfg.rewards, name, None)
        if term is not None:
            if "weight" in r:
                term.weight = r["weight"]
            if "params" in r:
                term.params.update(r["params"])
            notes.append(f"rewards.{name} weight={term.weight}")
        else:
            func = getattr(mdp, r["func"])
            setattr(env_cfg.rewards, name, RewTerm(func=func, weight=r["weight"], params=r.get("params", {})))
            notes.append(f"rewards.{name} += {r['func']} w={r['weight']}")
    return notes


def apply_events(env_cfg, events_spec: list[dict], mdp) -> list[str]:
    notes = []
    for e in events_spec:
        name = e["name"]
        term = getattr(env_cfg.events, name, None)
        if term is not None and "params" in e:
            term.params.update(e["params"])
            notes.append(f"events.{name} params updated")
        elif term is None:
            func = getattr(mdp, e["func"])
            setattr(env_cfg.events, name, EventTerm(func=func, mode=e.get("mode", "reset"), params=e.get("params", {})))
            notes.append(f"events.{name} += {e['func']}")
    return notes


def apply_template(env_cfg, template: dict[str, Any], mdp) -> list[str]:
    """Apply the template to a base env_cfg in place. mdp is the task's mdp module, which provides
    the reward and event functions."""
    notes = []
    if "scene" in template:
        notes += apply_scene(env_cfg, template["scene"])
    if "rewards" in template:
        notes += apply_rewards(env_cfg, template["rewards"], mdp)
    if "events" in template:
        notes += apply_events(env_cfg, template["events"], mdp)
    if "env" in template and "episode_length_s" in template["env"]:
        env_cfg.episode_length_s = template["env"]["episode_length_s"]
    return notes
