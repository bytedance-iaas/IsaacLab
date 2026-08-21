"""Inspect a USD for sufficient physics information and report readable findings.

Written for the person about to reference an asset from a task they are writing by hand: the
report carries the measured bounding box, which is where the poses and the geometry-derived
reward thresholds in the env cfg come from, and it says which physics the asset is missing.

Uses a standalone usd-core (an independent pxr) instead of starting the Isaac Sim app, which is
fast (~1s) and reliable. usd-core is installed in /opt/usd-core (see the Dockerfile) for this
script alone, so it does not pollute the training environment.

Usage: python inspect_usd.py <usd_path> [out.json]
  The report is printed as a single "__USD_REPORT__ <json>" line and optionally written to out.json.
"""
from __future__ import annotations

import json
import sys

# Standalone usd-core, used only by this process, to avoid clashing with Kit's pxr
sys.path.insert(0, "/opt/usd-core")


def inspect(path: str) -> dict:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(path)
    if not stage:
        return {"ok": False, "self_describing": False,
                "messages": ["Cannot open the USD: the file may be corrupt or not a valid .usd."]}

    has_rigid = has_collision = has_mass = False
    mass = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            has_rigid = True
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            has_collision = True
        if prim.HasAPI(UsdPhysics.MassAPI):
            has_mass = True
            m = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            if m:
                mass = float(m)

    try:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rng = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        size = rng.GetSize() if not rng.IsEmpty() else Gf.Vec3d(0, 0, 0)
    except Exception:
        size = None
    size_m = [round(float(size[i]), 3) for i in range(3)] if size else [0, 0, 0]
    max_dim = max(size_m) if size_m else 0

    messages = []
    if not has_rigid:
        messages.append("The USD defines no RigidBody; one will be added so the object can be grasped and moved.")
    if not has_collision:
        messages.append("The USD has no Collision geometry; a convex hull will be used, which may be imprecise for complex shapes.")
    if not has_mass:
        messages.append("The USD specifies no Mass; it will be estimated from a default density and can be adjusted in the preview.")
    if max_dim == 0:
        messages.append("No visible geometry was found in the USD; please check the file.")
    elif max_dim > 1.0:
        messages.append(f"The object measures about {max_dim} m, which is large; the file may be in mm or cm, so consider setting a scale.")
    elif max_dim < 0.005:
        messages.append(f"The object measures about {max_dim} m, which is small; the units may be wrong, so consider setting a scale.")

    self_describing = has_rigid and has_collision and has_mass
    return {
        "ok": max_dim > 0,
        "self_describing": self_describing,   # True: the USD carries full physics and can be used as is; False: defaults must be filled in
        "has_rigid": has_rigid, "has_collision": has_collision, "has_mass": has_mass,
        "mass": mass, "size_m": size_m,
        "messages": messages or ["The USD is complete and can be used as is."],
    }


def main() -> int:
    usd_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    try:
        report = inspect(usd_path)
    except Exception as e:  # noqa: BLE001
        report = {"ok": False, "self_describing": False, "messages": [f"Inspection failed: {type(e).__name__}: {e}"]}
    s = json.dumps(report, ensure_ascii=False)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(s)
    print("__USD_REPORT__ " + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
