"""SO-101 pick-and-place: drop a cube into a bin standing on the table.

Translated by hand from a scene authored in the Isaac Sim GUI (so101_pnp.usd), and written to be
read as an example of that translation. A scene file carries geometry; a task also needs intent,
so the numbers below fall into three kinds, and it is worth knowing which is which before copying
this file for a scene of your own.

Measured off the scene and the assets it references:

  ThorlabsTable   bbox x=[-0.1762, 0.7238]  y=[-0.3793, 0.3793]  z=[-0.7947, 0]
                  -> work surface sits exactly at z = 0, footprint 0.90 x 0.76
  DexCube         0.06 across, scaled 0.5 in the GUI scene -> a 3 cm cube
  Poses           robot and table both at the origin (the scene authors no transform on either),
                  cube at (0.20, 0, 0.015), bin at (0.18, -0.18, 0)

Those poses are worth dwelling on, because they fix the frame everything else is written in: the
robot sits at the origin and the work is at +X, which is exactly where the stock SO_ARM101_CFG
already rests the arm. Keep that frame and the arm faces its work on the first frame for free.

A warning attached to that, because it is the trap this file fell into and cost the most to find.
An asset's prim origin is not its geometric centre. ThorlabsTable's footprint centre is offset
(0.274, 0) from its own origin, so "table at the origin" does NOT mean "table centred on the
robot" -- it means the robot stands 0.18 m in from the near edge, which is a sensible mount. An
earlier version of this file assumed origin == centre, concluded the robot was buried mid-table,
translated the table to "fix" it, re-laid the props around the new centre (rotating the work area
to -Y), and then added a base-rotation override so the arm would face the props it had just moved.
Four edits, none of which errored, all undoing a problem that never existed. The bounding box
recorded a few lines above disproved the assumption the whole time.

So: measure the asset, then reason from the measurement, not from what the placement looks like it
implies. If a scene seems to need fixing before it can be translated, suspect the reading first.

Every pose here is the authored one, and both props keep the authored asset or its measured
dimensions, so a viewer watching the training stream sees the scene that was built, not a
rearrangement of it. Three things still had to be decided, and each has a reason the scene file
could not carry:

  * The cube became a distribution instead of a pose. A scene file holds one arrangement; a policy
    handed one pose memorises one reach. The authored (0.20, 0) is now the centre of a patch, sized
    by what the arm can reach and by keeping clear of the bin. Bounds are in EventCfg.
  * The bin is rebuilt from primitives at small_KLT's measured footprint instead of referencing it.
    Two reasons, and the second is the interesting one. First, the asset carries a DomeLight, which
    would be cloned into every parallel environment. Second, the crate is a shallow tray: 0.0585
    tall at the scene's 0.4 scale, with a floor about 2 mm thick. Drop a 3 cm cube into it and the
    cube settles at z = 0.0170, against 0.0150 for the same cube on the bare table -- "in the bin"
    is 2 mm different from "on the table", and both are under the 0.025 that object_is_lifted calls
    "lifted". That threshold also gates object_goal_distance, so referencing the asset as-is would
    drive 36 of the 37 total reward weight to zero at the exact moment the cube lands in the bin.
    Nothing errors; the run simply never learns to finish. The rebuild keeps the footprint and the
    colour and raises the floor to 20 mm.
  * Reward thresholds derived from geometry, e.g. "lifted" at 0.025 m because a 3 cm cube resting
    on the table has its centre at 0.015 m. Swap the cube for a taller object and that number
    silently stops meaning "lifted" -- it would already be true at rest.

Also worth knowing when comparing against the GUI: the scene places small_KLT at z = 0, and that
asset's origin is its centre, so in the authored file the bin sits 29.3 mm into the tabletop -- the
same origin-is-not-the-centre trap as the table, in a place where it is genuinely hard to see. It
does not look wrong in a render, because the sunken half is hidden by the table; it just reads as a
shallower crate. Here the bin is placed on the surface instead.

Inherited: rewards, terminations and observations come from the lift task's mdp package, so this
file defines a scene and a goal, not new learning code.

Scope: everything here is local to this task. Nothing shared is modified -- the robot config is
used exactly as it ships, so the stock reach and lift tasks are untouched.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils

# Re-exports isaaclab.envs.mdp plus the lift task's own rewards/terminations
import isaac_so_arm101.tasks.lift.mdp as mdp
from isaac_so_arm101.robots import SO_ARM101_CFG
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Layout constants -- every number is either measured or derived from a measurement
##

# Table and robot both sit at the origin, as the scene authored them. The asset's footprint
# centre is offset (0.274, 0) from its own origin, so the tabletop runs x = -0.176 .. 0.724,
# y = +-0.379: the arm is mounted 0.18 m in from the near edge and works down the table's length.
TABLE_POS = (0.0, 0.0, 0.0)

# Centre of the tabletop in world coordinates. Used by the collision slab below.
TABLE_TOP_XY = (0.274, 0.0)
TABLE_TOP_T = 0.02

CUBE_SIZE = 0.03  # DexCube is 0.06 and the GUI scene scaled it by 0.5

# The cube keeps its authored pose (0.20, 0), which is also, exactly, where the stock lift task
# puts its cube -- good evidence the scene was built from that task. A scene file can only hold one
# arrangement, though, and a policy handed one pose memorises one reach, so the authored pose
# becomes the centre of a patch and EventCfg jitters within it.
CUBE_INIT = (0.20, 0.0, CUBE_SIZE / 2)
CUBE_JITTER_X = 0.04
CUBE_JITTER_Y = 0.06

# Bin: authored pose, and an outer footprint matching small_KLT at the scene's 0.4 scale
# (measured: 0.079 x 0.119 x 0.059).
BIN_XY = (0.18, -0.18)
BIN_WALL_T = 0.006
BIN_INNER_HX = 0.0335   # -> outer 0.079 across X
BIN_INNER_HY = 0.0535   # -> outer 0.119 across Y
BIN_OUT_X = 2 * (BIN_INNER_HX + BIN_WALL_T)
BIN_OUT_Y = 2 * (BIN_INNER_HY + BIN_WALL_T)
BIN_WALL_H = 0.045

# The one dimension that does NOT come from the asset, and the reason the bin is rebuilt from
# primitives rather than referenced. small_KLT is 0.0585 tall at this scale and its floor is barely
# 2 mm thick: dropping a 3 cm cube into it, sat flat on the table, settles the cube at z = 0.0170
# (measured) against 0.0150 for the same cube on the bare table. Both are under the 0.025 that
# object_is_lifted calls "lifted", and that threshold also gates object_goal_distance, so 36 of the
# 37 total reward weight would fall to zero at the exact moment the cube lands in the bin --
# silently, with nothing in the logs but a policy that never finishes the task.
# A 20 mm floor puts a resting cube at 0.035 and keeps the reward alive.
BIN_FLOOR_T = 0.020

# Where the cube ends up once it is sitting on the bin floor.
GOAL_Z = BIN_FLOOR_T + CUBE_SIZE / 2

# Sampled from small_KLT's own diffuse texture (FOF_Map_Magenta_Box_D.png): mean (130, 84, 118),
# dominant (141, 88, 131). The rebuilt bin is plain where the real crate is ribbed, but at the
# distance the training view is watched from, colour is what tells you it is the same bin.
COLOUR_BIN_OUT = (0.44, 0.25, 0.40)
COLOUR_BIN_IN = (0.50, 0.31, 0.46)


def _wall(name: str, dx: float, dy: float, sx: float, sy: float) -> AssetBaseCfg:
    """One static black bin wall, positioned relative to the bin centre.

    Prim paths are flat: Isaac Lab spawns straight onto the given path and will
    not create intermediate Xforms, so a nested ".../Bin/Wall" would fail with
    "Unable to find source prim path".
    """
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/BinWall{name}",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(BIN_XY[0] + dx, BIN_XY[1] + dy, BIN_FLOOR_T + BIN_WALL_H / 2)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(sx, sy, BIN_WALL_H),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLOUR_BIN_OUT, roughness=0.6),
        ),
    )


##
# Scene
##


@configclass
class PnPSceneCfg(InteractiveSceneCfg):
    """Table with a bin on it, plus the robot and the cube it has to move."""

    # Stock config, unmodified: it rests the arm stretched along +X, which is where the scene
    # author put the work -- so the arm faces the bin and the cube from the first frame with no
    # override needed.
    robot: ArticulationCfg = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # White table with a dark frame -- exactly as the GUI authored it, at the origin.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )

    # ThorlabsTable's collision surface sits ~1.55 cm below the top you can see:
    # a cube dropped at z=0.015 settles at z=-0.0005, i.e. half sunk into what
    # looks like solid tabletop. Static props hide the problem because they are
    # simply placed, but anything rigid falls through the visible surface.
    # This invisible slab gives physics a top face exactly at z = 0.
    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTopCollider",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(TABLE_TOP_XY[0], TABLE_TOP_XY[1], -TABLE_TOP_T / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(0.90, 0.7586, TABLE_TOP_T),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
    )

    # Bin floor, white, sitting on the table surface.
    bin_floor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BinFloor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(BIN_XY[0], BIN_XY[1], BIN_FLOOR_T / 2)),
        spawn=sim_utils.CuboidCfg(
            size=(BIN_OUT_X, BIN_OUT_Y, BIN_FLOOR_T),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLOUR_BIN_IN, roughness=0.5),
        ),
    )

    bin_wall_px = _wall("PX", +(BIN_INNER_HX + BIN_WALL_T / 2), 0.0, BIN_WALL_T, BIN_OUT_Y)
    bin_wall_nx = _wall("NX", -(BIN_INNER_HX + BIN_WALL_T / 2), 0.0, BIN_WALL_T, BIN_OUT_Y)
    bin_wall_py = _wall("PY", 0.0, +(BIN_INNER_HY + BIN_WALL_T / 2), BIN_OUT_X, BIN_WALL_T)
    bin_wall_ny = _wall("NY", 0.0, -(BIN_INNER_HY + BIN_WALL_T / 2), BIN_OUT_X, BIN_WALL_T)

    # The cube the scene actually references, at the scale the scene actually used, so what the
    # training stream shows is the object from the scene rather than a look-alike.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_INIT, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # Nothing else is overridden: appearance and surface properties are DexCube's own,
            # which is the point of referencing the asset instead of rebuilding it.
        ),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.01, 0.0, -0.09]),
            ),
        ],
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # One light only. The GUI scene had three (two DistantLights plus the one
    # baked into the KLT asset), which would stack up per environment.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.85, 0.85), intensity=3000.0),
    )


##
# MDP
##


@configclass
class CommandsCfg:
    """Where the cube has to end up.

    This is the whole trick for turning "lift the cube" into "put it in the bin":
    the lift task already rewards driving the object to a commanded pose, so
    pinning that command inside the bin makes the existing reward mean exactly
    what we want. Poses are in the robot root frame, which here coincides with
    the environment frame (the robot sits at the origin, unrotated).
    """

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_link",
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(BIN_XY[0], BIN_XY[0]),
            pos_y=(BIN_XY[1], BIN_XY[1]),
            pos_z=(GOAL_Z, GOAL_Z + 0.01),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        open_command_expr={"gripper": 0.5},
        close_command_expr={"gripper": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # The cube starts somewhere in a 0.08 x 0.12 patch centred on its authored pose, so the policy
    # has to look at where it is instead of memorising one reach. The bounds come from two limits:
    # centres run x = 0.16 .. 0.24, y = -0.06 .. 0.06, which keeps every start between 0.16 and
    # 0.25 m from the base -- inside the range the scene itself uses -- and leaves 4.5 cm between
    # the cube and the near bin wall.
    # Widening this is the first thing to check if the cube ever spawns inside the bin or out of
    # reach -- the lift task's own +-0.1/0.2 would do both here.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-CUBE_JITTER_X, CUBE_JITTER_X),
                "y": (-CUBE_JITTER_Y, CUBE_JITTER_Y),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.05}, weight=1.0)

    # Cube centre rests at 0.015 on the table, so 0.025 means genuinely lifted.
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.025}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.025, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.025, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Table surface is z=0, so anything below -0.05 has gone over the edge.
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    # Success: the cube is in the bin. Unused by the lift task, exactly what we need.
    object_in_bin = DoneTerm(
        func=mdp.object_reached_goal,
        params={"command_name": "object_pose", "threshold": 0.035},
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment
##


@configclass
class SoArm101PnPEnvCfg(ManagerBasedRLEnvCfg):
    """SO-101 puts a cube into a bin on the table."""

    scene: PnPSceneCfg = PnPSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        # Longer than the lift task's 5 s: pick *and* place needs the extra time.
        self.episode_length_s = 8.0
        # Looks down the -Y working direction, roughly where the GUI camera was.
        self.viewer.eye = (0.9, 0.9, 0.7)
        self.viewer.lookat = (0.05, -0.18, 0.05)

        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class SoArm101PnPEnvCfg_PLAY(SoArm101PnPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
