"""SO-101 pick-and-place: drop a red cube into a bin standing on the table.

Translated by hand from a scene authored in the Isaac Sim GUI (so101_pnp.usd), and written to be
read as an example of that translation. A scene file carries geometry; a task also needs intent,
so the numbers below fall into three kinds, and it is worth knowing which is which before copying
this file for a scene of your own.

Measured off the scene and the assets it references:

  ThorlabsTable   bbox x=[-0.1762, 0.7238]  y=[-0.3793, 0.3793]  z=[-0.7947, 0]
                  -> work surface sits exactly at z = 0, footprint 0.90 x 0.76
  DexCube         0.06 across, scaled 0.5 in the GUI scene -> a 3 cm cube
  Bin, cube, robot poses  read from the scene, then adjusted as described below

Decided, because the scene cannot express it:

  * Table placement. The GUI put the table at the origin, which buries the robot in the middle of
    it. The table is 0.90 x 0.76 while the arm works in about a 0.25 m square, so it is positioned
    to centre the span that is actually in play rather than to sit flush with anything.
  * Rest pose. The stock SO_ARM101_CFG rests with the arm stretched along +X, which here points
    away from both the cube and the bin. A quarter turn of the base has it face the work area from
    the first frame. The sign of that turn is not derivable from the scene and the first guess was
    wrong -- check it against the simulated body position, not the USD transform, which stays at
    the authored pose no matter how the joints move.
  * Two props rebuilt from primitives instead of referenced: DexCube is blue and this task wants
    red; small_KLT is grey and ships a DomeLight inside it, which would be cloned into every
    parallel environment and wash out the scene.
  * Reward thresholds derived from geometry, e.g. "lifted" at 0.025 m because a 3 cm cube resting
    on the table has its centre at 0.015 m. Swap the cube for a taller object and that number
    silently stops meaning "lifted" -- it would already be true at rest.

Inherited: rewards, terminations and observations come from the lift task's mdp package, so this
file defines a scene and a goal, not new learning code.

Scope: everything here is local to this task. The robot rest pose is overridden on this task's own
scene config, so the stock reach and lift tasks are untouched.
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

# The table is much larger (0.90 x 0.76) than the area the arm actually works in, so it is placed
# to centre that area rather than to sit flush with the robot: everything in play spans from the
# robot at y = 0 to the far edge of the bin near y = -0.25, and centring that span on the table
# leaves roughly equal margins front and back instead of a crowded corner and an empty acre.
TABLE_POS = (-0.274, -0.125, 0.0)

# Centre of the tabletop once shifted: the asset's own footprint centre is
# (0.274, 0.0), so it lands at (0.0, -0.125). Used by the collision slab below.
TABLE_TOP_XY = (0.0, -0.125)
TABLE_TOP_T = 0.02

CUBE_SIZE = 0.03  # DexCube is 0.06 and the GUI scene scaled it by 0.5
# Front-left of the robot, clear of the bin: the cube is what the arm reaches for, so it must not
# start where the cube is supposed to end up.
CUBE_INIT = (-0.16, -0.12, CUBE_SIZE / 2)

# Bin: outer 0.16 x 0.13, walls 8 mm, floor 20 mm.
# The floor is deliberately thick: a cube resting inside sits at
# z = 0.020 + 0.015 = 0.035, which clears the 0.025 "is lifted" threshold, so
# the goal reward stays alive once the cube is actually in the bin.
# Directly in front of the robot: the arm faces -Y, so keeping the bin on x = 0 puts the drop
# target straight ahead instead of off to one side.
BIN_XY = (0.0, -0.22)
BIN_FLOOR_T = 0.020
BIN_WALL_T = 0.008
BIN_WALL_H = 0.055
BIN_INNER_HX = 0.072
BIN_INNER_HY = 0.057

# Where the cube ends up once it is sitting on the bin floor.
GOAL_Z = BIN_FLOOR_T + CUBE_SIZE / 2

# Base rotation that turns the arm from its stock +X rest pose toward the work area at -Y.
# Sign verified against the simulated gripper position: -1.5708 swings it to +Y, the wrong way.
REST_PAN = 1.5708

COLOUR_CUBE = (0.85, 0.10, 0.10)   # red
COLOUR_BIN_OUT = (0.04, 0.04, 0.04)  # black outside
COLOUR_BIN_IN = (0.95, 0.95, 0.95)   # white inside


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

    # Rest pose, overridden here rather than in SO_ARM101_CFG: the shared config starts the arm
    # stretched along +X, which in this scene points away from both the cube and the bin, so every
    # reset shows the arm facing the wrong way before it swings around. A quarter turn of the base
    # has it face the work area from the first frame. Keeping the override local leaves the stock
    # reach and lift tasks exactly as they were.
    robot: ArticulationCfg = SO_ARM101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={**SO_ARM101_CFG.init_state.joint_pos, "shoulder_pan": REST_PAN},
            joint_vel={".*": 0.0},
        ),
    )

    # White table with a dark frame -- kept as the GUI authored it, only moved.
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
            size=(0.16, 0.13, BIN_FLOOR_T),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLOUR_BIN_IN, roughness=0.5),
        ),
    )

    bin_wall_px = _wall("PX", +(BIN_INNER_HX + BIN_WALL_T / 2), 0.0, BIN_WALL_T, 0.114)
    bin_wall_nx = _wall("NX", -(BIN_INNER_HX + BIN_WALL_T / 2), 0.0, BIN_WALL_T, 0.114)
    bin_wall_py = _wall("PY", 0.0, +(BIN_INNER_HY + BIN_WALL_T / 2), 0.16, BIN_WALL_T)
    bin_wall_ny = _wall("NY", 0.0, -(BIN_INNER_HY + BIN_WALL_T / 2), 0.16, BIN_WALL_T)

    # Red cube, built from a primitive so the colour and size are exact.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_INIT, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
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
            # A small gripper needs grip: stock friction lets the cube squirt out.
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLOUR_CUBE, roughness=0.5),
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

    # Kept tight: the lift task jitters by +-0.1/0.2, which here would drop the
    # cube off the table edge or straight into the bin.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (0.0, 0.0)},
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
    """SO-101 puts a red cube into a bin on the table."""

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
