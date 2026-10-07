"""A fixed, bounded motor-readout interface to a tethered NeuroMechFly body.

This is an added engineering actuator interface, not a physiological muscle
model. It consumes two motor drives and has no cue, action label, reward, CPG,
or learned policy input. The mesh is the female NeuroMechFly morphology; using
Male CNS neural outputs does not turn it into a reconstructed male body.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BodyConfig:
    """Fixed mechanical settings; angular settings are in degrees."""

    timestep_s: float = 0.0001
    settling_s: float = 0.1
    motor_drive_scale: float = 1.0
    foreleg_femur_baseline_deg: float = -70.0
    foreleg_tibia_baseline_deg: float = 30.0
    femur_offset_deg: float = -45.0
    coxa_offset_deg: float = 0.0
    actuator_gain: float = 45.0
    actuator_force_limit: float = 65.0


class ForelegBody:
    """Tethered body with a fixed symmetric drive-to-foreleg map.

    ``step([left, right], dt)`` holds the two motor drives for ``dt`` seconds.
    Drives are nonnegative arbitrary units, normalized by a fixed scale and
    clipped to [0, 1]. Each side changes only its own foreleg Femur/Coxa pitch
    targets. All other active leg joints hold a fixed pose. Physical movement
    is measured from MuJoCo state, separately from actuator target angles.

    End effectors are the Tarsus5 body origins supplied by FlyGym, not inferred
    anatomical claw tips. Coordinates use millimetres and +x forward, +y left,
    +z dorsal in the thorax frame. This is a suspended preparation, not walking.
    """

    LEG_ORDER = ("LF", "LM", "LH", "RF", "RM", "RH")
    FORELEG_INDICES = np.array([0, 3])
    FLYGYM_COMMIT = "c7affce924cb1c6add16619adf83be5c6b223e89"
    FLYGYM_WHEEL_SHA256 = (
        "5db9bb89b7f57e2fda8d716fd8205b0ba7ac9a46e7c194ea6e752e38964f390d"
    )

    def __init__(self, config: BodyConfig | None = None):
        from flygym import Camera, Fly, SingleFlySimulation, get_data_path
        from flygym.arena import Tethered
        from flygym.preprogrammed import all_leg_dofs, get_preprogrammed_pose
        from flygym.state import KinematicPose

        self.config = config or BodyConfig()
        if self.config.timestep_s <= 0 or self.config.motor_drive_scale <= 0:
            raise ValueError("timestep and motor-drive scale must be positive")
        self.joint_names = tuple(all_leg_dofs)
        pose = dict(get_preprogrammed_pose("stretch").joint_pos)
        for side in "LR":
            pose[f"joint_{side}FFemur"] = np.deg2rad(
                self.config.foreleg_femur_baseline_deg
            )
            pose[f"joint_{side}FTibia"] = np.deg2rad(
                self.config.foreleg_tibia_baseline_deg
            )
        self.baseline_targets_rad = np.array([pose[n] for n in self.joint_names])
        self.fly = Fly(
            init_pose=KinematicPose(pose),
            spawn_pos=(0.0, 0.0, 2.0),
            control="position",
            self_collisions="legs",
            floor_collisions="none",
            enable_vision=False,
            enable_olfaction=False,
            enable_adhesion=False,
            actuator_gain=self.config.actuator_gain,
            actuator_forcerange=self.config.actuator_force_limit,
        )
        self.cameras = {
            view: Camera(
                self.fly.model.worldbody,
                camera_name=name,
                play_speed_text=False,
            )
            for view, name in {
                "front": "camera_front",
                "top": "camera_top",
                "oblique": "camera_left_top_zoomout",
            }.items()
        }
        self.sim = SingleFlySimulation(
            fly=self.fly,
            arena=Tethered(),
            cameras=list(self.cameras.values()),
            timestep=self.config.timestep_s,
        )
        self._thorax = self.fly.model.find("body", "Thorax")
        self._head = self.fly.model.find("body", "Head")
        self._baseline_tip_body_mm = np.zeros((6, 3))
        self._time_origin_s = 0.0
        self._raw_drive = np.zeros(2)
        self._bounded_drive = np.zeros(2)
        self._command = self.baseline_targets_rad.copy()
        xml = Path(get_data_path("flygym", "data")) / "mjcf" / (
            "neuromechfly_seqik_kinorder_ypr.xml"
        )
        self.morphology_xml_sha256 = sha256(xml.read_bytes()).hexdigest()
        self.reset()

    def _substeps(self, dt: float) -> int:
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        n = round(dt / self.config.timestep_s)
        if not np.isclose(n * self.config.timestep_s, dt, atol=1e-12, rtol=1e-9):
            raise ValueError("dt must be an integer multiple of timestep_s")
        return n

    def _advance(self, targets: np.ndarray, dt: float) -> None:
        for _ in range(self._substeps(dt)):
            _, _, terminated, truncated, _ = self.sim.step({"joints": targets})
            if terminated or truncated:
                raise RuntimeError("FlyGym terminated the body simulation")
            if not np.all(np.isfinite(self.sim.physics.data.qpos)):
                raise FloatingPointError("Nonfinite MuJoCo position state")

    def reset(self, seed: int = 0) -> dict:
        self.sim.reset(seed=seed)
        self._raw_drive = np.zeros(2)
        self._bounded_drive = np.zeros(2)
        self._command = self.baseline_targets_rad.copy()
        self._advance(self._command, self.config.settling_s)
        self._time_origin_s = self.sim.curr_time
        self._baseline_tip_body_mm = self._tip_positions()[1].copy()
        return self.observe()

    def step(self, motor_drive, dt: float = 0.001) -> dict:
        """Advance from motor readout only; no cue or desired side is accepted."""
        drive = np.asarray(motor_drive, dtype=float)
        if drive.shape != (2,) or not np.all(np.isfinite(drive)):
            raise ValueError("motor_drive must contain two finite values [L, R]")
        if np.any(drive < 0):
            raise ValueError("motor_drive must be nonnegative")
        self._raw_drive = drive.copy()
        self._bounded_drive = np.clip(drive / self.config.motor_drive_scale, 0, 1)
        self._command = self.baseline_targets_rad.copy()
        for i, side in enumerate("LR"):
            for suffix, offset in (
                ("Femur", self.config.femur_offset_deg),
                ("Coxa", self.config.coxa_offset_deg),
            ):
                j = self.joint_names.index(f"joint_{side}F{suffix}")
                self._command[j] += np.deg2rad(offset) * self._bounded_drive[i]
        self._advance(self._command, dt)
        return self.observe()

    def _tip_positions(self):
        obs = self.sim.get_observation()
        world = np.asarray(obs["end_effectors"], dtype=float)
        thorax = self.sim.physics.bind(self._thorax)
        head = self.sim.physics.bind(self._head)
        body_relative = (world - thorax.xpos) @ thorax.xmat.reshape(3, 3)
        head_relative = (world - head.xpos) @ head.xmat.reshape(3, 3)
        return world, body_relative, head_relative

    def observe(self) -> dict:
        """Return measured state and commands as separate arrays."""
        obs = self.sim.get_observation()
        world, body_relative, head_relative = self._tip_positions()
        lift = body_relative[:, 2] - self._baseline_tip_body_mm[:, 2]
        return {
            "time_s": self.sim.curr_time - self._time_origin_s,
            "joint_names": self.joint_names,
            "joint_angles_rad": obs["joints"][0].copy(),
            "joint_velocities_rad_s": obs["joints"][1].copy(),
            "joint_force_flygym": obs["joints"][2].copy(),
            "joint_targets_rad": self._command.copy(),
            "motor_drive_raw_au": self._raw_drive.copy(),
            "motor_drive_bounded": self._bounded_drive.copy(),
            "tip_world_mm": world,
            "tip_body_mm": body_relative,
            "tip_head_mm": head_relative,
            "tip_lift_mm": lift,
            "foreleg_lift_mm": lift[self.FORELEG_INDICES],
            "contact_forces_flygym": obs["contact_forces"].copy(),
            "thorax_world_mm": self.sim.physics.bind(self._thorax).xpos.copy(),
        }

    def render_image(self, view: str = "front", width: int = 960, height: int = 720):
        """Capture physical morphology without adding cue-dependent decoration."""
        if view not in self.cameras:
            raise ValueError(f"Unknown view {view!r}; choose {tuple(self.cameras)}")
        return self.sim.physics.render(
            width=width, height=height, camera_id=self.cameras[view].camera_id
        ).copy()

    def metadata(self) -> dict:
        return {
            "interface": "fixed_bounded_MN_readout_to_joint_position_actuators",
            "physiological_muscle_model": False,
            "cue_or_reward_input": False,
            "additional_cpg": False,
            "body_constraint": "tethered_thorax; other active joints hold fixed pose",
            "morphology": "NeuroMechFly female micro-CT-derived mesh, seqik variant",
            "flygym_version": version("flygym"),
            "flygym_commit": self.FLYGYM_COMMIT,
            "flygym_wheel_sha256": self.FLYGYM_WHEEL_SHA256,
            "morphology_xml_sha256": self.morphology_xml_sha256,
            "mujoco_version": version("mujoco"),
            "license": "Apache-2.0; upstream package retains license and assets",
            "config": asdict(self.config),
            "leg_order": self.LEG_ORDER,
            "joint_names": self.joint_names,
            "baseline_targets_rad": self.baseline_targets_rad.tolist(),
            "coordinate_frame": "+x forward, +y left, +z dorsal, thorax-local",
            "length_units": "mm",
            "angle_units": "rad",
            "force_units": "unchanged FlyGym convention; not independently audited",
            "end_effector_definition": "Tarsus5 body origin, not anatomical claw tip",
        }

    def close(self) -> None:
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
