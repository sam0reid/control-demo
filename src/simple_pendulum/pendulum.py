from __future__ import annotations

import time
import sys

from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer as mjv

import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass
class PendulumMujocoConfiguration:
    """
    Configuration for a simple pendulum in MuJoCo with a single revolute joint, and single actuator.
    """

    base_body_name: str
    tip_body_name: str
    joint_name: str
    actuator_name: str
    xml_path: str

    def get_base_body_id(self, m: mj.MjModel) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, self.base_body_name)

    def get_tip_body_id(self, m: mj.MjModel) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, self.tip_body_name)

    def get_actuator_id(self, m: mj.MjModel) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, self.actuator_name)

    def get_joint_id(self, m: mj.MjModel) -> int:
        return mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, self.joint_name)

    def get_joint_qposadr(self, m: mj.MjModel) -> int:
        return m.jnt_qposadr[self.get_joint_id(m)]

    def get_joint_dofadr(self, m: mj.MjModel) -> int:
        return m.jnt_dofadr[self.get_joint_id(m)]

    @staticmethod
    def default_pendulum() -> PendulumMujocoConfiguration:
        return PendulumMujocoConfiguration(
            base_body_name="base",
            tip_body_name="tip",
            joint_name="hinge",
            actuator_name="hinge_motor",
            xml_path="models/pendulum.xml",
        )


def swing_up_lqr(
    m: mj.MjModel,
    d: mj.MjData,
    p: PendulumMujocoConfiguration,
    lqr_gain: np.ndarray,
    k: float = 0.05,
    transition_height_fraction: float = 0.5,
):
    """
    Swing up the pendulum by controlling the motor torque.
    Uses energy-based swing-up and switches to LQR for stabilization.
    """
    tip_body_id = p.get_tip_body_id(m)
    base_body_id = p.get_base_body_id(m)
    actuator_id = p.get_actuator_id(m)

    length = np.linalg.norm(d.xpos[tip_body_id] - d.xpos[base_body_id])
    mass = m.body_mass[tip_body_id]
    gravity = m.opt.gravity[2]

    # Current state
    angle = d.qpos[p.get_joint_qposadr(m)]
    velocity = d.qvel[p.get_joint_dofadr(m)]

    # Normalize angle to [-pi, pi]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    # Swing-up phase
    height = -np.cos(angle) * length
    if height > transition_height_fraction * length:
        # Switch to LQR control
        d.ctrl[actuator_id] = -lqr_gain @ np.array(
            [np.sign(angle) * (np.abs(angle) - np.pi), velocity]
        )
    else:
        # Energy-based control
        potential_energy = -1 * mass * gravity * length * np.cos(angle)
        kinetic_energy = 0.5 * mass * length**2 * velocity**2
        desired_energy = mass * gravity * length
        energy_difference = kinetic_energy + potential_energy - desired_energy

        d.ctrl[actuator_id] = k * energy_difference * velocity


def lqr(m: mj.MjModel, p: PendulumMujocoConfiguration) -> np.ndarray:
    """
    Compute the LQR gain for the pendulum.
    """

    length = 1
    mass = m.body_mass[p.get_tip_body_id(m)]

    A = np.array([[0, 1], [-m.opt.gravity[2] / length, 0]])
    B = np.array([[0], [1 / (mass * length**2)]])

    # Weightings below determine controller behavior
    Q = np.array([[100, 0], [0, 10]])  # State weighting
    R = np.array([[0.01]])  # Control effort weighting

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P

    return K


def pd_control(
    m: mj.MjModel,
    d: mj.MjData,
    target_angle: float,
    p: PendulumMujocoConfiguration,
    k_p: float = 20,
    k_d: float = 2,
):
    """
    Simple PD controller for the pendulum.
    """

    assert k_p > 0, "Proportional coefficient must be positive"
    assert k_d > 0, "Derivative coefficient must be positive"

    angle = d.qpos[p.get_joint_qposadr(m)]
    velocity = d.qvel[p.get_joint_dofadr(m)]

    error = target_angle - angle
    error_derivative = -velocity

    d.ctrl[p.get_actuator_id(m)] = k_p * error + k_d * error_derivative


def main(
    control_strategy: str = "lqr",
):
    frame_delay = 0.0166667
    pendulum = PendulumMujocoConfiguration.default_pendulum()

    m = mj.MjModel.from_xml_path(pendulum.xml_path)
    d = mj.MjData(m)

    lqr_gain = lqr(m, pendulum)

    angle = np.random.uniform(-np.pi / 2, np.pi / 2)
    d.qpos[pendulum.get_joint_qposadr(m)] = angle
    mj.mj_forward(m, d)

    start = time.time()
    last_frame = start
    with mjv.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
        while viewer.is_running():
            mj.mj_step(m, d)

            if control_strategy == "lqr":
                swing_up_lqr(m, d, pendulum, lqr_gain)
            elif control_strategy == "pd":
                pd_control(m, d, np.pi, pendulum)
            else:
                print(f"Invalid control strategy: {control_strategy}")
                return 1

            if time.time() - last_frame > frame_delay:
                viewer.sync()
                last_frame = time.time()

            while time.time() - start < d.time:
                pass

    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    
    if len(args) == 0: 
        main() 
    else: 
        main(args[0])