# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
import time
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadTeleopConfig

if TYPE_CHECKING:
    from .gamepad_utils import InputController

logger = logging.getLogger(__name__)


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}

# Absolute joint targets for SO-10x followers. Degrees for the arm, 0–100 for the gripper.
JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
JOINT_LIMITS_DEGREES = {
    "shoulder_pan.pos": (-160.0, 160.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 100.0),
    "wrist_flex.pos": (-100.0, 100.0),
    "wrist_roll.pos": (-160.0, 160.0),
    "gripper.pos": (0.0, 100.0),
}
JOINT_LIMITS_NORMALIZED = {
    "shoulder_pan.pos": (-100.0, 100.0),
    "shoulder_lift.pos": (-100.0, 100.0),
    "elbow_flex.pos": (-100.0, 100.0),
    "wrist_flex.pos": (-100.0, 100.0),
    "wrist_roll.pos": (-100.0, 100.0),
    "gripper.pos": (0.0, 100.0),
}


def integrate_joint_targets(
    targets: dict[str, float],
    axes: dict[str, float],
    gripper_command: str,
    dt: float,
    joint_speed: float,
    gripper_speed: float,
    limits: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Integrate stick deflection into absolute SO-10x joint targets.

    Mapping matches a typical dual-stick pad: left stick pan/lift, right stick
    wrist_flex/elbow, d-pad left/right wrist_roll, triggers close/open the gripper.
    """
    deltas = {
        "shoulder_pan.pos": axes.get("lx", 0.0) * joint_speed * dt,
        "shoulder_lift.pos": -axes.get("ly", 0.0) * joint_speed * dt,
        "elbow_flex.pos": axes.get("ry", 0.0) * joint_speed * dt,
        "wrist_flex.pos": axes.get("rx", 0.0) * joint_speed * dt,
        "wrist_roll.pos": -axes.get("dpad_x", 0.0) * joint_speed * dt,
        "gripper.pos": 0.0,
    }
    if gripper_command == "open":
        deltas["gripper.pos"] = gripper_speed * dt
    elif gripper_command == "close":
        deltas["gripper.pos"] = -gripper_speed * dt

    updated: dict[str, float] = {}
    for name, delta in deltas.items():
        low, high = limits[name]
        updated[name] = min(high, max(low, targets[name] + delta))
    return updated


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.gamepad: InputController | None = None
        self.joint_targets: dict[str, float] | None = None
        self.joint_limits = dict(JOINT_LIMITS_DEGREES)
        self._joint_clock: float | None = None

        self.hidapi_fallback = config.hidapi_fallback
        if sys.platform == "darwin" and not self.hidapi_fallback:
            logger.warning(
                "On macOS, pygame may not reliably detect input from some controllers. "
                "If you experience issues, set `hidapi_fallback=true`."
            )

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        if self.hidapi_fallback:
            from .gamepad_utils import GamepadControllerHID

            self.gamepad = GamepadControllerHID()
        else:
            from .gamepad_utils import GamepadController

            self.gamepad = GamepadController()
        self.gamepad.start(announce=not self.config.joint_control)
        if self.config.joint_control:
            logger.info(
                "Joint mode: left stick shoulder_pan/shoulder_lift, "
                "right stick wrist_flex/elbow_flex, d-pad left/right wrist_roll, triggers gripper."
            )

    def seed_from_observation(self, observation: dict) -> None:
        """Latch the current arm pose so the first stick input does not jump."""
        if not self.config.joint_control or self.joint_targets is not None:
            return
        missing = [name for name in JOINT_NAMES if name not in observation]
        if missing:
            raise KeyError(f"Cannot seed gamepad joints; observation is missing {missing}")
        self.joint_targets = {name: float(observation[name]) for name in JOINT_NAMES}
        self._joint_clock = time.perf_counter()

    def reset_joint_targets(self) -> None:
        """Drop the latched pose so the next observation reseeds it (e.g. after a rest move)."""
        self.joint_targets = None
        self._joint_clock = None

    def _joint_action(self) -> RobotAction:
        if self.gamepad is None:
            raise DeviceNotConnectedError(f"{self} is not connected. Run `.connect()` first.")
        if self.joint_targets is None:
            raise RuntimeError(
                "Gamepad joint targets are not seeded. Record from the robot observation before get_action()."
            )

        self.gamepad.update()
        now = time.perf_counter()
        previous = self._joint_clock if self._joint_clock is not None else now
        dt = min(0.1, max(0.0, now - previous))
        self._joint_clock = now

        self.joint_targets = integrate_joint_targets(
            self.joint_targets,
            self.gamepad.get_axis_state(),
            self.gamepad.gripper_command() if self.config.use_gripper else "stay",
            dt,
            self.config.joint_speed,
            self.config.gripper_speed,
            self.joint_limits,
        )
        return dict(self.joint_targets)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self.config.joint_control:
            return self._joint_action()

        if self.gamepad is None:
            raise DeviceNotConnectedError(f"{self} is not connected. Run `.connect()` first.")

        # Update the controller to get fresh inputs
        self.gamepad.update()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict: RobotAction = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # Default gripper action is to stay
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update gamepad state to get fresh inputs
        self.gamepad.update()

        # Check if intervention is active
        is_intervention = self.gamepad.should_intervene()

        # Get episode end status
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    @property
    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    @property
    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass
