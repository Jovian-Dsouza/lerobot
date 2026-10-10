#!/usr/bin/env python

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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gamepad")
@dataclass
class GamepadTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    # Use hidapi instead of pygame for controllers that pygame cannot detect reliably.
    hidapi_fallback: bool = False
    # When True, sticks integrate into absolute joint targets (`*.pos`) for arms
    # such as SO-10x. When False, `get_action()` keeps the cartesian deltas used by HIL.
    joint_control: bool = False
    # Degrees (or normalized units) per second at full stick deflection.
    joint_speed: float = 45.0
    # Gripper units per second while a trigger is held. Gripper range is 0–100.
    gripper_speed: float = 40.0
    # Joint mode only: ease the follower into the folded rest pose at the start of each
    # reset phase and right before every episode (so episode 0 starts there too).
    rest_on_reset: bool = True
    # Degrees per second for the largest joint during the rest move.
    rest_speed: float = 60.0
