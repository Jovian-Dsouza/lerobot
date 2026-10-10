#!/usr/bin/env python

"""Ease an SO-10x follower into its folded rest pose before each recorded episode."""

import logging
import math
import time

from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

# Folded rest pose (degrees, gripper 0–100), read off the so_follower arm after
# folding it by hand into its collision-free resting position.
REST_POSE_DEGREES = {
    "shoulder_pan.pos": -2.7,
    "shoulder_lift.pos": -86.8,
    "elbow_flex.pos": 99.4,
    "wrist_flex.pos": 4.0,
    "wrist_roll.pos": -96.9,
    "gripper.pos": 0.8,
}
MIN_MOVE_S = 0.5


def plan_rest_trajectory(
    start: dict[str, float], target: dict[str, float], speed_deg_s: float, fps: int
) -> list[dict[str, float]]:
    """Linear joint-space frames from `start` to `target`, paced by the largest joint move."""
    largest = max(abs(target[name] - start[name]) for name in target)
    duration = max(MIN_MOVE_S, largest / speed_deg_s)
    steps = max(1, math.ceil(duration * fps))
    return [
        {name: start[name] + (target[name] - start[name]) * (i / steps) for name in target}
        for i in range(1, steps + 1)
    ]


def move_to_rest(robot, speed_deg_s: float, fps: int) -> None:
    """Drive the follower to REST_POSE_DEGREES. Blocks until the move has been sent."""
    obs = robot.get_observation()
    missing = [name for name in REST_POSE_DEGREES if name not in obs]
    if missing:
        raise KeyError(f"Cannot move to rest pose; observation is missing {missing}")
    start = {name: float(obs[name]) for name in REST_POSE_DEGREES}
    frames = plan_rest_trajectory(start, REST_POSE_DEGREES, speed_deg_s, fps)
    logger.info(f"Moving to rest pose over {len(frames) / fps:.1f} s")
    period = 1.0 / fps
    for frame in frames:
        tick = time.perf_counter()
        robot.send_action(frame)
        precise_sleep(max(0.0, period - (time.perf_counter() - tick)))
