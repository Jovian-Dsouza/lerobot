from lerobot.teleoperators.gamepad.rest_pose import REST_POSE_DEGREES, plan_rest_trajectory


def test_trajectory_ends_exactly_at_rest_pose():
    start = dict.fromkeys(REST_POSE_DEGREES, 0.0)
    frames = plan_rest_trajectory(start, REST_POSE_DEGREES, speed_deg_s=60.0, fps=30)
    assert frames[-1] == REST_POSE_DEGREES


def test_trajectory_is_monotonic_and_does_not_mutate_start():
    start = dict.fromkeys(REST_POSE_DEGREES, 0.0)
    snapshot = dict(start)
    frames = plan_rest_trajectory(start, REST_POSE_DEGREES, speed_deg_s=60.0, fps=30)
    assert start == snapshot
    elbow = [f["elbow_flex.pos"] for f in frames]
    assert elbow == sorted(elbow)


def test_duration_scales_with_largest_joint_move():
    start = dict(REST_POSE_DEGREES, **{"elbow_flex.pos": REST_POSE_DEGREES["elbow_flex.pos"] - 60.0})
    frames = plan_rest_trajectory(start, REST_POSE_DEGREES, speed_deg_s=60.0, fps=30)
    assert len(frames) == 30  # 60 deg at 60 deg/s = 1 s


def test_already_at_rest_still_sends_a_short_hold():
    frames = plan_rest_trajectory(dict(REST_POSE_DEGREES), REST_POSE_DEGREES, speed_deg_s=60.0, fps=30)
    assert 1 <= len(frames) <= 30
    assert all(f == REST_POSE_DEGREES for f in frames)
