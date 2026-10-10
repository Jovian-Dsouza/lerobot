from lerobot.teleoperators.gamepad.gamepad_utils import (
    LEGACY_LAYOUT,
    SDL_STANDARD_LAYOUT,
    resolve_layout,
    trigger_gripper_command,
)
from lerobot.teleoperators.gamepad.teleop_gamepad import JOINT_LIMITS_DEGREES, integrate_joint_targets

ZERO_TARGETS = dict.fromkeys(JOINT_LIMITS_DEGREES, 0.0) | {"gripper.pos": 50.0}


def _step(axes, gripper="stay"):
    return integrate_joint_targets(ZERO_TARGETS, axes, gripper, 1.0, 10.0, 10.0, JOINT_LIMITS_DEGREES)


def test_xbox_one_s_on_sdl_uses_standard_layout():
    # Xbox One S over pygame/SDL reports 15 buttons and 6 axes.
    layout = resolve_layout(num_buttons=15, num_axes=6)
    assert layout == SDL_STANDARD_LAYOUT
    assert (layout.lb_button, layout.rb_button) == (9, 10)
    assert (layout.lt_axis, layout.rt_axis) == (4, 5)
    assert (layout.dpad_left_button, layout.dpad_right_button) == (13, 14)


def test_small_pads_keep_legacy_layout():
    layout = resolve_layout(num_buttons=12, num_axes=4)
    assert layout == LEGACY_LAYOUT
    assert (layout.lb_button, layout.rb_button) == (4, 5)
    assert layout.lt_axis is None
    assert layout.dpad_left_button is None  # d-pad comes from hat 0


def test_trigger_axes_rest_at_minus_one_and_map_lt_close_rt_open():
    assert trigger_gripper_command(lt=-1.0, rt=-1.0) == (False, False)
    assert trigger_gripper_command(lt=0.99, rt=-1.0) == (True, False)
    assert trigger_gripper_command(lt=-1.0, rt=0.99) == (False, True)


def test_wrist_flex_follows_right_stick():
    assert _step({"rx": 1.0})["wrist_flex.pos"] > 0.0
    assert _step({"rx": -1.0})["wrist_flex.pos"] < 0.0


def test_shoulder_pan_follows_left_stick_x():
    assert _step({"lx": 1.0})["shoulder_pan.pos"] > 0.0
    assert _step({"lx": -1.0})["shoulder_pan.pos"] < 0.0


def test_shoulder_lift_follows_left_stick_y():
    # pygame reports stick up as negative y.
    assert _step({"ly": -1.0})["shoulder_lift.pos"] > 0.0
    assert _step({"ly": 1.0})["shoulder_lift.pos"] < 0.0


def test_elbow_flex_follows_right_stick_y():
    assert _step({"ry": -1.0})["elbow_flex.pos"] < 0.0
    assert _step({"ry": 1.0})["elbow_flex.pos"] > 0.0


def test_dpad_drives_wrist_roll_not_bumpers():
    assert _step({"dpad_x": 1.0})["wrist_roll.pos"] < 0.0
    assert _step({"dpad_x": -1.0})["wrist_roll.pos"] > 0.0
    assert _step({"lb": 1.0, "rb": 0.0})["wrist_roll.pos"] == 0.0


def test_gripper_commands_move_gripper():
    assert _step({}, "open")["gripper.pos"] > 50.0
    assert _step({}, "close")["gripper.pos"] < 50.0
