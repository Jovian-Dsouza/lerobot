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

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _hidapi_available, _pygame_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..utils import TeleopEvents

if TYPE_CHECKING or _pygame_available:
    import pygame
else:
    pygame = None  # type: ignore[assignment]

if TYPE_CHECKING or _hidapi_available:
    import hid
else:
    hid = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GamepadLayout:
    """Raw pygame indices for the shoulder buttons and gripper controls."""

    lb_button: int
    rb_button: int
    # Analog trigger axes (rest at -1, +1 fully pressed). None when triggers are buttons.
    lt_axis: int | None
    rt_axis: int | None
    # Digital trigger buttons used by pads that report triggers as buttons.
    close_button: int | None
    open_button: int | None
    # D-pad left/right buttons. None when the d-pad is reported as hat 0.
    dpad_left_button: int | None
    dpad_right_button: int | None


# SDL GameController ordering, used by pygame 2 for Xbox/PlayStation pads:
# A,B,X,Y,Back,Guide,Start,LS,RS,LB(9),RB(10),DPad(11-14); triggers on axes 4/5.
SDL_STANDARD_LAYOUT = GamepadLayout(
    lb_button=9,
    rb_button=10,
    lt_axis=4,
    rt_axis=5,
    close_button=None,
    open_button=None,
    dpad_left_button=13,
    dpad_right_button=14,
)
# DirectInput-style pads (e.g. Logitech): LB/RB on 4/5, triggers as buttons 6/7, d-pad on hat 0.
LEGACY_LAYOUT = GamepadLayout(
    lb_button=4,
    rb_button=5,
    lt_axis=None,
    rt_axis=None,
    close_button=6,
    open_button=7,
    dpad_left_button=None,
    dpad_right_button=None,
)

SDL_STANDARD_MIN_BUTTONS = 15
SDL_STANDARD_MIN_AXES = 6
TRIGGER_PRESSED_THRESHOLD = 0.5


def resolve_layout(num_buttons: int, num_axes: int) -> GamepadLayout:
    """Pick the button/axis layout from what the controller reports."""
    if num_buttons >= SDL_STANDARD_MIN_BUTTONS and num_axes >= SDL_STANDARD_MIN_AXES:
        return SDL_STANDARD_LAYOUT
    return LEGACY_LAYOUT


def trigger_gripper_command(lt: float, rt: float) -> tuple[bool, bool]:
    """Return (close, open) from analog trigger values: LT closes, RT opens."""
    return lt > TRIGGER_PRESSED_THRESHOLD, rt > TRIGGER_PRESSED_THRESHOLD


class InputController:
    """Base class for input controllers that generate motion deltas."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        """
        Initialize the controller.

        Args:
            x_step_size: Base movement step size in meters
            y_step_size: Base movement step size in meters
            z_step_size: Base movement step size in meters
        """
        self.x_step_size = x_step_size
        self.y_step_size = y_step_size
        self.z_step_size = z_step_size
        self.running = True
        self.episode_end_status = None  # None, "success", or "failure"
        self.intervention_flag = False
        self.open_gripper_command = False
        self.close_gripper_command = False

    def start(self):
        """Start the controller and initialize resources."""
        pass

    def stop(self):
        """Stop the controller and release resources."""
        pass

    def get_deltas(self):
        """Get the current movement deltas (dx, dy, dz) in meters."""
        return 0.0, 0.0, 0.0

    def get_axis_state(self) -> dict[str, float]:
        """Normalized sticks and bumpers for joint teleop.

        Keys are ``lx``, ``ly``, ``rx``, ``ry``, ``dpad_x`` in [-1, 1] and ``lb``, ``rb`` in [0, 1].
        """
        return {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "dpad_x": 0.0, "lb": 0.0, "rb": 0.0}

    def update(self):
        """Update controller state - call this once per frame."""
        pass

    def __enter__(self):
        """Support for use in 'with' statements."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are released when exiting 'with' block."""
        self.stop()

    def get_episode_end_status(self):
        """
        Get the current episode end status.

        Returns:
            None if episode should continue, "success" or "failure" otherwise
        """
        status = self.episode_end_status
        self.episode_end_status = None  # Reset after reading
        return status

    def should_intervene(self):
        """Return True if intervention flag was set."""
        return self.intervention_flag

    def gripper_command(self):
        """Return the current gripper command."""
        if self.open_gripper_command == self.close_gripper_command:
            return "stay"
        elif self.open_gripper_command:
            return "open"
        elif self.close_gripper_command:
            return "close"


class KeyboardController(InputController):
    """Generate motion deltas from keyboard input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0):
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.key_states = {
            "forward_x": False,
            "backward_x": False,
            "forward_y": False,
            "backward_y": False,
            "forward_z": False,
            "backward_z": False,
            "quit": False,
            "success": False,
            "failure": False,
        }
        self.listener = None

    def start(self):
        """Start the keyboard listener."""
        if not pynput_can_capture():
            logging.warning(
                "Keyboard control is unavailable in this environment. pynput cannot capture keys "
                "on Wayland or headless machines, or on macOS without Accessibility / Input "
                "Monitoring permission. Keyboard motion will be inactive."
            )
            self.running = False
            return

        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = True
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = True
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = True
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = True
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = True
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = True
                elif key == keyboard.Key.esc:
                    self.key_states["quit"] = True
                    self.running = False
                    return False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = True
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = True
                    self.episode_end_status = TeleopEvents.FAILURE
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.up:
                    self.key_states["forward_x"] = False
                elif key == keyboard.Key.down:
                    self.key_states["backward_x"] = False
                elif key == keyboard.Key.left:
                    self.key_states["forward_y"] = False
                elif key == keyboard.Key.right:
                    self.key_states["backward_y"] = False
                elif key == keyboard.Key.shift:
                    self.key_states["backward_z"] = False
                elif key == keyboard.Key.shift_r:
                    self.key_states["forward_z"] = False
                elif key == keyboard.Key.enter:
                    self.key_states["success"] = False
                elif key == keyboard.Key.backspace:
                    self.key_states["failure"] = False
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

        print("Keyboard controls:")
        print("  Arrow keys: Move in X-Y plane")
        print("  Shift and Shift_R: Move in Z axis")
        print("  Enter: End episode with SUCCESS")
        print("  Backspace: End episode with FAILURE")
        print("  ESC: Exit")

    def stop(self):
        """Stop the keyboard listener."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()

    def get_deltas(self):
        """Get the current movement deltas from keyboard state."""
        delta_x = delta_y = delta_z = 0.0

        if self.key_states["forward_x"]:
            delta_x += self.x_step_size
        if self.key_states["backward_x"]:
            delta_x -= self.x_step_size
        if self.key_states["forward_y"]:
            delta_y += self.y_step_size
        if self.key_states["backward_y"]:
            delta_y -= self.y_step_size
        if self.key_states["forward_z"]:
            delta_z += self.z_step_size
        if self.key_states["backward_z"]:
            delta_z -= self.z_step_size

        return delta_x, delta_y, delta_z


class GamepadController(InputController):
    """Generate motion deltas from gamepad input."""

    def __init__(self, x_step_size=1.0, y_step_size=1.0, z_step_size=1.0, deadzone=0.1):
        require_package("pygame", extra="gamepad")
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.joystick = None
        self.layout = LEGACY_LAYOUT
        self.intervention_flag = False

    def start(self, announce: bool = True):
        """Initialize pygame and the gamepad."""
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            logging.error("No gamepad detected. Please connect a gamepad and try again.")
            self.running = False
            return

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.layout = resolve_layout(self.joystick.get_numbuttons(), self.joystick.get_numaxes())
        logging.info(f"Initialized gamepad: {self.joystick.get_name()} ({self.layout})")

        if not announce:
            return

        print("Gamepad controls:")
        print("  Left analog stick: Move in X-Y plane")
        print("  Right analog stick (vertical): Move in Z axis")
        print("  B/Circle button: Exit")
        print("  Y/Triangle button: End episode with SUCCESS")
        print("  A/Cross button: End episode with FAILURE")
        print("  X/Square button: Rerecord episode")

    def stop(self):
        """Clean up pygame resources."""
        if pygame.joystick.get_init():
            if self.joystick:
                self.joystick.quit()
            pygame.joystick.quit()
        pygame.quit()

    def update(self):
        """Process pygame events to get fresh gamepad readings."""
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 3:
                    self.episode_end_status = TeleopEvents.SUCCESS
                # A button (1) for failure
                elif event.button == 1:
                    self.episode_end_status = TeleopEvents.FAILURE
                # X button (0) for rerecord
                elif event.button == 0:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE

                # Digital trigger buttons (legacy pads): close / open gripper
                elif event.button == self.layout.close_button:
                    self.close_gripper_command = True

                elif event.button == self.layout.open_button:
                    self.open_gripper_command = True

            # Reset episode status on button release
            elif event.type == pygame.JOYBUTTONUP:
                if event.button in [0, 2, 3]:
                    self.episode_end_status = None

                elif event.button == self.layout.close_button:
                    self.close_gripper_command = False

                elif event.button == self.layout.open_button:
                    self.open_gripper_command = False

            # RB held flags a human intervention
            self.intervention_flag = bool(self.joystick.get_button(self.layout.rb_button))

        # Analog triggers (SDL standard pads): LT closes, RT opens
        if self.joystick is not None and self.layout.lt_axis is not None:
            try:
                self.close_gripper_command, self.open_gripper_command = trigger_gripper_command(
                    self.joystick.get_axis(self.layout.lt_axis), self.joystick.get_axis(self.layout.rt_axis)
                )
            except pygame.error:
                logging.error("Error reading gamepad triggers. Is it still connected?")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        try:
            # Read joystick axes
            # Left stick X and Y (typically axes 0 and 1)
            y_input = self.joystick.get_axis(0)  # Up/Down (often inverted)
            x_input = self.joystick.get_axis(1)  # Left/Right

            # Right stick Y (typically axis 3 or 4)
            z_input = self.joystick.get_axis(3)  # Up/Down for Z

            # Apply deadzone to avoid drift
            x_input = 0 if abs(x_input) < self.deadzone else x_input
            y_input = 0 if abs(y_input) < self.deadzone else y_input
            z_input = 0 if abs(z_input) < self.deadzone else z_input

            # Calculate deltas (note: may need to invert axes depending on controller)
            delta_x = -x_input * self.x_step_size  # Forward/backward
            delta_y = -y_input * self.y_step_size  # Left/right
            delta_z = -z_input * self.z_step_size  # Up/down

            return delta_x, delta_y, delta_z

        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return 0.0, 0.0, 0.0

    def get_axis_state(self) -> dict[str, float]:
        """Sticks on axes 0/1 (left) and 2/3 (right); d-pad and LB/RB indices come from the layout."""
        zeros = {"lx": 0.0, "ly": 0.0, "rx": 0.0, "ry": 0.0, "dpad_x": 0.0, "lb": 0.0, "rb": 0.0}
        if self.joystick is None:
            return zeros

        def axis(index: int) -> float:
            if index >= self.joystick.get_numaxes():
                return 0.0
            value = float(self.joystick.get_axis(index))
            if abs(value) < self.deadzone:
                return 0.0
            return max(-1.0, min(1.0, value))

        def button(index: int) -> float:
            if index >= self.joystick.get_numbuttons():
                return 0.0
            return 1.0 if self.joystick.get_button(index) else 0.0

        def dpad_x() -> float:
            if self.layout.dpad_left_button is not None:
                return button(self.layout.dpad_right_button) - button(self.layout.dpad_left_button)
            if self.joystick.get_numhats() == 0:
                return 0.0
            return float(self.joystick.get_hat(0)[0])

        try:
            return {
                "lx": axis(0),
                "ly": axis(1),
                "rx": axis(2),
                "ry": axis(3),
                "dpad_x": dpad_x(),
                "lb": button(self.layout.lb_button),
                "rb": button(self.layout.rb_button),
            }
        except pygame.error:
            logging.error("Error reading gamepad. Is it still connected?")
            return zeros


class GamepadControllerHID(InputController):
    """Generate motion deltas from gamepad input using HIDAPI."""

    def __init__(
        self,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        deadzone=0.1,
    ):
        """
        Initialize the HID gamepad controller.

        Args:
            step_size: Base movement step size in meters
            z_scale: Scaling factor for Z-axis movement
            deadzone: Joystick deadzone to prevent drift
        """
        require_package("hidapi", extra="gamepad", import_name="hid")
        super().__init__(x_step_size, y_step_size, z_step_size)
        self.deadzone = deadzone
        self.device = None
        self.device_info = None

        # Movement values (normalized from -1.0 to 1.0)
        self.left_x = 0.0
        self.left_y = 0.0
        self.right_x = 0.0
        self.right_y = 0.0
        self.lb = False
        self.rb = False

        # Button states
        self.buttons = {}

    def find_device(self):
        """Look for the gamepad device by vendor and product ID."""
        devices = hid.enumerate()
        for device in devices:
            device_name = device["product_string"]
            if any(controller in device_name for controller in ["Logitech", "Xbox", "PS4", "PS5"]):
                return device

        logging.error(
            "No gamepad found, check the connection and the product string in HID to add your gamepad"
        )
        return None

    def start(self, announce: bool = True):
        """Connect to the gamepad using HIDAPI."""
        self.device_info = self.find_device()
        if not self.device_info:
            self.running = False
            return

        try:
            logging.info(f"Connecting to gamepad at path: {self.device_info['path']}")
            self.device = hid.device()
            self.device.open_path(self.device_info["path"])
            self.device.set_nonblocking(1)

            manufacturer = self.device.get_manufacturer_string()
            product = self.device.get_product_string()
            logging.info(f"Connected to {manufacturer} {product}")

            if announce:
                logging.info("Gamepad controls (HID mode):")
                logging.info("  Left analog stick: Move in X-Y plane")
                logging.info("  Right analog stick: Move in Z axis (vertical)")
                logging.info("  Button 1/B/Circle: Exit")
                logging.info("  Button 2/A/Cross: End episode with SUCCESS")
                logging.info("  Button 3/X/Square: End episode with FAILURE")

        except OSError as e:
            logging.error(f"Error opening gamepad: {e}")
            logging.error("You might need to run this with sudo/admin privileges on some systems")
            self.running = False

    def stop(self):
        """Close the HID device connection."""
        if self.device:
            self.device.close()
            self.device = None

    def update(self):
        """
        Read and process the latest gamepad data.
        Due to an issue with the HIDAPI, we need to read the read the device several times in order to get a stable reading
        """
        for _ in range(10):
            self._update()

    def _update(self):
        """Read and process the latest gamepad data."""
        if not self.device or not self.running:
            return

        try:
            # Read data from the gamepad
            data = self.device.read(64)
            # Interpret gamepad data - this will vary by controller model
            # These offsets are for the Logitech RumblePad 2
            if data and len(data) >= 8:
                # Normalize joystick values from 0-255 to -1.0-1.0
                self.left_y = (data[1] - 128) / 128.0
                self.left_x = (data[2] - 128) / 128.0
                self.right_x = (data[3] - 128) / 128.0
                self.right_y = (data[4] - 128) / 128.0

                # Apply deadzone
                self.left_y = 0 if abs(self.left_y) < self.deadzone else self.left_y
                self.left_x = 0 if abs(self.left_x) < self.deadzone else self.left_x
                self.right_x = 0 if abs(self.right_x) < self.deadzone else self.right_x
                self.right_y = 0 if abs(self.right_y) < self.deadzone else self.right_y

                # Parse button states (byte 5 in the Logitech RumblePad 2)
                buttons = data[5]

                # data[6] bit 0 is LB and bit 1 is RB on the Logitech report this parser expects.
                raw_shoulders = data[6] if len(data) > 6 else 0
                self.lb = bool(raw_shoulders & 1)
                self.rb = bool(raw_shoulders & 2)
                # Check if RB is pressed then the intervention flag should be set
                self.intervention_flag = self.rb

                # Check if RT (bit 3) / LT (bit 2) is pressed. Other shoulder bits may be set too.
                self.open_gripper_command = bool(data[6] & 8)
                self.close_gripper_command = bool(data[6] & 4)

                # Check if Y/Triangle button (bit 7) is pressed for saving
                # Check if X/Square button (bit 5) is pressed for failure
                # Check if A/Cross button (bit 4) is pressed for rerecording
                if buttons & 1 << 7:
                    self.episode_end_status = TeleopEvents.SUCCESS
                elif buttons & 1 << 5:
                    self.episode_end_status = TeleopEvents.FAILURE
                elif buttons & 1 << 4:
                    self.episode_end_status = TeleopEvents.RERECORD_EPISODE
                else:
                    self.episode_end_status = None

        except OSError as e:
            logging.error(f"Error reading from gamepad: {e}")

    def get_deltas(self):
        """Get the current movement deltas from gamepad state."""
        # Calculate deltas - invert as needed based on controller orientation
        delta_x = -self.left_x * self.x_step_size  # Forward/backward
        delta_y = -self.left_y * self.y_step_size  # Left/right
        delta_z = -self.right_y * self.z_step_size  # Up/down

        return delta_x, delta_y, delta_z

    def get_axis_state(self) -> dict[str, float]:
        def clip(value: float) -> float:
            return max(-1.0, min(1.0, float(value)))

        return {
            "lx": clip(self.left_x),
            "ly": clip(self.left_y),
            "rx": clip(self.right_x),
            "ry": clip(self.right_y),
            "dpad_x": (1.0 if self.rb else 0.0) - (1.0 if self.lb else 0.0),
            "lb": 1.0 if self.lb else 0.0,
            "rb": 1.0 if self.rb else 0.0,
        }
