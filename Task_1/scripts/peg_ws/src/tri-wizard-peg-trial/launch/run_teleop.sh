#!/usr/bin/env bash
# Keyboard teleop. Robot bringup (+ cameras for the 1-9 helper keys) must already be running.
set -e
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
source ../../install/setup.bash 2>/dev/null || source install/setup.bash
ros2 launch peg_trial vision.launch.py > /tmp/peg_vision.log 2>&1 &
VISION_PID=$!
trap "kill $VISION_PID 2>/dev/null" EXIT
sleep 2
ros2 run peg_trial teleop_keyboard
