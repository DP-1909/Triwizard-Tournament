#!/usr/bin/env bash
# Full autonomous run. Robot bringup + cameras must already be running (see README step 6).
#   ./launch/run_autonomy.sh            -> real robot / Gazebo (backend from config/task.yaml)
#   ./launch/run_autonomy.sh dry_run    -> no robot, just test the logic
set -e
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
source ../../install/setup.bash 2>/dev/null || source install/setup.bash
mkdir -p logs
if [ -n "$1" ]; then
    ros2 launch peg_trial autonomy.launch.py backend:="$1" log_dir:="$(pwd)/logs"
else
    ros2 launch peg_trial autonomy.launch.py log_dir:="$(pwd)/logs"
fi
python3 scripts/evaluate_timing.py --log-dir logs --last 1
