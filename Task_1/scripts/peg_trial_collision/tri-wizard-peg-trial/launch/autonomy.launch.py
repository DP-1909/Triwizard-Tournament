"""Vision + full autonomy.

    ros2 launch peg_trial autonomy.launch.py                    # backend from task.yaml
    ros2 launch peg_trial autonomy.launch.py backend:=dry_run    # no robot, test logic only
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    backend = LaunchConfiguration("backend")
    log_dir = LaunchConfiguration("log_dir")
    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="", description="ros2_control | omx_services | dry_run"),
        DeclareLaunchArgument("log_dir", default_value="logs"),
        Node(package="peg_trial", executable="vision_node", name="peg_vision", output="screen"),
        # give the cameras + vision 3 s to produce stable detections first
        TimerAction(period=3.0, actions=[
            Node(package="peg_trial", executable="autonomy_node", name="peg_autonomy", output="screen",
                 parameters=[{"backend": backend, "log_dir": log_dir}]),
        ]),
    ])
