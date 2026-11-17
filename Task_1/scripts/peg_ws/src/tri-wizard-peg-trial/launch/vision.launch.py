"""Vision only (use with teleop, or to check detections in rqt_image_view)."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="peg_trial", executable="vision_node", name="peg_vision", output="screen"),
    ])
