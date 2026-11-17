"""Start the Intel RealSense camera (skip this if you use other cameras or Gazebo).

    ros2 launch peg_trial cameras.launch.py serial1:=_123456789

Find serials with:  rs-enumerate-devices | grep Serial   (keep the leading underscore!)
Topics become /cam1/camera/... (check: ros2 topic list).
If yours differ, edit the topic names in config/camera_calib.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    rs = os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
    out = [
        DeclareLaunchArgument("serial1", default_value="_332322071822"),
    ]

    # single RealSense camera (cam1)
    out.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(rs), launch_arguments={
        "camera_namespace": "cam1",
        "camera_name": "camera",
        "serial_no": LaunchConfiguration("serial1"),
        "align_depth.enable": "true",
        "rgb_camera.color_profile": "640x480x30",
        "depth_module.depth_profile": "640x480x30",
        "spatial_filter.enable": "true",
        "temporal_filter.enable": "true",
        "hole_filling_filter.enable": "true",
    }.items()))

    return LaunchDescription(out)
