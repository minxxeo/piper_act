import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

from _moveit_config_builder import build_moveit_config, declare_common_args


def _launch(context):
    moveit_config = build_moveit_config(context)
    namespace = LaunchConfiguration("namespace").perform(context).strip("/")
    manager = f"/{namespace}/controller_manager" if namespace else "/controller_manager"
    controllers = moveit_config["trajectory_execution"]["moveit_simple_controller_manager"]["controller_names"]
    return [
        Node(package="controller_manager", executable="spawner.py",
             arguments=[name, "--controller-manager", manager,
                        "--controller-manager-timeout", "60"], output="screen")
        for name in ["joint_state_broadcaster", *controllers]
    ]


def generate_launch_description():
    return LaunchDescription(declare_common_args() + [OpaqueFunction(function=_launch)])
