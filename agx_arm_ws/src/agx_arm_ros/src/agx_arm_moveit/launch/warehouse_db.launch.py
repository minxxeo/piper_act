import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from launch_ros.actions import Node

from _moveit_config_builder import declare_common_args


def _launch(context):
    return [Node(
        package="warehouse_ros_mongo", executable="mongo_wrapper_ros.py",
        parameters=[{"warehouse_port": 33829, "warehouse_host": "localhost",
                     "warehouse_plugin": "warehouse_ros_mongo::MongoDatabaseConnection"}],
        output="screen",
    )]


def generate_launch_description():
    return LaunchDescription(declare_common_args() + [OpaqueFunction(function=_launch)])
