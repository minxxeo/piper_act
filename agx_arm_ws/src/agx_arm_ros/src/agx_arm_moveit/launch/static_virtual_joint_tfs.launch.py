import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch import LaunchDescription
from launch.actions import OpaqueFunction

import xml.etree.ElementTree as ET
from launch_ros.actions import Node

from _moveit_config_builder import build_moveit_config, declare_common_args


def _launch(context):
    moveit_config = build_moveit_config(context)
    robot = ET.fromstring(moveit_config["robot_description_semantic"]["robot_description_semantic"])
    nodes = []
    for joint in robot.findall("virtual_joint"):
        if joint.attrib["type"] != "fixed":
            raise ValueError("Static TF publisher requires fixed virtual joints")
        nodes.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            arguments=["0", "0", "0", "0", "0", "0",
                       joint.attrib["parent_frame"], joint.attrib["child_link"]],
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription(declare_common_args() + [OpaqueFunction(function=_launch)])
