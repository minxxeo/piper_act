"""Guarded ROS topic bridge between two independently connected Piper arms."""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool

from agx_arm_msgs.msg import AgxArmStatus


ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
MAX_AGE = 0.25
MAX_START_ERROR = math.radians(5)
MAX_TRACKING_ERROR = math.radians(12)
MAX_STEP = math.radians(1)


def positions(msg, require_gripper=False):
    """Return named arm angles and optional gripper width, or reject bad data."""
    if msg is None or len(msg.name) != len(msg.position):
        return None
    values = dict(zip(msg.name, msg.position))
    if len(values) != len(msg.name) or any(n not in values for n in ARM_JOINTS):
        return None
    if any(not math.isfinite(values[n]) for n in ARM_JOINTS):
        return None
    if require_gripper and "gripper" not in values:
        return None
    if "gripper" in values and (not math.isfinite(values["gripper"])
                                or not 0.0 <= values["gripper"] <= 0.1):
        return None
    return values


class PiperTopicTeleop(Node):
    def __init__(self):
        super().__init__("piper_topic_teleop")
        self.leader = None
        self.follower = None
        self.status = None
        self.leader_at = 0.0
        self.follower_at = 0.0
        self.status_at = 0.0
        self.enabled = False
        self.last_target = None
        self.create_subscription(
            JointState, "/leader/feedback/teleop_joint_states",
            self._leader, 1)
        self.create_subscription(
            JointState, "/follower/feedback/joint_states",
            self._follower, 1)
        self.create_subscription(
            AgxArmStatus, "/follower/feedback/arm_status",
            self._status, 1)
        self.publisher = self.create_publisher(
            JointState, "/follower/control/joint_states", 1)
        self.create_service(SetBool, "/piper_teleop/enable", self._enable)
        self.create_timer(0.02, self._tick)
        self.get_logger().info("Topic teleoperation is disarmed")

    def _leader(self, msg):
        self.leader, self.leader_at = msg, time.monotonic()

    def _follower(self, msg):
        self.follower, self.follower_at = msg, time.monotonic()

    def _status(self, msg):
        self.status, self.status_at = msg, time.monotonic()

    def _valid(self):
        now = time.monotonic()
        if (now - self.leader_at > MAX_AGE or
                now - self.follower_at > MAX_AGE or
                now - self.status_at > MAX_AGE):
            return None, None, "Leader or follower feedback is stale"
        lead = positions(self.leader, require_gripper=True)
        follow = positions(self.follower, require_gripper=True)
        if lead is None or follow is None:
            return None, None, "Joint feedback is incomplete or invalid"
        if self.status.ctrl_mode != 1 or self.status.arm_status != 0:
            return None, None, "Follower is not in healthy CAN control mode"
        return lead, follow, ""

    def _enable(self, request, response):
        if not request.data:
            self.enabled = False
            self.last_target = None
            response.success = True
            response.message = "Topic teleoperation disarmed"
            return response
        lead, follow, reason = self._valid()
        if reason:
            response.success = False
            response.message = reason
            return response
        error = max(abs(lead[n] - follow[n]) for n in ARM_JOINTS)
        if error > MAX_START_ERROR:
            response.success = False
            response.message = "Align arms within 5 degrees before enabling"
            return response
        if abs(lead["gripper"] - follow["gripper"]) > 0.02:
            response.success = False
            response.message = "Align grippers within 20 mm before enabling"
            return response
        self.last_target = {n: follow[n] for n in ARM_JOINTS}
        self.enabled = True
        response.success = True
        response.message = "Topic teleoperation enabled"
        return response

    def _tick(self):
        if not self.enabled:
            return
        lead, follow, reason = self._valid()
        if reason:
            self._stop(reason)
            return
        if max(abs(self.last_target[n] - follow[n]) for n in ARM_JOINTS) > MAX_TRACKING_ERROR:
            self._stop("Follower tracking error exceeds 12 degrees")
            return
        target = {}
        for name in ARM_JOINTS:
            delta = lead[name] - self.last_target[name]
            target[name] = self.last_target[name] + max(-MAX_STEP, min(MAX_STEP, delta))
        self.last_target = target
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(ARM_JOINTS)
        msg.position = [target[n] for n in ARM_JOINTS]
        msg.name.append("gripper")
        msg.position.append(lead["gripper"])
        msg.effort = [0.0] * len(ARM_JOINTS) + [1.0]
        self.publisher.publish(msg)

    def _stop(self, reason):
        self.enabled = False
        self.last_target = None
        self.get_logger().error("Teleoperation disarmed: " + reason)


def main(args=None):
    rclpy.init(args=args)
    node = PiperTopicTeleop()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
