"""Seed joint_state_publisher_gui once from current arm feedback."""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class SeedGuiJointState(Node):
    def __init__(self):
        super().__init__('seed_gui_joint_state')
        self.feedback = None
        self.feedback_at = 0.0
        self.gui_seen = False
        self.started = time.monotonic()
        self.sent = 0
        self.done = False
        self.publisher = self.create_publisher(
            JointState, 'control/gui_initial_joint_states', 10)
        self.create_subscription(
            JointState, 'feedback/teleop_joint_states', self._feedback, 10)
        self.create_subscription(
            JointState, 'control/joint_states', self._gui, 10)
        self.create_timer(0.1, self._tick)

    def _feedback(self, msg):
        if all(name in msg.name for name in ('joint1', 'joint2', 'joint3',
                                             'joint4', 'joint5', 'joint6')):
            self.feedback = msg
            self.feedback_at = time.monotonic()

    def _gui(self, _msg):
        self.gui_seen = True

    def _tick(self):
        if (self.gui_seen and self.feedback is not None and
                time.monotonic() - self.feedback_at < 0.25 and
                self.publisher.get_subscription_count() > 0):
            self.publisher.publish(self.feedback)
            self.sent += 1
            if self.sent >= 3:
                self.get_logger().info('Seeded GUI sliders from real arm feedback')
                self.done = True
        elif time.monotonic() - self.started > 20:
            self.get_logger().warn('GUI seeding timed out; control gate remains closed')
            self.done = True


def main():
    rclpy.init()
    node = SeedGuiJointState()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
