"""Offline callback tests: no ROS connection, model loading, or robot commands."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from delta_actions import action_spec


def load_offline_module():
    stubs = {name: ModuleType(name) for name in
             ("rclpy", "rclpy.node", "sensor_msgs", "sensor_msgs.msg", "cv_bridge", "policy")}
    stubs["rclpy.node"].Node = object
    stubs["sensor_msgs.msg"].Image = object
    stubs["sensor_msgs.msg"].JointState = lambda: SimpleNamespace(header=SimpleNamespace())
    stubs["cv_bridge"].CvBridge = object
    stubs["policy"].ACTPolicy = Mock()
    spec = importlib.util.spec_from_file_location(
        "offline_piper_inference", Path(__file__).with_name("inference_piper_act.py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", stubs):
        spec.loader.exec_module(module)
    module.DEVICE = "cpu"
    return module


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_offline_module()

    def artifacts(self):
        config = {
            "action_spec": action_spec(), "policy_class": "ACT", "temporal_agg": False,
            "policy_config": {"state_dim": 7, "num_queries": 50, "camera_names": ["wrist"]},
        }
        stats = {"action_spec": action_spec(), "chunk_size": 50,
                 "qpos_mean": np.zeros(7), "qpos_std": np.ones(7),
                 "action_mean": np.zeros(7), "action_std": np.ones(7)}
        return config, stats

    def test_artifact_contract(self):
        config, stats = self.artifacts()
        self.module.validate_inference_artifacts(config, stats, 10)
        for mutate in (
            lambda c, s: c.update(action_spec={"schema_version": 3}),
            lambda c, s: s.update(action_spec={"schema_version": 3}),
            lambda c, s: s.update(chunk_size=10),
            lambda c, s: s.update(action_std=np.zeros(7)),
            lambda c, s: s.update(qpos_mean=np.full(7, np.nan)),
            lambda c, s: c.update(temporal_agg=True),
        ):
            config, stats = self.artifacts()
            mutate(config, stats)
            with self.assertRaises(ValueError):
                self.module.validate_inference_artifacts(config, stats, 10)
        config, stats = self.artifacts()
        for horizon in (0, 51):
            with self.assertRaises(ValueError):
                self.module.validate_inference_artifacts(config, stats, horizon)

    def node(self):
        node = self.module.PiperACTInference.__new__(self.module.PiperACTInference)
        node.latest_image = np.zeros((2, 3, 3), dtype=np.uint8)
        node.latest_qpos = np.arange(7, dtype=np.float32) / 10
        node.qpos_mean = np.full(7, .1, dtype=np.float32)
        node.qpos_std = np.full(7, .5, dtype=np.float32)
        node.action_mean = np.arange(7, dtype=np.float32) / 100
        node.action_std = np.full(7, .2, dtype=np.float32)
        node.policy_config = {"num_queries": 50, "state_dim": 7}
        node.current_actions = None
        node.chunk_start_qpos = None
        node.action_index = 0
        node.inference_count = 0
        node.command_pub = Mock()
        node.get_logger = Mock(return_value=Mock())
        node.get_clock = Mock(return_value=Mock())
        return node

    def test_fixed_reference_denormalization_and_requery(self):
        node = self.node()
        prediction = torch.arange(350, dtype=torch.float32).reshape(1, 50, 7) / 100
        node.policy = Mock(return_value=prediction)
        reference = node.latest_qpos.copy()
        with patch("builtins.print"):
            for k in range(self.module.EXEC_HORIZON):
                if k:
                    node.latest_qpos = reference + 10 + k
                node.control_callback()
                expected = reference + prediction[0, k].numpy() * node.action_std + node.action_mean
                msg = node.command_pub.publish.call_args.args[0]
                np.testing.assert_allclose(msg.position, expected, atol=1e-6)
                self.assertEqual(msg.name, self.module.EXPECTED_JOINT_NAMES)
            self.assertEqual(node.policy.call_count, 1)
            np.testing.assert_array_equal(node.chunk_start_qpos, reference)
            # Once the executed prefix ends, a new chunk gets a new reference.
            next_reference = node.latest_qpos.copy()
            node.control_callback()
        self.assertEqual(node.policy.call_count, 2)
        expected = next_reference + prediction[0, 0].numpy() * node.action_std + node.action_mean
        np.testing.assert_allclose(node.command_pub.publish.call_args.args[0].position, expected, atol=1e-6)
        q_input = node.policy.call_args.args[0].numpy()[0]
        np.testing.assert_allclose(q_input, (next_reference-node.qpos_mean)/node.qpos_std)

    def test_invalid_prediction_is_not_published(self):
        for prediction in (torch.zeros(1, 49, 7), torch.full((1, 50, 7), float("nan"))):
            node = self.node()
            node.policy = Mock(return_value=prediction)
            node.control_callback()
            node.command_pub.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
