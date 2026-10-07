"""Regression tests for fixed-reference future follower chunks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from delta_actions import (action_spec, absolute_to_delta, delta_to_absolute,
                           follower_absolute_actions, make_delta_chunk)
from convert_delta_dataset import convert
from verify_delta_dataset import verify
from utils import EpisodicDataset, get_norm_stats


class DeltaTests(unittest.TestCase):
    def trajectory(self, length):
        return (np.arange(length, dtype=np.float32)[:, None] ** 2
                * np.arange(1, 8, dtype=np.float32)[None, :])

    def write_source(self, directory, episode, q):
        with h5py.File(Path(directory) / f'episode_{episode}.hdf5', 'w') as h:
            h.attrs['sim'] = False
            # Deliberately unrelated leader commands, including the gripper.
            h['action'] = np.full_like(q, -999)
            h['observations/qpos'] = q
            h['observations/qvel'] = np.zeros_like(q)
            h['observations/images/wrist'] = np.zeros((len(q), 2, 3, 3), dtype=np.uint8)
            for name in ['camera', 'leader', 'follower']:
                h[f'timestamps/{name}'] = np.arange(len(q), dtype=np.float64)

    def test_fixed_reference_and_all_seven_dimensions(self):
        q = self.trajectory(55)
        delta, mask = make_delta_chunk(q, 2, 50)
        np.testing.assert_array_equal(delta, q[3:53] - q[2])
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(delta_to_absolute(delta, q[2]), q[3:53])
        self.assertNotEqual(delta[1, 0], q[4, 0] - q[3, 0])
        np.testing.assert_array_equal(follower_absolute_actions(q), q[1:])
        batched = np.stack([delta, delta])
        np.testing.assert_array_equal(delta_to_absolute(batched, np.stack([q[2], q[2]])),
                                      np.stack([q[3:53], q[3:53]]))

    def test_tail_and_final_start(self):
        q = self.trajectory(4)
        delta, mask = make_delta_chunk(q, 2, 50)
        np.testing.assert_array_equal(delta[0], q[3] - q[2])
        np.testing.assert_array_equal(mask, np.arange(50) >= 1)
        np.testing.assert_array_equal(delta[1:], np.zeros((49, 7)))
        with self.assertRaises(ValueError):
            make_delta_chunk(q, 3, 50)

    def test_invalid_inputs_and_old_per_step_reference(self):
        for bad in [np.zeros((1, 7)), np.full((2, 7), np.nan), np.zeros((3, 6))]:
            with self.assertRaises(ValueError):
                follower_absolute_actions(bad)
        with self.assertRaises(ValueError):
            absolute_to_delta(np.zeros((50, 7)), np.zeros((50, 7)))
        with self.assertRaises(ValueError):
            make_delta_chunk(self.trajectory(3), 0, 0)

    def test_conversion_loader_statistics_and_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory) / 'source', Path(directory) / 'target'
            src.mkdir()
            q = self.trajectory(5)
            self.write_source(src, 0, q)
            self.write_source(src, 1, self.trajectory(7) * 1000)
            convert(src, dst)
            verify(dst, chunk_size=3)
            stats = get_norm_stats(dst, 2, episode_ids=[0], chunk_size=3)
            expected = np.concatenate([q[t+1:min(t+4, len(q))] - q[t] for t in range(len(q)-1)])
            np.testing.assert_allclose(stats['action_mean'], expected.mean(0))
            np.testing.assert_allclose(stats['action_std'], expected.std(0), rtol=1e-6)
            np.testing.assert_allclose(stats['qpos_mean'], q[:-1].mean(0))
            self.assertEqual(stats['num_action_samples'], len(expected))
            dataset = EpisodicDataset([0, 1], dst, ['wrist'], stats, chunk_size=3)
            for start in range(len(q)-1):
                with patch('numpy.random.choice', return_value=start) as choose:
                    _, observation, actions, mask = dataset[0]
                choose.assert_called_once_with(len(q)-1)
                n = min(3, len(q)-start-1)
                physical = actions.numpy() * stats['action_std'] + stats['action_mean']
                np.testing.assert_allclose(physical[:n], q[start+1:start+1+n]-q[start], atol=1e-5)
                np.testing.assert_allclose(observation.numpy()*stats['qpos_std']+stats['qpos_mean'], q[start], atol=1e-5)
                np.testing.assert_array_equal(mask.numpy(), np.arange(3) >= n)
            with self.assertRaises(ValueError):
                EpisodicDataset([0], dst, ['wrist'], stats, chunk_size=50)
            with h5py.File(dst / 'episode_0.hdf5', 'r+') as h:
                h['source_action'][:] = 12345
            with patch('numpy.random.choice', return_value=1):
                _, _, actions, _ = dataset[0]
            np.testing.assert_allclose(actions.numpy()*stats['action_std']+stats['action_mean'], q[2:5]-q[1], atol=1e-5)
            with self.assertRaises(AssertionError):
                verify(dst, chunk_size=3)

    def test_old_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_source(directory, 0, self.trajectory(3))
            with h5py.File(Path(directory) / 'episode_0.hdf5', 'r+') as h:
                h.attrs['action_spec_json'] = json.dumps({'schema_version': 3})
            with self.assertRaises(ValueError):
                get_norm_stats(directory, 1)


if __name__ == '__main__':
    unittest.main()

