"""Read-only verification of future follower targets and every fixed-reference chunk."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from convert_delta_dataset import sha256, ROOT
from delta_actions import action_spec, absolute_to_delta, delta_to_absolute, make_delta_chunk


def verify(target=ROOT / 'dataset_delta', chunk_size=50):
    target = Path(target)
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    manifest = json.loads((target / 'delta_manifest.json').read_text())
    expected = action_spec()
    assert manifest['action_spec'] == expected, 'Old or incompatible action definition'
    source = Path(manifest['source_dataset'])
    rows = manifest['episodes']
    assert len(rows) == manifest['num_episodes']
    assert {r['file'] for r in rows} == {p.name for p in target.glob('episode_*.hdf5')}
    assert len({r['file'] for r in rows}) == len(rows)
    assert sum(r['frames'] for r in rows) == manifest['num_frames']
    assert sum(r['valid_starts'] for r in rows) == manifest['num_valid_starts']
    for row in rows:
        src, dst = source / row['file'], target / row['file']
        assert sha256(src) == row['source_sha256'], f'Original source changed: {src}'
        assert not src.samefile(dst)
        with h5py.File(src, 'r') as a, h5py.File(dst, 'r') as d:
            assert json.loads(d.attrs['action_spec_json']) == expected
            assert d.attrs['action_representation'] == expected['stored_action_representation']
            assert d.attrs['action_time_offset_frames'] == 1
            assert d.attrs['source_sha256'] == row['source_sha256']
            q = a['observations/qpos'][:].astype(np.float32)
            targets = d['action'][:]
            assert q.shape == (row['frames'], 7)
            assert targets.shape == (row['valid_starts'], 7) == (len(q) - 1, 7)
            np.testing.assert_array_equal(targets, q[1:])
            np.testing.assert_array_equal(d['source_action'][:], a['action'][:])
            for t in range(len(q) - 1):
                n = min(chunk_size, len(q) - t - 1)
                delta, mask = make_delta_chunk(q, t, chunk_size)
                expected_delta = np.stack([q[t + k + 1] - q[t] for k in range(n)])
                np.testing.assert_array_equal(delta[:n], expected_delta)
                np.testing.assert_array_equal(absolute_to_delta(targets[t:t+n], q[t]), expected_delta)
                np.testing.assert_allclose(delta_to_absolute(delta[:n], q[t]), q[t+1:t+1+n],
                                           rtol=1e-5, atol=1e-7)
                np.testing.assert_array_equal(mask, np.arange(chunk_size) >= n)
                np.testing.assert_array_equal(delta[n:], np.zeros((chunk_size-n, 7), dtype=np.float32))
            for key in ['observations/qpos', 'observations/qvel', 'timestamps/camera', 'timestamps/leader', 'timestamps/follower']:
                np.testing.assert_array_equal(a[key][:], d[key][:])
            for i in [0, len(q)//2, len(q)-1]:
                np.testing.assert_array_equal(a['observations/images/wrist'][i], d['observations/images/wrist'][i])
        print(f'Verified {row["file"]}: source hash, future follower targets, all chunks/masks, states/timestamps, 3 images', flush=True)
    print(f'PASS: {len(rows)} episodes / {manifest["num_frames"]} observation frames / '
          f'{manifest["num_valid_starts"]} valid starts; chunk_size={chunk_size}; sources unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=ROOT / 'dataset_delta')
    parser.add_argument('--chunk-size', type=int, default=50)
    args = parser.parse_args()
    verify(args.target, args.chunk_size)

