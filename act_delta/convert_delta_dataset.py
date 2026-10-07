"""Copy episodes and store action[t] = follower_qpos[t+1] for chunk-relative training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

import h5py
import numpy as np
from delta_actions import follower_absolute_actions, action_spec


ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate directories')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}; choose a new output directory')
    files = sorted(source.glob('episode_*.hdf5'), key=lambda p: int(p.stem.split('_')[-1]))
    if not files:
        raise ValueError('No input episodes')
    spec = action_spec()
    total_bytes = sum(f.stat().st_size for f in files)
    if shutil.disk_usage(destination.parent).free < total_bytes + 1024**3:
        raise RuntimeError('Not enough space for independent dataset copies plus 1 GiB margin')
    # Preflight every source before producing any output.
    for f in files:
        with h5py.File(f, 'r') as h:
            if h.attrs.get('action_representation', 'absolute') != 'absolute':
                raise ValueError(f'{f}: already converted or unknown action representation')
            a, q = h['action'][:], h['observations/qpos'][:]
            follower_absolute_actions(q)
            if a.shape != q.shape or h['observations/images/wrist'].shape[0] != len(q):
                raise ValueError(f'{f}: inconsistent frame count')
            for key in ['observations/qvel', 'timestamps/camera', 'timestamps/leader', 'timestamps/follower']:
                if h[key].shape[0] != len(q):
                    raise ValueError(f'{f}: inconsistent frame count for {key}')
            if 'source_action' in h or 'action_absolute' in h:
                raise ValueError(f'{f}: reserved converted dataset name present')
    destination.mkdir()
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_dataset': str(source), 'destination_dataset': str(destination),
        'action_spec': spec, 'episodes': [],
        'storage': 'Independent copies; observations/images/timestamps unchanged',
    }
    for f in files:
        source_stat = f.stat()
        target = destination / f.name
        temporary = destination / (f.name + '.partial')
        source_hash = hashlib.sha256()
        # Hash exactly the source bytes that are copied.
        with open(f, 'rb') as src, open(temporary, 'xb') as dst:
            for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
                source_hash.update(chunk)
                dst.write(chunk)
        with h5py.File(temporary, 'r+') as h:
            a, q = h['action'][:], h['observations/qpos'][:]
            targets = follower_absolute_actions(q)
            h.move('action', 'source_action')
            h.create_dataset('action', data=targets)
            h.attrs['action_representation'] = spec['stored_action_representation']
            h.attrs['delta_reference'] = spec['delta_reference']
            h.attrs['gripper_mode'] = spec['gripper_mode']
            h.attrs['action_time_offset_frames'] = 1
            h.attrs['action_spec_json'] = json.dumps(spec, sort_keys=True)
            h.attrs['source_sha256'] = source_hash.hexdigest()
            np.testing.assert_array_equal(h['action'][:], q[1:].astype(np.float32))
        if (f.stat().st_size, f.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise RuntimeError(f'Source changed during conversion: {f}')
        os.replace(temporary, target)
        manifest['episodes'].append({'file': f.name, 'frames': len(a),
                                    'source_sha256': source_hash.hexdigest(),
                                    'valid_starts': len(targets)})
        print(f'Converted {f.name}: {len(q)} observation frames, {len(targets)} future targets', flush=True)
    manifest['num_episodes'] = len(files)
    manifest['num_frames'] = sum(e['frames'] for e in manifest['episodes'])
    manifest['num_valid_starts'] = sum(e['valid_starts'] for e in manifest['episodes'])
    (destination / 'delta_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'COMPLETE: {len(files)} episodes, {manifest["num_frames"]} frames -> {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--destination', type=Path, default=ROOT / 'dataset_delta')
    args = parser.parse_args()
    convert(args.source, args.destination)
