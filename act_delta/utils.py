import numpy as np
import torch
import os
import h5py
import json
from pathlib import Path
from delta_actions import action_spec, absolute_to_delta
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size=50):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        if chunk_size < 1 or norm_stats.get('chunk_size') != chunk_size:
            raise ValueError('Normalization chunk_size must match dataset chunk_size')
        if norm_stats.get('action_spec') != action_spec():
            raise ValueError('Normalization must use the current follower chunk definition')
        self.chunk_size = chunk_size
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if json.loads(root.attrs.get('action_spec_json', '{}')) != action_spec():
                raise ValueError(f'Expected future follower targets: {dataset_path}')
            if root['observations/qpos'].shape != (episode_len + 1, 7) or episode_len < 1:
                raise ValueError(f'Expected T observations and T-1 targets: {dataset_path}')
            # Stored action[s] = q[s+1]. All targets share observation q[start_ts].
            action = absolute_to_delta(root['/action'][start_ts:start_ts + self.chunk_size], qpos)
            action_len = len(action)

        self.is_sim = is_sim
        padded_len = self.chunk_size
        padded_action = np.zeros((padded_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(padded_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, episode_ids=None, chunk_size=50):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    all_qpos_data = []
    all_action_data = []
    for episode_idx in (range(num_episodes) if episode_ids is None else episode_ids):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            if json.loads(root.attrs.get('action_spec_json', '{}')) != action_spec():
                raise ValueError(f'Wrong Delta definition: {dataset_path}')
        if qpos.shape != (len(action) + 1, 7) or len(action) < 1:
            raise ValueError(f'Expected T observations and T-1 targets: {dataset_path}')
        np.testing.assert_array_equal(action, qpos[1:].astype(np.float32))
        all_qpos_data.append(torch.from_numpy(qpos[:-1]).float())
        # Every valid (start, horizon) pair, excluding padding and validation episodes.
        for start in range(len(action)):
            chunk = absolute_to_delta(action[start:start + chunk_size], qpos[start])
            all_action_data.append(torch.from_numpy(chunk))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True, unbiased=False)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True, unbiased=False)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos, "chunk_size": chunk_size,
             "action_spec": action_spec(),
             "normalization_source": "all valid training start/horizon pairs; no padding",
             "num_action_samples": len(all_action_data)}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, num_train_episodes=None, chunk_size=50):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    # Reuse the exact split of the completed Absolute experiment, independent of RNG state.
    split = json.loads((Path(__file__).resolve().parent / 'absolute_dataset_split.json').read_text())
    train_indices = np.asarray(split['train_episode_ids'], dtype=np.int64)
    val_indices = np.asarray(split['validation_episode_ids'], dtype=np.int64)
    if (len(train_indices) != num_train_episodes or not len(val_indices)
            or len(set(train_indices) & set(val_indices))
            or sorted(np.concatenate([train_indices, val_indices]).tolist()) != list(range(num_episodes))):
        raise ValueError('Saved Absolute split is incompatible with this task')
    print(f'Train episodes ({len(train_indices)}): {train_indices.tolist()}')
    print(f'Validation episodes ({len(val_indices)}): {val_indices.tolist()}')

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, episode_ids=train_indices, chunk_size=chunk_size)

    # construct dataset and dataloader
    episode_lengths = []
    for episode_idx in range(num_episodes):
        with h5py.File(os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5'), 'r') as root:
            if json.loads(root.attrs.get('action_spec_json', '{}')) != action_spec():
                raise ValueError(f'Wrong Delta data definition in episode {episode_idx}')
            episode_lengths.append(root['/action'].shape[0])
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils



### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
