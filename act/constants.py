import pathlib

PIPER_TASK_CONFIGS = {
    'piper_wrist': {
        'dataset_dir': str(pathlib.Path(__file__).resolve().parent.parent / 'dataset'),
        'num_episodes': 60,
        'num_train_episodes': 50,
        'episode_len': 502,
        'camera_names': ['wrist'],
        'state_dim': 7,
    },
}

# Nominal camera timestep for episode visualization.
DT = 1 / 30
