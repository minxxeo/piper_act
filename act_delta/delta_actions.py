"""Follower future-state targets relative to a fixed chunk-start observation."""
import numpy as np


def action_spec():
    return {
        "schema_version": 4,
        "action_representation": "follower_chunk_start_delta",
        "stored_action_representation": "next_follower_qpos_absolute",
        "action_source": "observations/qpos",
        "absolute_action_formula": "action[t] = follower_qpos[t + 1]",
        "delta_reference": "fixed_chunk_start_follower_qpos",
        "delta_indices": list(range(7)),
        "gripper_mode": "delta",
        "action_time_offset_frames": 1,
        "chunk_formula": "delta[t, k] = follower_qpos[t + k + 1] - follower_qpos[t]",
        "execution_formula": "absolute_target[k] = chunk_start_follower_qpos + predicted_delta[k]",
        "tail_policy": "Exclude final observation as a start; mask unavailable future targets",
        "units": "Original dataset units; no time division or unit conversion",
    }


def _positions(value, name):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim < 1 or value.shape[-1] != 7 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite with shape (..., 7)")
    return value


def follower_absolute_actions(qpos):
    qpos = _positions(qpos, "qpos")
    if qpos.ndim != 2 or len(qpos) < 2:
        raise ValueError("Need qpos with shape (T, 7), T >= 2")
    return qpos[1:].copy()


def _chunk_reference(values, chunk_start_qpos):
    values = _positions(values, "values")
    reference = _positions(chunk_start_qpos, "chunk_start_qpos")
    # (K, 7) uses (7,); (B, K, 7) uses (B, 7).
    # Reject per-step references to prevent silently using the old definition.
    expected = values.shape[:-2] + (7,) if values.ndim >= 2 else (7,)
    if reference.shape != expected:
        raise ValueError(f"Expected one fixed chunk reference {expected}, got {reference.shape}")
    return values, np.expand_dims(reference, -2) if values.ndim >= 2 else reference


def absolute_to_delta(absolute_targets, chunk_start_qpos):
    targets, reference = _chunk_reference(absolute_targets, chunk_start_qpos)
    return targets - reference


def delta_to_absolute(delta, chunk_start_qpos):
    delta, reference = _chunk_reference(delta, chunk_start_qpos)
    return delta + reference


def make_delta_chunk(qpos, start, chunk_size):
    qpos = _positions(qpos, "qpos")
    if qpos.ndim != 2 or len(qpos) < 2:
        raise ValueError("Need qpos with shape (T, 7), T >= 2")
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(start, (int, np.integer)) or not 0 <= start < len(qpos) - 1:
        raise ValueError("Start must have at least one future observation")
    future = qpos[start + 1:start + 1 + chunk_size]
    delta = np.zeros((chunk_size, 7), dtype=np.float32)
    is_pad = np.ones(chunk_size, dtype=bool)
    delta[:len(future)] = absolute_to_delta(future, qpos[start])
    is_pad[:len(future)] = False
    return delta, is_pad

