#!/usr/bin/env python3

import os
import threading
import time
from collections import deque

import h5py
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge


# ============================================================
# Configuration
# ============================================================

CAMERA_TOPIC = "/camera/camera/color/image_raw"
FOLLOWER_TOPIC = "/follower/feedback/joint_states"
LEADER_TOPIC = "/leader/feedback/leader_joint_states"

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

CAMERA_NAME = "wrist"

JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper",
]

# PiPER ≈ 200 Hz
# 약 2초 분량 보관
ROBOT_BUFFER_SIZE = 400

# Camera ↔ Robot 최대 허용 sync error
MAX_SYNC_ERROR = 0.020  # 20 ms

# 카메라 frame이 도착한 직후에는 미래 robot sample이 아직
# callback되지 않았을 수 있으므로 약간 기다렸다가 매칭한다.
SYNC_DELAY = 0.010  # 10 ms


# ============================================================
# Utilities
# ============================================================

def stamp_to_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def extract_joint_data(msg):
    """
    JointState 메시지에서

    qpos:
        joint1~joint6 + gripper position

    qvel:
        joint1~joint6 + gripper velocity

    를 지정된 순서대로 추출한다.
    """

    position_dict = dict(zip(msg.name, msg.position))

    # velocity 배열 길이가 정상적인 경우
    if len(msg.velocity) == len(msg.name):
        velocity_dict = dict(zip(msg.name, msg.velocity))
    else:
        velocity_dict = {}

    try:
        qpos = np.array(
            [position_dict[name] for name in JOINT_NAMES],
            dtype=np.float32,
        )
    except KeyError:
        return None

    qvel = np.array(
        [
            velocity_dict.get(name, 0.0)
            for name in JOINT_NAMES
        ],
        dtype=np.float32,
    )

    return qpos, qvel


# ============================================================
# Collector
# ============================================================

class PiperACTCollector(Node):

    def __init__(self):

        super().__init__("piper_act_collector")

        self.bridge = CvBridge()

        self.lock = threading.Lock()

        # ----------------------------------------------------
        # Robot buffers
        # ----------------------------------------------------

        self.follower_buffer = deque(
            maxlen=ROBOT_BUFFER_SIZE
        )

        self.leader_buffer = deque(
            maxlen=ROBOT_BUFFER_SIZE
        )

        # ----------------------------------------------------
        # Camera pending queue
        #
        # 카메라 frame이 도착하자마자 매칭하지 않고
        # 잠깐 기다렸다가 앞/뒤 robot sample을 모두 확보한다.
        # ----------------------------------------------------

        self.pending_images = deque()

        # ----------------------------------------------------
        # Episode buffer
        # ----------------------------------------------------

        self.images = []

        self.qpos = []
        self.qvel = []

        self.actions = []

        self.camera_timestamps = []
        self.follower_timestamps = []
        self.leader_timestamps = []

        self.recording = False

        # ----------------------------------------------------
        # ROS subscribers
        # ----------------------------------------------------

        self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.camera_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            JointState,
            FOLLOWER_TOPIC,
            self.follower_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            JointState,
            LEADER_TOPIC,
            self.leader_callback,
            qos_profile_sensor_data,
        )

        # Pending camera frames 처리
        self.create_timer(
            0.005,
            self.process_pending_images,
        )

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.get_logger().info(
            "PiPER ACT data collector started"
        )

        self.get_logger().info(
            f"Camera   : {CAMERA_TOPIC}"
        )

        self.get_logger().info(
            f"Follower : {FOLLOWER_TOPIC}"
        )

        self.get_logger().info(
            f"Leader   : {LEADER_TOPIC}"
        )

    # ========================================================
    # Robot callbacks
    # ========================================================

    def follower_callback(self, msg):

        result = extract_joint_data(msg)

        if result is None:
            return

        qpos, qvel = result

        timestamp = stamp_to_sec(
            msg.header.stamp
        )

        with self.lock:

            self.follower_buffer.append(
                (
                    timestamp,
                    qpos,
                    qvel,
                )
            )

    def leader_callback(self, msg):

        result = extract_joint_data(msg)

        if result is None:
            return

        action, _ = result

        timestamp = stamp_to_sec(
            msg.header.stamp
        )

        with self.lock:

            self.leader_buffer.append(
                (
                    timestamp,
                    action,
                )
            )

    # ========================================================
    # Camera callback
    # ========================================================

    def camera_callback(self, msg):

        if not self.recording:
            return

        camera_timestamp = stamp_to_sec(
            msg.header.stamp
        )

        # ROS Image -> RGB NumPy
        image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="rgb8",
        )

        # callback 도착 시간을 monotonic clock으로 기록
        arrival_time = time.monotonic()

        with self.lock:

            self.pending_images.append(
                (
                    arrival_time,
                    camera_timestamp,
                    image.copy(),
                )
            )

    # ========================================================
    # Synchronization
    # ========================================================

    @staticmethod
    def find_nearest_follower(
        buffer,
        target_timestamp,
    ):

        if not buffer:
            return None

        nearest = min(
            buffer,
            key=lambda x: abs(
                x[0] - target_timestamp
            ),
        )

        timestamp, qpos, qvel = nearest

        error = abs(
            timestamp - target_timestamp
        )

        if error > MAX_SYNC_ERROR:
            return None

        return timestamp, qpos, qvel

    @staticmethod
    def find_nearest_leader(
        buffer,
        target_timestamp,
    ):

        if not buffer:
            return None

        nearest = min(
            buffer,
            key=lambda x: abs(
                x[0] - target_timestamp
            ),
        )

        timestamp, action = nearest

        error = abs(
            timestamp - target_timestamp
        )

        if error > MAX_SYNC_ERROR:
            return None

        return timestamp, action

    def process_pending_images(self):

        if not self.recording:
            return

        while True:

            with self.lock:

                if not self.pending_images:
                    return

                arrival_time, camera_time, image = (
                    self.pending_images[0]
                )

                # 카메라가 도착한 후 10ms 대기
                if (
                    time.monotonic() - arrival_time
                    < SYNC_DELAY
                ):
                    return

                self.pending_images.popleft()

                follower = (
                    self.find_nearest_follower(
                        self.follower_buffer,
                        camera_time,
                    )
                )

                leader = (
                    self.find_nearest_leader(
                        self.leader_buffer,
                        camera_time,
                    )
                )

                if (
                    follower is None
                    or leader is None
                ):

                    self.get_logger().warning(
                        "Sync failed. "
                        "Camera frame skipped."
                    )

                    continue

                (
                    follower_time,
                    qpos,
                    qvel,
                ) = follower

                (
                    leader_time,
                    action,
                ) = leader

                # --------------------------------------------
                # ACT timestep 생성
                # --------------------------------------------

                self.images.append(
                    image
                )

                self.qpos.append(
                    qpos.copy()
                )

                self.qvel.append(
                    qvel.copy()
                )

                self.actions.append(
                    action.copy()
                )

                self.camera_timestamps.append(
                    camera_time
                )

                self.follower_timestamps.append(
                    follower_time
                )

                self.leader_timestamps.append(
                    leader_time
                )

                frame_count = len(
                    self.images
                )

                if frame_count % 30 == 0:

                    follower_error = abs(
                        camera_time
                        - follower_time
                    ) * 1000

                    leader_error = abs(
                        camera_time
                        - leader_time
                    ) * 1000

                    self.get_logger().info(
                        f"Frames: {frame_count} | "
                        f"Follower sync: "
                        f"{follower_error:.2f} ms | "
                        f"Leader sync: "
                        f"{leader_error:.2f} ms"
                    )

    # ========================================================
    # Episode control
    # ========================================================

    def start_episode(self):

        with self.lock:

            self.images.clear()

            self.qpos.clear()
            self.qvel.clear()

            self.actions.clear()

            self.camera_timestamps.clear()
            self.follower_timestamps.clear()
            self.leader_timestamps.clear()

            self.pending_images.clear()

            self.recording = True

        print()
        print(
            "========== RECORDING START =========="
        )

    def stop_episode(self):

        with self.lock:
            self.recording = False

        # 마지막 pending callback이 들어오지 않도록
        # 짧게 기다린다.
        time.sleep(0.05)

        print(
            "========== RECORDING STOP =========="
        )

    def discard_episode(self):

        with self.lock:

            self.images.clear()

            self.qpos.clear()
            self.qvel.clear()

            self.actions.clear()

            self.camera_timestamps.clear()
            self.follower_timestamps.clear()
            self.leader_timestamps.clear()

            self.pending_images.clear()

        print("Episode discarded.")

    # ========================================================
    # Episode numbering
    # ========================================================

    def get_next_episode_path(self):

        episode_idx = 0

        while True:

            filename = os.path.join(
                SAVE_DIR,
                f"episode_{episode_idx}.hdf5",
            )

            if not os.path.exists(filename):
                return filename

            episode_idx += 1

    # ========================================================
    # Validation
    # ========================================================

    def validate_episode(
        self,
        images,
        qpos,
        qvel,
        actions,
    ):

        T = len(images)

        if T == 0:
            raise ValueError(
                "Episode contains zero frames."
            )

        if len(qpos) != T:
            raise ValueError(
                "qpos length mismatch."
            )

        if len(qvel) != T:
            raise ValueError(
                "qvel length mismatch."
            )

        if len(actions) != T:
            raise ValueError(
                "action length mismatch."
            )

        if qpos.shape[1] != 7:
            raise ValueError(
                f"Expected qpos dim=7, "
                f"got {qpos.shape}"
            )

        if qvel.shape[1] != 7:
            raise ValueError(
                f"Expected qvel dim=7, "
                f"got {qvel.shape}"
            )

        if actions.shape[1] != 7:
            raise ValueError(
                f"Expected action dim=7, "
                f"got {actions.shape}"
            )

    # ========================================================
    # HDF5
    # ========================================================

    def save_episode(self):

        with self.lock:

            if not self.images:

                print(
                    "No data recorded."
                )

                return

            images = np.stack(
                self.images
            ).astype(np.uint8)

            qpos = np.stack(
                self.qpos
            ).astype(np.float32)

            qvel = np.stack(
                self.qvel
            ).astype(np.float32)

            actions = np.stack(
                self.actions
            ).astype(np.float32)

            camera_ts = np.asarray(
                self.camera_timestamps,
                dtype=np.float64,
            )

            follower_ts = np.asarray(
                self.follower_timestamps,
                dtype=np.float64,
            )

            leader_ts = np.asarray(
                self.leader_timestamps,
                dtype=np.float64,
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        self.validate_episode(
            images,
            qpos,
            qvel,
            actions,
        )

        filename = (
            self.get_next_episode_path()
        )

        # ----------------------------------------------------
        # Write ACT-compatible HDF5
        # ----------------------------------------------------

        with h5py.File(
            filename,
            "w",
        ) as root:

            # ACT에서 real robot 여부 판단
            root.attrs["sim"] = False

            # 우리가 추가하는 metadata
            root.attrs["fps"] = 30
            root.attrs["camera_name"] = CAMERA_NAME
            root.attrs["image_encoding"] = "rgb8"
            root.attrs["state_dim"] = 7

            # ----------------------------------------------
            # observations
            # ----------------------------------------------

            observations = (
                root.create_group(
                    "observations"
                )
            )

            observations.create_dataset(
                "qpos",
                data=qpos,
                dtype=np.float32,
            )

            observations.create_dataset(
                "qvel",
                data=qvel,
                dtype=np.float32,
            )

            images_group = (
                observations.create_group(
                    "images"
                )
            )

            images_group.create_dataset(
                CAMERA_NAME,
                data=images,
                dtype=np.uint8,
            )

            # ----------------------------------------------
            # action
            # ----------------------------------------------

            root.create_dataset(
                "action",
                data=actions,
                dtype=np.float32,
            )

            # ----------------------------------------------
            # Additional timestamps
            # ACT에서는 사용하지 않음.
            # synchronization 검증용.
            # ----------------------------------------------

            timestamps = (
                root.create_group(
                    "timestamps"
                )
            )

            timestamps.create_dataset(
                "camera",
                data=camera_ts,
            )

            timestamps.create_dataset(
                "follower",
                data=follower_ts,
            )

            timestamps.create_dataset(
                "leader",
                data=leader_ts,
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        follower_sync = (
            np.abs(
                camera_ts
                - follower_ts
            )
            * 1000
        )

        leader_sync = (
            np.abs(
                camera_ts
                - leader_ts
            )
            * 1000
        )

        duration = (
            camera_ts[-1]
            - camera_ts[0]
        )

        actual_fps = (
            (len(camera_ts) - 1)
            / duration
            if duration > 0
            else 0
        )

        print()
        print(
            "========================================"
        )
        print(
            " ACT Episode Saved"
        )
        print(
            "========================================"
        )

        print(
            f"File       : {filename}"
        )

        print(
            f"Frames     : {len(images)}"
        )

        print(
            f"Duration   : {duration:.2f} sec"
        )

        print(
            f"FPS        : {actual_fps:.2f}"
        )

        print()

        print(
            f"Images     : {images.shape}"
        )

        print(
            f"qpos       : {qpos.shape}"
        )

        print(
            f"qvel       : {qvel.shape}"
        )

        print(
            f"action     : {actions.shape}"
        )

        print()

        print(
            "Follower sync mean/max : "
            f"{follower_sync.mean():.3f} / "
            f"{follower_sync.max():.3f} ms"
        )

        print(
            "Leader sync mean/max   : "
            f"{leader_sync.mean():.3f} / "
            f"{leader_sync.max():.3f} ms"
        )

        print(
            "========================================"
        )
        print()


# ============================================================
# Terminal UI
# ============================================================

def command_loop(node):

    print()
    print(
        "========================================"
    )
    print(
        " PiPER ACT Data Collector"
    )
    print(
        "========================================"
    )
    print()
    print(
        "ENTER : Start episode"
    )
    print(
        "s     : Stop + Save"
    )
    print(
        "d     : Stop + Discard"
    )
    print(
        "q     : Quit"
    )
    print()

    while rclpy.ok():

        command = (
            input("> ")
            .strip()
            .lower()
        )

        if command == "":

            if node.recording:

                print(
                    "Already recording."
                )

            else:

                node.start_episode()

        elif command == "s":

            if not node.recording:

                print(
                    "Not recording."
                )

                continue

            node.stop_episode()

            node.save_episode()

        elif command == "d":

            node.stop_episode()

            node.discard_episode()

        elif command == "q":

            if node.recording:

                node.stop_episode()

            rclpy.shutdown()

            break


# ============================================================
# Main
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = PiperACTCollector()

    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        daemon=True,
    )

    ros_thread.start()

    try:

        command_loop(node)

    except KeyboardInterrupt:

        pass

    finally:

        if rclpy.ok():
            rclpy.shutdown()

        ros_thread.join(
            timeout=1.0
        )

        node.destroy_node()


if __name__ == "__main__":
    main()
