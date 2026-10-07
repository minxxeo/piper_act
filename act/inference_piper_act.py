import os
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

from policy import ACTPolicy


# ============================================================
# Paths
# ============================================================

CKPT_DIR = os.environ.get(
    "PIPER_ACT_CKPT_DIR",
    str(Path(__file__).resolve().parent.parent / "checkpoints"),
)

CKPT_PATH = os.path.join(
    CKPT_DIR,
    "policy_best.ckpt" if Path(CKPT_DIR, "policy_best.ckpt").is_file() else "best.ckpt"
)

STATS_PATH = os.path.join(
    CKPT_DIR,
    "dataset_stats.pkl"
)

CONFIG_PATH = os.path.join(
    CKPT_DIR,
    "training_config.json"
)


# ============================================================
# ROS topics
# ============================================================

IMAGE_TOPIC = "/camera/camera/color/image_raw"

JOINT_TOPIC = "/follower/feedback/joint_states"
COMMAND_TOPIC = "/follower/control/move_j"


# ============================================================
# Settings
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Demonstration data was collected at approximately 30 Hz.
CONTROL_HZ = 30.0

# ACT itself was trained with num_queries=50.
# Do NOT change the model/checkpoint horizon.
#
# At inference time, execute only the first 30 actions from each
# predicted 50-step chunk, then query ACT again.
EXEC_HORIZON = 30

# Add 5 mm to the magnitude of the ACT-predicted gripper width.
GRIPPER_OFFSET = 0.005


# ============================================================
# Video recording
# ============================================================

VIDEO_DIR = str(Path(__file__).resolve().parent.parent / "inference_videos")

# D405 영상의 실제 publish FPS에 맞춰 사용.
# 현재 30 Hz를 기준으로 저장.
VIDEO_FPS = 30.0


EXPECTED_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper",
]


# ============================================================
# ROS2 ACT Inference Node
# ============================================================

class PiperACTInference(Node):

    def __init__(self):

        super().__init__("piper_act_inference")

        print("=" * 70)
        print("PiPER ACT ROS2 Inference")
        print("=" * 70)

        print(f"\n[DEVICE] {DEVICE}")

        # ----------------------------------------------------
        # 1. Load training configuration
        # ----------------------------------------------------

        with open(CONFIG_PATH, "r") as f:
            training_config = json.load(f)

        self.config = training_config["config"]
        self.policy_config = self.config["policy_config"]

        print("\n[CONFIG]")
        print(
            f"  state_dim      : "
            f"{self.policy_config['state_dim']}"
        )
        print(
            f"  num_queries    : "
            f"{self.policy_config['num_queries']}"
        )
        print(
            f"  cameras        : "
            f"{self.policy_config['camera_names']}"
        )
        print(
            f"  temporal_agg   : "
            f"{self.config['temporal_agg']}"
        )
        print(
            f"  exec_horizon   : "
            f"{EXEC_HORIZON}"
        )
        print(
            f"  gripper_offset : "
            f"{GRIPPER_OFFSET:.3f} m "
            f"({GRIPPER_OFFSET * 1000:.1f} mm)"
        )

        # ----------------------------------------------------
        # 2. Load normalization statistics
        # ----------------------------------------------------

        with open(STATS_PATH, "rb") as f:
            stats = pickle.load(f)

        self.qpos_mean = np.asarray(
            stats["qpos_mean"],
            dtype=np.float32
        )

        self.qpos_std = np.asarray(
            stats["qpos_std"],
            dtype=np.float32
        )

        self.action_mean = np.asarray(
            stats["action_mean"],
            dtype=np.float32
        )

        self.action_std = np.asarray(
            stats["action_std"],
            dtype=np.float32
        )

        print("\n[STATS] loaded")

        # ----------------------------------------------------
        # 3. Create ACT
        # ----------------------------------------------------

        print("\n[MODEL] Creating ACTPolicy...")

        self.policy = ACTPolicy(
            self.policy_config
        )

        print("\n[CHECKPOINT] Loading:")
        print(f"  {CKPT_PATH}")

        state_dict = torch.load(
            CKPT_PATH,
            map_location=DEVICE
        )

        status = self.policy.load_state_dict(
            state_dict
        )

        print(f"[CHECKPOINT] {status}")

        self.policy.to(DEVICE)
        self.policy.eval()

        print("[MODEL] ACT policy ready")

        # ----------------------------------------------------
        # 4. ROS data storage
        # ----------------------------------------------------

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_qpos = None

        self.image_count = 0
        self.joint_count = 0
        self.inference_count = 0

        self.last_image_time = None
        self.last_joint_time = None

        # ACT chunk currently being executed.
        # ACT predicts 50 actions, but only the first
        # EXEC_HORIZON (=30) actions are executed.
        self.current_actions = None
        self.action_index = 0

        # ----------------------------------------------------
        # 5. Video recording setup
        # ----------------------------------------------------

        os.makedirs(
            VIDEO_DIR,
            exist_ok=True
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.video_path = os.path.join(
            VIDEO_DIR,
            f"inference_{timestamp}.mp4"
        )

        self.video_writer = None

        print("\n[VIDEO]")
        print(
            f"  output : {self.video_path}"
        )
        print(
            f"  fps    : {VIDEO_FPS:.1f}"
        )

        # ----------------------------------------------------
        # 6. Subscribers
        # ----------------------------------------------------

        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10,
        )

        self.joint_sub = self.create_subscription(
            JointState,
            JOINT_TOPIC,
            self.joint_callback,
            10,
        )

        # ----------------------------------------------------
        # 7. Command publisher + 30 Hz control timer
        # ----------------------------------------------------

        self.command_pub = self.create_publisher(
            JointState,
            COMMAND_TOPIC,
            1,
        )

        timer_period = 1.0 / CONTROL_HZ

        self.timer = self.create_timer(
            timer_period,
            self.control_callback,
        )

        print("\n[ROS]")
        print(f"  image topic   : {IMAGE_TOPIC}")
        print(f"  joint topic   : {JOINT_TOPIC}")
        print(f"  command topic : {COMMAND_TOPIC}")
        print(
            f"  control       : "
            f"{CONTROL_HZ:.1f} Hz"
        )

        print(
            "\nWaiting for image + joint state...\n"
        )

    # ========================================================
    # Image callback
    # ========================================================

    def image_callback(self, msg):

        try:

            # ------------------------------------------------
            # ROS Image -> RGB numpy
            # ------------------------------------------------

            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="rgb8",
            )

            image = np.asarray(image)

            if image.ndim != 3:

                self.get_logger().warning(
                    f"Unexpected image ndim: "
                    f"{image.ndim}"
                )

                return

            if image.shape[2] != 3:

                self.get_logger().warning(
                    f"Unexpected image shape: "
                    f"{image.shape}"
                )

                return

            # ------------------------------------------------
            # Store latest image for ACT
            # ------------------------------------------------

            self.latest_image = image.copy()

            self.image_count += 1
            self.last_image_time = time.time()

            # ------------------------------------------------
            # Start video writer on first camera frame
            # ------------------------------------------------

            if self.video_writer is None:

                height, width = image.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                self.video_writer = cv2.VideoWriter(
                    self.video_path,
                    fourcc,
                    VIDEO_FPS,
                    (width, height),
                )

                if not self.video_writer.isOpened():

                    self.get_logger().error(
                        "Failed to open video writer: "
                        f"{self.video_path}"
                    )

                    self.video_writer = None

                else:

                    print(
                        "\n[VIDEO] Recording started"
                    )
                    print(
                        f"  resolution : "
                        f"{width}x{height}"
                    )
                    print(
                        f"  fps        : "
                        f"{VIDEO_FPS:.1f}"
                    )
                    print(
                        f"  file       : "
                        f"{self.video_path}\n"
                    )

            # ------------------------------------------------
            # Write current camera frame
            #
            # ACT uses RGB.
            # OpenCV VideoWriter expects BGR.
            # ------------------------------------------------

            if self.video_writer is not None:

                frame_bgr = cv2.cvtColor(
                    image,
                    cv2.COLOR_RGB2BGR
                )

                self.video_writer.write(
                    frame_bgr
                )

        except Exception as e:

            self.get_logger().error(
                f"Image callback failed: {e}"
            )

    # ========================================================
    # Joint callback
    # ========================================================

    def joint_callback(self, msg):

        try:

            names = list(msg.name)
            positions = list(msg.position)

            # ------------------------------------------------
            # Reorder by joint name so that the state matches
            # the order used during training:
            #
            # joint1 ... joint6, gripper
            # ------------------------------------------------

            joint_map = dict(
                zip(names, positions)
            )

            missing = [
                name
                for name in EXPECTED_JOINT_NAMES
                if name not in joint_map
            ]

            if missing:

                self.get_logger().warning(
                    f"Missing joints: {missing}"
                )

                return

            qpos = np.array(
                [
                    joint_map[name]
                    for name
                    in EXPECTED_JOINT_NAMES
                ],
                dtype=np.float32,
            )

            if qpos.shape != (7,):

                self.get_logger().warning(
                    f"Unexpected qpos shape: "
                    f"{qpos.shape}"
                )

                return

            if not np.all(np.isfinite(qpos)):

                self.get_logger().warning(
                    "qpos contains NaN or Inf"
                )

                return

            self.latest_qpos = qpos

            self.joint_count += 1
            self.last_joint_time = time.time()

        except Exception as e:

            self.get_logger().error(
                f"Joint callback failed: {e}"
            )

    # ========================================================
    # ACT inference
    # ========================================================

    def control_callback(self):

        if (
            self.latest_image is None
            or self.latest_qpos is None
        ):
            return

        # ----------------------------------------------------
        # Query ACT when previous 30-step execution
        # horizon has been exhausted.
        # ----------------------------------------------------

        if (
            self.current_actions is None
            or self.action_index
            >= len(self.current_actions)
        ):

            image_np = self.latest_image.copy()
            qpos_np = self.latest_qpos.copy()

            # ------------------------------------------------
            # Normalize qpos
            # ------------------------------------------------

            qpos_normalized = (
                (qpos_np - self.qpos_mean)
                / self.qpos_std
            )

            qpos_tensor = (
                torch.from_numpy(
                    qpos_normalized
                )
                .float()
                .unsqueeze(0)
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Prepare image
            # ------------------------------------------------

            image_np = np.ascontiguousarray(
                image_np
            )

            image_tensor = (
                torch.from_numpy(image_np)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            # (B, K, C, H, W)
            # = (1, 1, 3, 480, 848)

            image_tensor = (
                image_tensor
                .unsqueeze(0)
                .unsqueeze(0)
                .to(DEVICE)
            )

            # ------------------------------------------------
            # ACT inference
            # ------------------------------------------------

            try:

                with torch.inference_mode():

                    pred_normalized = self.policy(
                        qpos_tensor,
                        image_tensor,
                    )

            except Exception as e:

                self.get_logger().error(
                    f"ACT inference failed: {e}"
                )

                return

            expected_shape = (
                1,
                self.policy_config[
                    "num_queries"
                ],
                self.policy_config[
                    "state_dim"
                ],
            )

            if (
                tuple(pred_normalized.shape)
                != expected_shape
            ):

                self.get_logger().error(
                    "Unexpected ACT output shape: "
                    f"{tuple(pred_normalized.shape)}"
                )

                return

            pred_normalized_np = (
                pred_normalized
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Denormalize ACT actions
            #
            # Full ACT output = (50, 7)
            # ------------------------------------------------

            predicted_actions = (
                pred_normalized_np
                * self.action_std
                + self.action_mean
            ).astype(np.float32)

            if not np.all(
                np.isfinite(predicted_actions)
            ):

                self.get_logger().error(
                    "ACT output contains NaN or Inf"
                )

                self.current_actions = None
                return

            # ------------------------------------------------
            # Execution horizon
            #
            # ACT predicts 50 steps.
            # Execute only first 30 steps.
            # Then observe again and re-query ACT.
            # ------------------------------------------------

            self.current_actions = (
                predicted_actions[
                    :EXEC_HORIZON
                ].copy()
            )

            self.action_index = 0
            self.inference_count += 1

            print(
                f"\n[ACT #{self.inference_count}] "
                f"predicted="
                f"{predicted_actions.shape}, "
                f"execute="
                f"{self.current_actions.shape}"
            )

        # ----------------------------------------------------
        # Execute one predicted action every 1/30 s
        # ----------------------------------------------------

        action = (
            self.current_actions[
                self.action_index
            ].copy()
        )

        # ----------------------------------------------------
        # Gripper opening correction
        #
        # ACT predicts gripper width.
        # PiPER uses abs(gripper_position).
        #
        # Add 5 mm to predicted opening magnitude.
        # ----------------------------------------------------

        raw_gripper = float(
            action[6]
        )

        if action[6] < 0:

            action[6] -= (
                GRIPPER_OFFSET
            )

        else:

            action[6] += (
                GRIPPER_OFFSET
            )

        corrected_gripper = float(
            action[6]
        )

        # ----------------------------------------------------
        # Publish JointState command
        # ----------------------------------------------------

        msg = JointState()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.name = (
            EXPECTED_JOINT_NAMES.copy()
        )

        msg.position = [
            float(x)
            for x in action
        ]

        # Empty effort -> PiPER uses configured
        # gripper_default_effort.
        msg.velocity = []
        msg.effort = []

        self.command_pub.publish(
            msg
        )

        print(
            f"[CMD "
            f"{self.action_index + 1:02d}/"
            f"{len(self.current_actions):02d}] "
            f"gripper "
            f"{raw_gripper * 1000:+.2f} mm "
            f"-> "
            f"{corrected_gripper * 1000:+.2f} mm | "
            f"{np.round(action[:6], 5)}"
        )

        self.action_index += 1

    # ========================================================
    # Release video
    # ========================================================

    def close_video(self):

        if self.video_writer is not None:

            self.video_writer.release()
            self.video_writer = None

            print("\n[VIDEO] Recording stopped")
            print(
                f"[VIDEO] Saved: "
                f"{self.video_path}"
            )


# ============================================================
# Entry point
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = PiperACTInference()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        print("\n[STOP] Ctrl+C")

    finally:

        # Save and close MP4 before destroying node.
        node.close_video()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        print(
            "[STOP] ACT inference stopped."
        )


if __name__ == "__main__":
    main()