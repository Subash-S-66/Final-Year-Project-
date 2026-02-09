from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np


POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21

FEATURES_PER_HAND = 1 + 1 + HAND_LANDMARK_COUNT * 3  # present + handedness + (x,y,z)*21
FEATURES_POSE = 1 + POSE_LANDMARK_COUNT * 4  # pose_present + (x,y,z,visibility)*33
FEATURE_DIM = FEATURES_PER_HAND * 2 + FEATURES_POSE  # 65*2 + 133 = 263


@dataclass
class ExtractorDebug:
    pose_present: bool
    left_hand_present: bool
    right_hand_present: bool
    canonical_mirrored: bool


class LandmarkExtractor:
    """Extract pose + hands landmarks into a fixed-size feature vector per frame.

    Output feature order per frame (float32, length=263):
      - left hand:  present, handedness, 21*(x,y,z)
      - right hand: present, handedness, 21*(x,y,z)
      - pose:       pose_present, 33*(x,y,z,visibility)

    Canonical mirroring (v1):
      - If ONLY left hand is present, we mirror X and swap left/right so the active
        hand lands in the right-hand slot.
      - If both hands are present or right hand is present, we do not mirror.
    """

    def __init__(
        self,
        *,
        enable_canonical_mirroring: bool = True,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.enable_canonical_mirroring = enable_canonical_mirroring

        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._holistic.close()

    def extract(self, image_bgr: np.ndarray) -> tuple[np.ndarray, ExtractorDebug]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid image")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(rgb)

        pose_present = results.pose_landmarks is not None
        left_present = results.left_hand_landmarks is not None
        right_present = results.right_hand_landmarks is not None

        canonical_mirrored = False
        if self.enable_canonical_mirroring and left_present and not right_present:
            canonical_mirrored = True

        left_hand = _hand_landmarks_to_array(results.left_hand_landmarks)
        right_hand = _hand_landmarks_to_array(results.right_hand_landmarks)
        pose = _pose_landmarks_to_array(results.pose_landmarks)

        # Canonical mirroring: if only left hand was found, mirror X and swap slots.
        if canonical_mirrored:
            mirrored = _mirror_hand(left_hand)
            right_hand = mirrored
            left_hand = None
            pose = _mirror_pose(pose)

        # Normalize coordinates using a pose reference point if possible.
        ref = _compute_reference_point(pose)
        scale = _compute_scale(pose)
        left_hand = _normalize_hand(left_hand, ref=ref, scale=scale)
        right_hand = _normalize_hand(right_hand, ref=ref, scale=scale)
        pose = _normalize_pose(pose, ref=ref, scale=scale)

        feature = np.zeros((FEATURE_DIM,), dtype=np.float32)
        offset = 0
        offset = _write_hand_features(feature, offset, hand=left_hand, handedness=-1.0)
        offset = _write_hand_features(feature, offset, hand=right_hand, handedness=+1.0)
        _write_pose_features(feature, offset, pose=pose)

        debug = ExtractorDebug(
            pose_present=pose_present,
            left_hand_present=left_present and not canonical_mirrored,
            right_hand_present=right_present or canonical_mirrored,
            canonical_mirrored=canonical_mirrored,
        )
        return feature, debug


def _hand_landmarks_to_array(landmarks: Any) -> Optional[np.ndarray]:
    if landmarks is None:
        return None
    pts = np.zeros((HAND_LANDMARK_COUNT, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        pts[i, 0] = lm.x
        pts[i, 1] = lm.y
        pts[i, 2] = lm.z
    return pts


def _pose_landmarks_to_array(landmarks: Any) -> Optional[np.ndarray]:
    if landmarks is None:
        return None
    pts = np.zeros((POSE_LANDMARK_COUNT, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks.landmark):
        pts[i, 0] = lm.x
        pts[i, 1] = lm.y
        pts[i, 2] = lm.z
        pts[i, 3] = getattr(lm, "visibility", 0.0)
    return pts


def _mirror_hand(hand: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if hand is None:
        return None
    out = hand.copy()
    out[:, 0] = 1.0 - out[:, 0]
    out[:, 2] = -out[:, 2]
    return out


def _mirror_pose(pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if pose is None:
        return None
    out = pose.copy()
    out[:, 0] = 1.0 - out[:, 0]
    out[:, 2] = -out[:, 2]
    return out


def _compute_reference_point(pose: Optional[np.ndarray]) -> np.ndarray:
    # Prefer mid-hip; fallback to mid-shoulder; fallback to center.
    if pose is None:
        return np.array([0.5, 0.5, 0.0], dtype=np.float32)

    left_hip = pose[23, :3]
    right_hip = pose[24, :3]
    left_shoulder = pose[11, :3]
    right_shoulder = pose[12, :3]

    hip_vis = pose[23, 3] + pose[24, 3]
    shoulder_vis = pose[11, 3] + pose[12, 3]

    if hip_vis > 0.1:
        return ((left_hip + right_hip) / 2.0).astype(np.float32)
    if shoulder_vis > 0.1:
        return ((left_shoulder + right_shoulder) / 2.0).astype(np.float32)
    return np.array([0.5, 0.5, 0.0], dtype=np.float32)


def _compute_scale(pose: Optional[np.ndarray]) -> float:
    # Use shoulder width if available, else hip width, else 1.0.
    if pose is None:
        return 1.0

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    left_shoulder = pose[11, :3]
    right_shoulder = pose[12, :3]
    left_hip = pose[23, :3]
    right_hip = pose[24, :3]

    shoulder_vis = pose[11, 3] + pose[12, 3]
    hip_vis = pose[23, 3] + pose[24, 3]

    if shoulder_vis > 0.1:
        return max(dist(left_shoulder, right_shoulder), 1e-3)
    if hip_vis > 0.1:
        return max(dist(left_hip, right_hip), 1e-3)
    return 1.0


def _normalize_hand(hand: Optional[np.ndarray], *, ref: np.ndarray, scale: float) -> Optional[np.ndarray]:
    if hand is None:
        return None
    out = hand.copy()
    out[:, :3] = (out[:, :3] - ref) / scale
    return out


def _normalize_pose(pose: Optional[np.ndarray], *, ref: np.ndarray, scale: float) -> Optional[np.ndarray]:
    if pose is None:
        return None
    out = pose.copy()
    out[:, :3] = (out[:, :3] - ref) / scale
    return out


def _write_hand_features(dst: np.ndarray, offset: int, *, hand: Optional[np.ndarray], handedness: float) -> int:
    present = 1.0 if hand is not None else 0.0
    dst[offset] = present
    dst[offset + 1] = handedness if present > 0 else 0.0
    offset += 2

    if hand is None:
        offset += HAND_LANDMARK_COUNT * 3
        return offset

    flat = hand.reshape(-1)
    dst[offset : offset + flat.size] = flat
    offset += flat.size
    return offset


def _write_pose_features(dst: np.ndarray, offset: int, *, pose: Optional[np.ndarray]) -> None:
    present = 1.0 if pose is not None else 0.0
    dst[offset] = present
    offset += 1
    if pose is None:
        return
    flat = pose.reshape(-1)
    dst[offset : offset + flat.size] = flat
