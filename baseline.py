"""Original deterministic CPU baseline; no learned weights or network calls."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import List, Tuple

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class BaselineConfig:
    # Pixel parameters target the official 2048x2048 images.
    disk_threshold: int = 12
    limb_margin: int = 32
    background_kernel: int = 61
    contrast_threshold: float = 0.14
    grow_ratio: float = 0.60
    min_area: int = 64
    max_area_fraction: float = 0.02
    min_elongation: float = 1.5

    def __post_init__(self):
        for name in ("disk_threshold", "limb_margin", "background_kernel", "min_area"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(name + " must be an integer")
        if not 1 <= self.disk_threshold <= 255:
            raise ValueError("disk_threshold must be in 1..255")
        if self.limb_margin < 0 or self.min_area < 1:
            raise ValueError("limb_margin must be >=0 and min_area >=1")
        if self.background_kernel < 3 or self.background_kernel % 2 != 1:
            raise ValueError("background_kernel must be odd and >=3")
        for name in ("contrast_threshold", "grow_ratio", "max_area_fraction", "min_elongation"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(value):
                raise ValueError(name + " must be finite numeric")
        if not 0 < self.contrast_threshold <= 1 or not 0 < self.grow_ratio <= 1:
            raise ValueError("contrast_threshold and grow_ratio must be in (0,1]")
        if not 0 < self.max_area_fraction <= 1 or self.min_elongation < 1:
            raise ValueError("invalid area fraction or elongation")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Detection:
    instance_id: int
    area: int
    bbox_xyxy: Tuple[int, int, int, int]
    elongation: float
    mean_contrast: float


@dataclass
class SegmentationResult:
    labels: np.ndarray
    detections: Tuple[Detection, ...]
    diagnostics: dict

    def masks(self) -> List[np.ndarray]:
        return [self.labels == item.instance_id for item in self.detections]


def _image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2 or not image.size:
        raise ValueError("image must be a nonempty two-dimensional grayscale array")
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8; do not silently rescale scientific data")
    return np.ascontiguousarray(image)


def solar_disk_mask(image: np.ndarray, config: BaselineConfig = BaselineConfig()) -> np.ndarray:
    """Largest bright region, with interior holes filled and limb excluded.

    Filling includes dark filaments inside the disk rather than erasing them
    from the search region. No fitted circle or instrument metadata is used.
    """
    image = _image(image)
    components, count = ndimage.label(image >= config.disk_threshold)
    if not count:
        return np.zeros(image.shape, dtype=bool)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    disk = ndimage.binary_fill_holes(components == int(np.argmax(sizes)))
    if config.limb_margin:
        # Explicit zero padding also gives correct borders for a full-frame disk.
        padded = np.pad(disk.astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        disk = distance > config.limb_margin
    return disk


def segment_image(image: np.ndarray, config: BaselineConfig = BaselineConfig()) -> SegmentationResult:
    """Background-normalized blackhat with seed/weak-region hysteresis.

    There is no binary closing/dilation that intentionally joins instances.
    Weak pixels connected to strong seeds retain dim filament sections.
    Area/shape filters remove tiny fragments, round spots, and grossly merged
    regions. They cannot guarantee biological instance separation.
    """
    image = _image(image)
    disk = solar_disk_mask(image, config)
    labels_out = np.zeros(image.shape, dtype=np.int32)
    disk_area = int(np.count_nonzero(disk))
    diagnostics = {"disk_area": disk_area, "candidate_count": 0,
                   "rejected_small": 0, "rejected_large": 0,
                   "rejected_round": 0, "kept": 0}
    if not disk_area:
        return SegmentationResult(labels_out, (), diagnostics)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (config.background_kernel, config.background_kernel))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel).astype(np.float32)
    contrast = np.maximum(background - image.astype(np.float32), 0)
    contrast /= np.maximum(background, 1.0)
    strong = disk & (contrast >= config.contrast_threshold)
    weak = disk & (contrast >= config.contrast_threshold * config.grow_ratio)
    structure = np.ones((3, 3), dtype=bool)
    candidates = ndimage.binary_propagation(strong, structure=structure, mask=weak)
    raw_labels, count = ndimage.label(candidates, structure=structure)
    diagnostics["candidate_count"] = int(count)
    areas = np.bincount(raw_labels.ravel())
    detections = []
    for raw_id, roi in enumerate(ndimage.find_objects(raw_labels), start=1):
        if roi is None:
            continue
        area = int(areas[raw_id])
        if area < config.min_area:
            diagnostics["rejected_small"] += 1
            continue
        if area > config.max_area_fraction * disk_area:
            diagnostics["rejected_large"] += 1
            continue
        local = raw_labels[roi] == raw_id
        ys, xs = np.nonzero(local)
        coordinates = np.column_stack((xs, ys)).astype(np.float64)
        coordinates -= coordinates.mean(axis=0)
        covariance = coordinates.T @ coordinates / area
        eigenvalues = np.linalg.eigvalsh(covariance)
        elongation = float(np.sqrt((eigenvalues[-1] + 1) / (eigenvalues[0] + 1)))
        if elongation < config.min_elongation:
            diagnostics["rejected_round"] += 1
            continue
        instance_id = len(detections) + 1
        labels_out[roi][local] = instance_id
        y_slice, x_slice = roi
        detections.append(Detection(
            instance_id=instance_id, area=area,
            bbox_xyxy=(x_slice.start, y_slice.start, x_slice.stop, y_slice.stop),
            elongation=elongation,
            mean_contrast=float(np.mean(contrast[roi][local]))))
    diagnostics["kept"] = len(detections)
    return SegmentationResult(labels_out, tuple(detections), diagnostics)


def predict_instances(image: np.ndarray, config: BaselineConfig = BaselineConfig()) -> List[np.ndarray]:
    return segment_image(image, config).masks()
