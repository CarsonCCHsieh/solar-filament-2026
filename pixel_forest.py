"""Original bounded CPU pixel classifier; no downloads or external services.

Training consumes one uint8 grayscale observation and one binary union target
at a time. Only training annotators may be unioned. Keep evaluation annotators
and their instances separate, and split by exact observation file_name first.
Balanced pixel sampling means forest probabilities are model scores, not
calibrated foreground prevalence. Connected components do not separate touching
filaments. No real-data performance is established by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import zip_longest
import math

import cv2
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class PixelForestConfig:
    n_estimators: int = 64
    max_depth: int = 12
    n_jobs: int = 2
    random_state: int = 2026
    max_training_pixels: int = 262144
    samples_per_image: int = 8192
    positive_fraction: float = 0.35
    min_samples_leaf: int = 5
    chunk_rows: int = 128
    gaussian_sigmas: tuple = (4.0, 12.0)
    blackhat_sizes: tuple = (15, 41)
    std_sizes: tuple = (9, 31)
    disk_threshold: int = 12
    limb_margin: int = 16
    probability_threshold: float = 0.65
    min_area: int = 64
    max_area_fraction: float = 0.05
    min_elongation: float = 1.2
    max_instances: int = 256

    def __post_init__(self):
        ranges = {
            "n_estimators": (1, 256), "max_depth": (1, 24), "n_jobs": (1, 2),
            "random_state": (0, 2**32 - 1),
            "max_training_pixels": (2, 524288), "samples_per_image": (2, 524288),
            "min_samples_leaf": (1, 524288), "chunk_rows": (1, 2048),
            "disk_threshold": (1, 255), "limb_margin": (0, 2048),
            "min_area": (1, 2**31 - 1), "max_instances": (1, 4096),
        }
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(name + " is outside its bounded integer range")
        for name in ("positive_fraction", "probability_threshold", "max_area_fraction", "min_elongation"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
                raise ValueError(name + " must be finite numeric")
        if not 0 < self.positive_fraction < 1 or not 0 < self.probability_threshold <= 1:
            raise ValueError("positive fraction and model threshold are invalid")
        if not 0 < self.max_area_fraction <= 1 or self.min_elongation < 1:
            raise ValueError("invalid component area or elongation limit")
        if not self.gaussian_sigmas or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number)) or not math.isfinite(v) or not 0 < v <= 64 for v in self.gaussian_sigmas):
            raise ValueError("gaussian_sigmas must contain finite positive scales <=64")
        for name in ("blackhat_sizes", "std_sizes"):
            values = getattr(self, name)
            if not values or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 3 or v > 121 or v % 2 != 1 for v in values):
                raise ValueError(name + " must contain odd integer sizes in 3..121")

    def to_dict(self):
        return asdict(self)


def _image(image):
    image = np.asarray(image)
    if image.ndim != 2 or not image.size or image.dtype != np.uint8:
        raise ValueError("image must be a nonempty uint8 grayscale array")
    return np.ascontiguousarray(image)


def _target(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape or mask.dtype.kind not in "biuf" or not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)):
        raise ValueError("training target must be a shape-matched binary union mask")
    return mask.astype(bool, copy=False)


def _disk(image, config):
    # Raw intensity is used only to identify the solar search region, not to
    # decide filament foreground. Fill dark interior holes before limb erosion.
    labels, count = ndimage.label(image >= config.disk_threshold)
    if not count:
        return np.zeros(image.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    disk = ndimage.binary_fill_holes(labels == int(np.argmax(sizes)))
    if config.limb_margin:
        padded = np.pad(disk.astype(np.uint8), 1)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
        disk = distance > config.limb_margin
    return disk


def feature_names(config=PixelForestConfig()):
    names = ["intensity"]
    for sigma in config.gaussian_sigmas:
        names.extend(("local_mean_" + str(sigma), "dark_contrast_" + str(sigma)))
    names.extend("relative_blackhat_" + str(k) for k in config.blackhat_sizes)
    names.extend("local_std_" + str(k) for k in config.std_sizes)
    names.append("gradient_magnitude")
    return tuple(names)


def feature_halo(config=PixelForestConfig()):
    # Closing is dilation followed by erosion: full support is k-1. Gaussian
    # support is explicitly truncated to ceil(3*sigma), not a library default.
    return max(1, *(int(math.ceil(3 * s)) for s in config.gaussian_sigmas),
               *(int(k) - 1 for k in config.blackhat_sizes),
               *(int(k) // 2 for k in config.std_sizes))


def _feature_block(image, start, stop, config):
    """Shared train/inference features for rows [start, stop), with full halo."""
    halo = feature_halo(config)
    top, bottom = max(0, start - halo), min(image.shape[0], stop + halo)
    patch = image[top:bottom]
    gray = patch.astype(np.float32) * np.float32(1.0 / 255.0)
    core = slice(start - top, stop - top)
    output = np.empty((stop - start, image.shape[1], len(feature_names(config))), dtype=np.float32)
    output[..., 0] = gray[core]
    column = 1
    for sigma in config.gaussian_sigmas:
        radius = int(math.ceil(3 * sigma))
        mean = cv2.GaussianBlur(gray, (2 * radius + 1, 2 * radius + 1),
                                sigmaX=float(sigma), sigmaY=float(sigma),
                                borderType=cv2.BORDER_REFLECT_101)
        output[..., column] = mean[core]
        output[..., column + 1] = (np.maximum(mean - gray, 0) / np.maximum(mean, np.float32(0.05)))[core]
        column += 2
    for size in config.blackhat_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
        closed = cv2.morphologyEx(patch, cv2.MORPH_CLOSE, kernel,
                                 borderType=cv2.BORDER_REFLECT_101).astype(np.float32)
        output[..., column] = (np.maximum(closed - patch, 0) / np.maximum(closed, np.float32(1.0)))[core]
        column += 1
    for size in config.std_sizes:
        dimensions = (int(size), int(size))
        mean = cv2.boxFilter(gray, cv2.CV_32F, dimensions, normalize=True, borderType=cv2.BORDER_REFLECT_101)
        square = cv2.boxFilter(gray * gray, cv2.CV_32F, dimensions, normalize=True, borderType=cv2.BORDER_REFLECT_101)
        output[..., column] = np.sqrt(np.maximum(square - mean * mean, 0))[core]
        column += 1
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    output[..., column] = cv2.magnitude(gx, gy)[core] * np.float32(0.25)
    return output


def iter_feature_chunks(image, config=PixelForestConfig()):
    """Yield (start_row, stop_row, float32_features); never all images/features."""
    image = _image(image)
    for start in range(0, image.shape[0], config.chunk_rows):
        stop = min(image.shape[0], start + config.chunk_rows)
        yield start, stop, _feature_block(image, start, stop, config)


def iter_coco_training(index, image_root, observations):
    """Yield image/union-target pairs for EXPLICIT training observation names.

    Only dataset's COCO segmentations are decoded; metadata is not a feature.
    Pass split.train (or a selected development subset), never held-out names.
    This training union is not valid ground truth for the official PQ evaluator.
    """
    from dataset import ground_truth_masks, read_observation
    seen = set()
    for observation in observations:
        if observation in seen:
            raise ValueError("duplicate training observation")
        seen.add(observation)
        image = read_observation(index, image_root, observation)
        target = np.zeros(image.shape, dtype=bool)
        for image_id in index.observations[observation]:
            for instance in ground_truth_masks(index, image_id):
                target |= instance
        yield image, target


class PixelForest:
    def __init__(self, config=None):
        self.config = config or PixelForestConfig()
        if not isinstance(self.config, PixelForestConfig):
            raise TypeError("config must be PixelForestConfig")
        self.model_ = None
        self.training_summary_ = None

    def fit(self, observations, masks=None):
        """Fit from an iterator of (image, union_mask), or two parallel iterables.

        One image/mask pair also works. At most max_training_pixels are retained;
        iteration stops at that cap. Input order therefore selects the bounded
        training subset and should be fixed/shuffled using training names only.
        """
        self.model_, self.training_summary_ = None, None
        if masks is not None and isinstance(observations, np.ndarray) and observations.ndim == 2:
            observations, masks = [observations], [masks]
        if masks is None:
            pairs = iter(observations)
        else:
            missing = object()
            def paired():
                for image, mask in zip_longest(observations, masks, fillvalue=missing):
                    if image is missing or mask is missing:
                        raise ValueError("image and mask iterable lengths differ")
                    yield image, mask
            pairs = paired()
        config = self.config
        rng = np.random.default_rng(config.random_state)
        x = np.empty((config.max_training_pixels, len(feature_names(config))), dtype=np.float32)
        y = np.empty(config.max_training_pixels, dtype=np.uint8)
        used, images_seen, excluded_positive = 0, 0, 0
        for pair in pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("each observation must be an (image, binary_union_mask) pair")
            image = _image(pair[0])
            target = _target(pair[1], image.shape)
            valid = _disk(image, config)
            images_seen += 1
            excluded_positive += int(np.count_nonzero(target & ~valid))
            positive = np.flatnonzero((target & valid).ravel())
            negative = np.flatnonzero((~target & valid).ravel())
            quota = min(config.samples_per_image, config.max_training_pixels - used)
            n_positive = min(positive.size, int(round(quota * config.positive_fraction)))
            n_negative = min(negative.size, quota - n_positive)
            n_positive += min(positive.size - n_positive, quota - n_positive - n_negative)
            n_negative += min(negative.size - n_negative, quota - n_positive - n_negative)
            if not n_positive + n_negative:
                continue
            chosen = np.sort(np.concatenate((
                rng.choice(positive, n_positive, replace=False),
                rng.choice(negative, n_negative, replace=False))))
            y[used:used + chosen.size] = target.ravel()[chosen]
            width = image.shape[1]
            for start in range(0, image.shape[0], config.chunk_rows):
                stop = min(image.shape[0], start + config.chunk_rows)
                low, high = np.searchsorted(chosen, (start * width, stop * width))
                if low == high:
                    continue
                features = _feature_block(image, start, stop, config)
                x[used + low:used + high] = features.reshape(-1, x.shape[1])[chosen[low:high] - start * width]
            used += chosen.size
            if used >= config.max_training_pixels:
                break
        if not used or np.unique(y[:used]).size != 2:
            raise ValueError("training requires sampled foreground and background pixels")
        model = RandomForestClassifier(
            n_estimators=config.n_estimators, max_depth=config.max_depth,
            n_jobs=config.n_jobs, random_state=config.random_state,
            min_samples_leaf=config.min_samples_leaf, max_features="sqrt",
            class_weight=None)
        model.fit(x[:used], y[:used])
        self.model_ = model
        self.training_summary_ = {
            "observations_seen": images_seen, "sampled_pixels": int(used),
            "sampled_positive": int(np.count_nonzero(y[:used])),
            "sampled_negative": int(used - np.count_nonzero(y[:used])),
            "positive_pixels_outside_search_region": excluded_positive,
            "feature_count": x.shape[1], "feature_dtype": "float32",
            "feature_halo": feature_halo(config),
            "sample_cap_reached": used == config.max_training_pixels,
            "probability_note": "Model score under sampled class balance; not calibrated prevalence",
        }
        return self

    def fit_coco(self, index, image_root, observations):
        return self.fit(iter_coco_training(index, image_root, observations))

    def _positive_probability(self, features):
        if self.model_ is None:
            raise RuntimeError("fit must complete before inference")
        positive_columns = np.flatnonzero(self.model_.classes_ == 1)
        if positive_columns.size != 1:
            raise RuntimeError("fitted model lacks a unique foreground class")
        values = self.model_.predict_proba(features.reshape(-1, features.shape[-1]))
        return values[:, positive_columns[0]].astype(np.float32).reshape(features.shape[:2])

    def predict_proba(self, image):
        """One float32 score image; no full-resolution feature tensor is kept."""
        if self.model_ is None:
            raise RuntimeError("fit must complete before inference")
        image = _image(image)
        disk = _disk(image, self.config)
        output = np.zeros(image.shape, dtype=np.float32)
        if disk.any():
            for start, stop, features in iter_feature_chunks(image, self.config):
                output[start:stop] = self._positive_probability(features)
            output[~disk] = 0
        return output

    def predict_labels(self, image, probability_threshold=None):
        """Return consecutive int32 connected-instance labels; zero is background."""
        return self.labels_from_probability(
            image, self.predict_proba(image), probability_threshold)

    def labels_from_probability(self, image, probabilities, probability_threshold=None):
        """Reuse one score image for a bounded, predefined threshold comparison.

        This is pure postprocessing: fitting is not required when scores are
        supplied. The input score map is validated and is never modified.
        """
        image = _image(image)
        probabilities = np.asarray(probabilities)
        if probabilities.shape != image.shape or probabilities.dtype.kind not in "biuf" or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("probabilities must match image shape and be finite scores in [0,1]")
        config = self.config
        threshold = config.probability_threshold if probability_threshold is None else probability_threshold
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.number)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("probability_threshold must be a model score in (0,1]")
        disk = _disk(image, config)
        output = np.zeros(image.shape, dtype=np.int32)
        if not disk.any():
            return output
        foreground = disk & (probabilities >= threshold)
        count, raw_labels, stats, centroids = cv2.connectedComponentsWithStats(
            foreground.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        del centroids
        maximum_area = config.max_area_fraction * int(np.count_nonzero(disk))
        kept = 0
        for raw_id in range(1, count):
            left, top, width, height, area = map(int, stats[raw_id])
            if area < config.min_area or area > maximum_area:
                continue
            region = np.s_[top:top + height, left:left + width]
            local = raw_labels[region] == raw_id
            yy, xx = np.nonzero(local)
            coordinates = np.column_stack((xx, yy)).astype(np.float64)
            coordinates -= coordinates.mean(axis=0)
            eigenvalues = np.linalg.eigvalsh(coordinates.T @ coordinates / area)
            elongation = math.sqrt((eigenvalues[-1] + 1) / (eigenvalues[0] + 1))
            if elongation < config.min_elongation:
                continue
            kept += 1
            if kept > config.max_instances:
                raise RuntimeError("instance safety cap exceeded; no predictions were silently truncated")
            output[region][local] = kept
        return output

    def iter_instances(self, image, probability_threshold=None):
        """Yield bool masks. The existing PQ evaluator materializes its inputs."""
        labels = self.predict_labels(image, probability_threshold)
        for instance_id in range(1, int(labels.max()) + 1):
            yield labels == instance_id

    def predict_instances(self, image, probability_threshold=None):
        """Return nonempty binary masks accepted by evaluation.instance_pq."""
        return list(self.iter_instances(image, probability_threshold))
