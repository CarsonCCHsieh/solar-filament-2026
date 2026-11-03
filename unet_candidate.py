"""Original small U-Net candidate, trained from scratch on explicit train names.

No pretrained weights, downloads, network access or automatic data discovery.
Training unions annotators only for the semantic target; official instance PQ
must still evaluate each original annotator separately at the original size.
No real-data quality is established by this module or its synthetic tests.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage


@dataclass(frozen=True)
class UNetConfig:
    input_size: int = 512
    channels: tuple = (16, 32, 64, 128)
    epochs: int = 8
    batch_size: int = 4
    learning_rate: float = 0.001
    positive_weight: float = 12.0
    random_state: int = 2026
    cache_bytes: int = 256 * 1024 * 1024
    probability_threshold: float = 0.65
    min_area: int = 128
    max_area_fraction: float = 0.05
    min_elongation: float = 1.2
    disk_threshold: int = 12
    limb_margin: int = 16
    max_instances: int = 256
    augment: bool = True
    device: str = "cpu"

    def __post_init__(self):
        bounds = {
            "input_size": (32, 1024), "epochs": (1, 100),
            "batch_size": (1, 16), "random_state": (0, 2**32 - 1),
            "cache_bytes": (0, 256 * 1024 * 1024), "min_area": (1, 2**31 - 1),
            "disk_threshold": (1, 255), "limb_margin": (0, 2048),
            "max_instances": (1, 4096),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(name + " outside its bounded integer range")
        if self.input_size % 8:
            raise ValueError("input_size must be divisible by eight")
        if not isinstance(self.channels, tuple) or len(self.channels) != 4 or any(
            isinstance(c, bool) or not isinstance(c, int) or c < 4 or c > 128 or c % 4
            for c in self.channels
        ):
            raise ValueError("channels must be four multiples of four in 4..128")
        ranges = {
            "learning_rate": (0, 0.1), "positive_weight": (0, 100),
            "probability_threshold": (0, 1), "max_area_fraction": (0, 1),
        }
        for name, (low, high) in ranges.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low < value <= high:
                raise ValueError(name + " must be finite and within bounds")
        if not isinstance(self.min_elongation, (int, float)) or isinstance(self.min_elongation, bool) or not math.isfinite(self.min_elongation) or self.min_elongation < 1:
            raise ValueError("min_elongation must be finite and at least one")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must explicitly be cpu or cuda")
        if not isinstance(self.augment, bool):
            raise ValueError("augment must be bool")

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
        raise ValueError("target must be a shape-matched binary union mask")
    return mask.astype(np.uint8)


def prepare_pair(image, mask, input_size):
    """Return compact uint8 image and quantized foreground coverage target.

    Area-resampled masks preserve partial coverage of thin filaments instead
    of silently dropping them with nearest-neighbour downsampling.
    """
    image = _image(image)
    target = _target(mask, image.shape)
    image_small = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_AREA)
    coverage = cv2.resize(target.astype(np.float32), (input_size, input_size), interpolation=cv2.INTER_AREA)
    return image_small, np.rint(np.clip(coverage, 0, 1) * 255).astype(np.uint8)


def image_channels(image_small):
    """Two image-only channels: intensity and signed local normalized contrast."""
    gray = image_small.astype(np.float32) / np.float32(255.0)
    mean = cv2.GaussianBlur(gray, (49, 49), sigmaX=8.0, borderType=cv2.BORDER_REFLECT_101)
    contrast = np.clip((mean - gray) / np.maximum(mean, np.float32(0.05)), -1, 1)
    return np.stack((gray, contrast), axis=0).astype(np.float32)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels), nn.SiLU(),
        )

    def forward(self, image):
        return self.layers(image)


class SmallUNet(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128)):
        super().__init__()
        a, b, c, d = channels
        self.encoder = nn.ModuleList((
            _ConvBlock(2, a), _ConvBlock(a, b), _ConvBlock(b, c), _ConvBlock(c, d),
        ))
        self.decoder = nn.ModuleList((
            _ConvBlock(d + c, c), _ConvBlock(c + b, b), _ConvBlock(b + a, a),
        ))
        self.head = nn.Conv2d(a, 1, 1)

    def forward(self, image):
        skips = []
        value = image
        for number, layer in enumerate(self.encoder):
            if number:
                value = F.max_pool2d(value, 2)
            value = layer(value)
            skips.append(value)
        for layer, skip in zip(self.decoder, reversed(skips[:-1])):
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            value = layer(torch.cat((value, skip), dim=1))
        return self.head(value)


def segmentation_loss(logits, target, positive_weight=12.0):
    """BCE-with-logits plus soft Dice, both averaged over the batch."""
    positive_weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)
    scores = torch.sigmoid(logits)
    axes = (1, 2, 3)
    dice = (2 * (scores * target).sum(axes) + 1) / (scores.sum(axes) + target.sum(axes) + 1)
    return bce + (1 - dice).mean()


class _PairCache:
    """Bounded LRU of compressed uint8 image/coverage pairs, not tensors."""
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.bytes = 0
        self.peak_bytes = 0
        self.values = OrderedDict()

    def get(self, name, loader):
        if name in self.values:
            self.values.move_to_end(name)
            return self.values[name]
        value = loader(name)
        size = sum(array.nbytes for array in value)
        if size <= self.max_bytes:
            while self.values and self.bytes + size > self.max_bytes:
                _, removed = self.values.popitem(last=False)
                self.bytes -= sum(array.nbytes for array in removed)
            self.values[name] = value
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        return value


def _training_names(observations, allowed_training_observations, forbidden_observations):
    names = tuple(observations)
    if not names or any(not isinstance(n, str) or not n or "/" in n or "\\" in n for n in names):
        raise ValueError("explicit nonempty observation filename list required")
    if len(set(names)) != len(names):
        raise ValueError("duplicate training observation")
    if allowed_training_observations is None:
        raise ValueError("allowed_training_observations must explicitly establish the train split")
    allowed, forbidden = set(allowed_training_observations), set(forbidden_observations)
    if not set(names) <= allowed or set(names) & forbidden:
        raise ValueError("training observations cross the declared split")
    return names


class UNetCandidate:
    def __init__(self, config=None):
        self.config = config or UNetConfig()
        if not isinstance(self.config, UNetConfig):
            raise TypeError("config must be UNetConfig")
        self.model_ = None
        self.training_summary_ = None

    def _device(self):
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
        return torch.device(self.config.device)

    def _fit_loader(self, names, loader, progress=None, checkpoint_dir=None):
        config = self.config
        self.model_, self.training_summary_ = None, None
        device = self._device()
        torch.manual_seed(config.random_state)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.random_state)
        rng = np.random.default_rng(config.random_state)
        cache = _PairCache(config.cache_bytes)
        model = SmallUNet(config.channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        history = []
        started = time.perf_counter()
        model.train()
        for epoch in range(config.epochs):
            shuffled = rng.permutation(len(names))
            total_loss, total_examples = 0.0, 0
            for start in range(0, len(names), config.batch_size):
                inputs, targets = [], []
                for number in shuffled[start:start + config.batch_size]:
                    gray, coverage = cache.get(names[int(number)], loader)
                    # Copies ensure augmentation never mutates cached arrays.
                    gray, coverage = gray.copy(), coverage.copy()
                    if config.augment:
                        rotation = int(rng.integers(0, 4))
                        gray, coverage = np.rot90(gray, rotation), np.rot90(coverage, rotation)
                        if rng.random() < 0.5:
                            gray, coverage = np.fliplr(gray), np.fliplr(coverage)
                    inputs.append(image_channels(np.ascontiguousarray(gray)))
                    targets.append(np.ascontiguousarray(coverage[None], dtype=np.float32) / 255.0)
                batch = torch.from_numpy(np.stack(inputs)).to(device)
                target = torch.from_numpy(np.stack(targets)).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = segmentation_loss(model(batch), target, config.positive_weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(inputs)
                total_examples += len(inputs)
            item = {"epoch": epoch + 1, "mean_training_loss": total_loss / total_examples,
                    "elapsed_seconds": time.perf_counter() - started}
            history.append(item)
            if checkpoint_dir is not None:
                manifest = {"status": "epoch_checkpoint_not_heldout_evaluated",
                            "observations": list(names), "observation_count": len(names),
                            "epochs_completed": epoch + 1, "epochs_requested": config.epochs,
                            "history": history, "external_weights": False,
                            "torch_version": str(torch.__version__), "device": str(device),
                            "cache_peak_bytes": cache.peak_bytes}
                self._save_bundle(checkpoint_dir, model, manifest)
            if progress is not None:
                progress(dict(item))
        model.eval()
        self.model_ = model
        self.training_summary_ = {
            "observations": list(names), "observation_count": len(names),
            "epochs": config.epochs, "history": history,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "cache_peak_bytes": cache.peak_bytes, "cache_limit_bytes": config.cache_bytes,
            "device": str(device), "torch_version": str(torch.__version__),
            "external_weights": False, "training_targets": "union of training annotators only",
            "label_resize": "area coverage quantized to uint8/255",
            "quality_claim": "training loss is not held-out instance PQ",
        }
        return self

    def fit_coco(self, index, image_root, observations, *,
                 allowed_training_observations, forbidden_observations=(), progress=None,
                 checkpoint_dir=None):
        """Train only on explicit, disjoint observation names from split.train."""
        from dataset import ground_truth_masks, read_observation
        names = _training_names(observations, allowed_training_observations, forbidden_observations)
        if not set(names) <= set(index.observations):
            raise ValueError("unknown training observation")

        def load(name):
            image = read_observation(index, image_root, name)
            union = np.zeros(image.shape, dtype=bool)
            for image_id in index.observations[name]:
                for mask in ground_truth_masks(index, image_id):
                    union |= mask
            return prepare_pair(image, union, self.config.input_size)
        return self._fit_loader(names, load, progress, checkpoint_dir)

    def fit_pairs(self, observations, *, allowed_training_observations,
                  forbidden_observations=(), progress=None, checkpoint_dir=None):
        """Small in-memory/synthetic adapter mapping name -> (image, union)."""
        names = _training_names(tuple(observations), allowed_training_observations, forbidden_observations)
        def load(name):
            image, union = observations[name]
            return prepare_pair(image, union, self.config.input_size)
        return self._fit_loader(names, load, progress, checkpoint_dir)

    def predict_proba(self, image):
        if self.model_ is None:
            raise RuntimeError("fit or load_weights must complete before inference")
        image = _image(image)
        gray = cv2.resize(image, (self.config.input_size, self.config.input_size), interpolation=cv2.INTER_AREA)
        batch = torch.from_numpy(image_channels(gray)[None]).to(self._device())
        self.model_.eval()
        with torch.inference_mode():
            prediction = torch.sigmoid(self.model_(batch))
            prediction = F.interpolate(prediction, size=image.shape, mode="bilinear", align_corners=False)
            probability = prediction[0, 0].cpu().numpy().astype(np.float32)
        if probability.shape != image.shape or not np.isfinite(probability).all():
            raise RuntimeError("invalid model probability output")
        return np.clip(probability, 0, 1)

    def labels_from_probability(self, image, probabilities, probability_threshold=None):
        """Shared, bounded connected-component postprocessing at original size."""
        image = _image(image)
        probabilities = np.asarray(probabilities)
        if probabilities.shape != image.shape or probabilities.dtype.kind not in "biuf" or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("probabilities must match image shape and lie in [0,1]")
        threshold = self.config.probability_threshold if probability_threshold is None else probability_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0,1]")
        config = self.config
        disk_labels, count = ndimage.label(image >= config.disk_threshold)
        output = np.zeros(image.shape, dtype=np.int32)
        if not count:
            return output
        sizes = np.bincount(disk_labels.ravel())
        sizes[0] = 0
        disk = ndimage.binary_fill_holes(disk_labels == int(np.argmax(sizes)))
        if config.limb_margin:
            distance = cv2.distanceTransform(np.pad(disk.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
            disk = distance > config.limb_margin
        count, raw_labels, stats, _ = cv2.connectedComponentsWithStats(
            (disk & (probabilities >= threshold)).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        maximum = config.max_area_fraction * int(np.count_nonzero(disk))
        kept = 0
        for raw_id in range(1, count):
            left, top, width, height, area = map(int, stats[raw_id])
            if area < config.min_area or area > maximum:
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
                raise RuntimeError("instance safety cap exceeded; no predictions silently truncated")
            output[region][local] = kept
        return output

    def predict_labels(self, image, probability_threshold=None):
        return self.labels_from_probability(image, self.predict_proba(image), probability_threshold)

    def predict_instances(self, image, probability_threshold=None):
        labels = self.predict_labels(image, probability_threshold)
        return [labels == label for label in range(1, int(labels.max()) + 1)]

    def save_weights(self, path):
        """Save state_dict tensors only; keep config and source alongside it."""
        if self.model_ is None:
            raise RuntimeError("fit must complete before saving")
        state = {name: value.detach().cpu() for name, value in self.model_.state_dict().items()}
        torch.save(state, Path(path))

    def _save_bundle(self, directory, model, manifest):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        torch.save(state, directory / "weights.pt")
        (directory / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2) + "\n", encoding="utf-8")
        (directory / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def save_artifacts(self, directory):
        """Private/local bundle: tensor state, config, and actual train manifest."""
        if self.model_ is None:
            raise RuntimeError("fit must complete before saving")
        self._save_bundle(directory, self.model_, self.training_summary_)

    @classmethod
    def load_artifacts(cls, directory, *, device=None):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        config["channels"] = tuple(config["channels"])
        if device is not None:
            config["device"] = device
        predictor = cls(UNetConfig(**config)).load_weights(directory / "weights.pt")
        predictor.training_summary_ = json.loads((directory / "training-manifest.json").read_text(encoding="utf-8"))
        return predictor

    def load_weights(self, path):
        """Load only a reviewed local state_dict, never arbitrary pickle objects."""
        model = SmallUNet(self.config.channels).to(self._device())
        model.load_state_dict(torch.load(Path(path), map_location=self._device(), weights_only=True), strict=True)
        self.model_ = model.eval()
        self.training_summary_ = {"loaded_local_state_dict": True, "training_history_available": False}
        return self
