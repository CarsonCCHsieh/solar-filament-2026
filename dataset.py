"""COCO polygon input and compressed-RLE output, without metadata features.

Requires actual pycocotools for rasterization/encoding. There is deliberately
no approximate OpenCV polygon fallback for evaluation ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    file_name: str
    width: int
    height: int


@dataclass
class CocoIndex:
    images: Dict[str, ImageRecord]
    observations: Dict[str, Tuple[str, ...]]
    segmentations: Dict[str, Tuple[Any, ...]]


def index_coco(payload: dict) -> CocoIndex:
    """Only id/file_name/shape and segmentation enter the index.

    file_name is the observation key, while image_id identifies an annotation
    set. Multiple image IDs for one filename must stay in the same split.
    URLs, dates, categories/spines and other annotation metadata are unused.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list) or not isinstance(payload.get("annotations"), list):
        raise ValueError("COCO payload needs images and annotations lists")
    images, observations, segmentations = {}, {}, {}
    for item in payload["images"]:
        image_id = item.get("id")
        file_name = item.get("file_name")
        width, height = item.get("width"), item.get("height")
        if not isinstance(image_id, str) or not image_id:
            raise ValueError("expected the actual dataset's nonempty string image id")
        if image_id in images:
            raise ValueError("duplicate image id: " + image_id)
        if not isinstance(file_name, str) or not file_name or "/" in file_name or "\\" in file_name or file_name in (".", ".."):
            raise ValueError("file_name must be a single safe image filename")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
            raise ValueError("image dimensions must be positive integers")
        for previous_id in observations.get(file_name, ()):
            previous = images[previous_id]
            if (previous.width, previous.height) != (width, height):
                raise ValueError("annotators disagree on observation dimensions")
        images[image_id] = ImageRecord(image_id, file_name, width, height)
        observations.setdefault(file_name, []).append(image_id)
        segmentations[image_id] = []
    for annotation in payload["annotations"]:
        image_id = annotation.get("image_id")
        if image_id not in images:
            raise ValueError("annotation references an unknown image_id")
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, (list, dict)) or not segmentation:
            raise ValueError("annotation needs a nonempty COCO polygon or RLE segmentation")
        if isinstance(segmentation, list):
            for polygon in segmentation:
                try:
                    coords = np.asarray(polygon, dtype=np.float64)
                except (ValueError, TypeError) as error:
                    raise ValueError("invalid polygon coordinates") from error
                if coords.ndim != 1 or coords.size < 6 or coords.size % 2 or not np.isfinite(coords).all():
                    raise ValueError("polygon must contain >=3 finite x/y pairs")
        segmentations[image_id].append(segmentation)
    return CocoIndex(images, {k: tuple(v) for k, v in observations.items()},
                     {k: tuple(v) for k, v in segmentations.items()})


def load_coco_index(path) -> CocoIndex:
    with Path(path).open("r", encoding="utf-8") as stream:
        return index_coco(json.load(stream))


def _mask_api():
    try:
        from pycocotools import mask
    except ImportError as error:
        raise RuntimeError("Official COCO polygon/RLE conversion requires pycocotools; no approximate ground-truth fallback is enabled") from error
    return mask


def ground_truth_masks(index: CocoIndex, image_id: str) -> List[np.ndarray]:
    api = _mask_api()
    record = index.images[image_id]
    result = []
    for segmentation in index.segmentations[image_id]:
        if isinstance(segmentation, list):
            rle = api.merge(api.frPyObjects(segmentation, record.height, record.width))
        elif isinstance(segmentation.get("counts"), list):
            rle = api.frPyObjects(segmentation, record.height, record.width)
        else:
            rle = dict(segmentation)
            if isinstance(rle.get("counts"), str):
                rle["counts"] = rle["counts"].encode("ascii")
        decoded = np.asarray(api.decode(rle))
        if decoded.shape != (record.height, record.width):
            raise ValueError("decoded annotation mask shape mismatch")
        binary = decoded.astype(bool)
        if not binary.any():
            raise ValueError("decoded annotation is an empty instance")
        result.append(binary)
    return result


def read_observation(index: CocoIndex, image_root, observation: str) -> np.ndarray:
    image_root = Path(image_root).resolve()
    record = index.images[index.observations[observation][0]]
    path = (image_root / record.file_name).resolve()
    if path.parent != image_root:
        raise ValueError("image path escapes the supplied image directory")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("could not read observation: " + record.file_name)
    if image.shape != (record.height, record.width) or image.dtype != np.uint8:
        raise ValueError("actual image must match annotation dimensions and uint8 grayscale")
    return image


def submission_rows(file_name: str, masks, expected_shape=(2048, 2048), first_instance=1) -> List[dict]:
    """One compressed counts string per nonempty instance; no data upload.

    first_instance controls only the unique numeric suffix. Confirm the
    official sample's suffix convention before a real final submission.
    """
    api = _mask_api()
    stem = Path(file_name).stem
    if not stem or isinstance(first_instance, bool) or not isinstance(first_instance, int) or first_instance < 0:
        raise ValueError("invalid image stem or first_instance")
    rows = []
    for number, mask in enumerate(masks, start=first_instance):
        mask = np.asarray(mask)
        if mask.shape != tuple(expected_shape) or mask.dtype.kind not in "biuf" or not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)) or not np.any(mask):
            raise ValueError("submission masks must be nonempty binary masks of the official shape")
        rle = api.encode(np.asfortranarray(mask, dtype=np.uint8))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("ascii")
        if not isinstance(counts, str):
            raise ValueError("expected pycocotools compressed RLE counts")
        if not np.array_equal(api.decode(rle).astype(bool), mask.astype(bool)):
            raise ValueError("RLE round-trip mismatch")
        rows.append({"filament_id": stem + "_" + str(number), "segmentation_rle": counts})
    return rows
