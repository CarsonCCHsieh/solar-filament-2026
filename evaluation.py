"""Original mask-only PQ and observation-group validation utilities.

The pair score follows the organizer rule supplied for this project: every
pair with IoU strictly greater than 0.5 is a hit. No Hungarian matching is
performed. FP/FN are instances without any hit. The organizer self-eval v6
returns 0.0 when the denominator is zero and computes the final score from
counts accumulated over all annotator-image comparisons.

COCO polygon rasterization belongs in the data adapter, not this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from numbers import Integral, Real
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class PQResult:
    pq: float
    sum_iou: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ObservationSplit:
    train: Tuple[str, ...]
    validation: Tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnnotationEvaluation:
    # Official primary result: aggregate counts before taking the ratio.
    official_global_pq: float
    global_counts: Mapping[str, object]
    # Diagnostic only: mean over individual annotator-image comparisons.
    macro_pq: float
    # Diagnostic only: equal-observation weighting.
    observation_macro_pq: float
    comparison_count: int
    by_observation: Mapping[str, Mapping[str, PQResult]]

    def to_dict(self) -> dict:
        return {
            "official_global_pq": self.official_global_pq,
            "global_counts": dict(self.global_counts),
            "macro_pq": self.macro_pq,
            "observation_macro_pq": self.observation_macro_pq,
            "comparison_count": self.comparison_count,
            "by_observation": {
                observation: {
                    annotator: result.to_dict()
                    for annotator, result in comparisons.items()
                }
                for observation, comparisons in self.by_observation.items()
            },
        }


def _shape(shape: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if shape is None:
        return None
    try:
        values = tuple(shape)
    except TypeError as error:
        raise ValueError("expected_shape must contain two positive integers") from error
    if len(values) != 2 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value <= 0
        for value in values
    ):
        raise ValueError("expected_shape must contain two positive integers")
    return int(values[0]), int(values[1])


def _masks(
    masks: Iterable[np.ndarray],
    expected_shape: Optional[Tuple[int, int]],
    label: str,
) -> Tuple[Tuple[np.ndarray, ...], Optional[Tuple[int, int]]]:
    if isinstance(masks, np.ndarray) and masks.ndim != 3:
        raise ValueError(label + " must be a collection of 2D masks, not one 2D mask")
    try:
        values = tuple(masks)
    except TypeError as error:
        raise ValueError(label + " must be an iterable of 2D masks") from error
    result = []
    shape = expected_shape
    for index, value in enumerate(values):
        mask = np.asarray(value)
        name = "{}[{}]".format(label, index)
        if mask.ndim != 2 or any(dimension <= 0 for dimension in mask.shape):
            raise ValueError(name + " must be a nonempty 2D array")
        if shape is None:
            shape = tuple(mask.shape)
        if tuple(mask.shape) != shape:
            raise ValueError(
                "{} shape {} does not match {}".format(name, mask.shape, shape)
            )
        if mask.dtype.kind not in "biuf":
            raise ValueError(name + " must contain binary numeric or boolean values")
        if not np.all(np.isfinite(mask)):
            raise ValueError(name + " contains a nonfinite value")
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(name + " must contain only 0 and 1")
        binary = mask.astype(bool, copy=False)
        if not np.any(binary):
            raise ValueError(name + " is an empty instance; omit it instead")
        result.append(binary)
    return tuple(result), shape


def _bounds(mask: np.ndarray) -> Tuple[int, int, int, int]:
    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    return int(rows[0]), int(rows[-1] + 1), int(columns[0]), int(columns[-1] + 1)


def instance_pq(
    pred_masks: Iterable[np.ndarray],
    gt_masks: Iterable[np.ndarray],
    expected_shape: Optional[Sequence[int]] = None,
) -> PQResult:
    """Calculate strict IoU > 0.5 all-hit instance PQ.

    Masks must be 2D binary arrays of identical shape; an instance with zero
    pixels is invalid. Pass [] for no instances. Lists, generators, and
    (n, height, width) NumPy arrays are accepted. A single 2D array must be
    explicitly wrapped in a list.

    Every qualifying pair contributes to tp and sum_iou. An instance is
    unmatched only if no pair involving it qualifies. This deliberately does
    not impose one-to-one assignment, even for duplicated/overlapping masks.
    """
    target_shape = _shape(expected_shape)
    predictions, target_shape = _masks(pred_masks, target_shape, "pred_masks")
    ground_truth, target_shape = _masks(gt_masks, target_shape, "gt_masks")
    pred_area = [int(np.count_nonzero(mask)) for mask in predictions]
    gt_area = [int(np.count_nonzero(mask)) for mask in ground_truth]
    pred_bounds = [_bounds(mask) for mask in predictions]
    gt_bounds = [_bounds(mask) for mask in ground_truth]
    pred_hit = np.zeros(len(predictions), dtype=bool)
    gt_hit = np.zeros(len(ground_truth), dtype=bool)
    matched_ious = []
    for pred_index, prediction in enumerate(predictions):
        py0, py1, px0, px1 = pred_bounds[pred_index]
        for gt_index, target in enumerate(ground_truth):
            gy0, gy1, gx0, gx1 = gt_bounds[gt_index]
            y0, y1 = max(py0, gy0), min(py1, gy1)
            x0, x1 = max(px0, gx0), min(px1, gx1)
            if y0 >= y1 or x0 >= x1:
                continue
            intersection = int(
                np.count_nonzero(
                    prediction[y0:y1, x0:x1] & target[y0:y1, x0:x1]
                )
            )
            union = pred_area[pred_index] + gt_area[gt_index] - intersection
            # Compare integers for the strict threshold, avoiding rounding a
            # true boundary IoU=0.5 above the cutoff.
            if 2 * intersection > union:
                matched_ious.append(intersection / union)
                pred_hit[pred_index] = True
                gt_hit[gt_index] = True
    true_positives = len(matched_ious)
    false_positives = int(len(predictions) - np.count_nonzero(pred_hit))
    false_negatives = int(len(ground_truth) - np.count_nonzero(gt_hit))
    sum_iou = math.fsum(matched_ious)
    denominator = true_positives + 0.5 * false_positives + 0.5 * false_negatives
    pq = sum_iou / denominator if denominator else 0.0
    return PQResult(
        pq=float(pq),
        sum_iou=float(sum_iou),
        tp=true_positives,
        fp=false_positives,
        fn=false_negatives,
    )


def group_by_observation_split(
    observation_ids: Iterable[str],
    validation_fraction: float = 0.2,
    seed: int = 2026,
) -> ObservationSplit:
    """Split unique observations, never individual annotator/image IDs.

    Supply the shared physical-observation key (the image file_name in the
    current dataset), repeated as needed for its annotators. Hash ordering
    makes the split independent of input order and Python's randomized hash.
    At least two distinct observations are required. Both subsets are kept
    nonempty; therefore very small groups may not realize the exact fraction.
    """
    if (
        isinstance(validation_fraction, (bool, np.bool_))
        or not isinstance(validation_fraction, Real)
        or not math.isfinite(validation_fraction)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError("validation_fraction must be finite and between 0 and 1")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    ids = tuple(observation_ids)
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("observation IDs must be nonempty strings")
    unique_ids = set(ids)
    if len(unique_ids) < 2:
        raise ValueError("at least two distinct observations are required")
    ordered = sorted(
        unique_ids,
        key=lambda value: (
            hashlib.sha256((str(int(seed)) + "\0" + value).encode("utf-8")).digest(),
            value,
        ),
    )
    validation_count = max(
        1, min(len(ordered) - 1, int(round(len(ordered) * validation_fraction)))
    )
    return ObservationSplit(
        train=tuple(sorted(ordered[validation_count:])),
        validation=tuple(sorted(ordered[:validation_count])),
    )


def evaluate_annotators(
    predictions_by_observation: Mapping[str, Iterable[np.ndarray]],
    annotations_by_observation: Mapping[
        str, Mapping[str, Iterable[np.ndarray]]
    ],
    expected_shape: Optional[Sequence[int]] = None,
) -> AnnotationEvaluation:
    """Compare one prediction set separately against every annotator.

    The nested annotation mapping is observation_key -> annotator_or_image_id
    -> masks. Do not combine different annotators' instance masks. Missing
    predictions for a known observation mean no predicted instances; an
    unknown prediction observation is rejected as a probable keying error.

    official_global_pq follows organizer self-eval v6: sum IoUs, TP, FP and
    FN over every annotator-image comparison, then calculate one PQ ratio.
    A zero denominator returns 0.0. The two macro averages are diagnostics
    only: macro_pq weights each comparison equally; observation_macro_pq
    averages annotators first and then observations.
    """
    if not annotations_by_observation:
        raise ValueError("at least one annotated observation is required")
    extras = set(predictions_by_observation) - set(annotations_by_observation)
    if extras:
        raise ValueError("predictions include unannotated observations: " + repr(sorted(extras)))
    target_shape = _shape(expected_shape)
    by_observation = {}
    all_scores = []
    observation_scores = []
    all_results = []
    for observation in sorted(annotations_by_observation):
        annotators = annotations_by_observation[observation]
        if not annotators:
            raise ValueError("each observation requires at least one annotator")
        # Materialize once so generators can be reused for every annotator.
        predictions = tuple(predictions_by_observation.get(observation, ()))
        results = {}
        for annotator in sorted(annotators):
            result = instance_pq(
                predictions, annotators[annotator], expected_shape=target_shape
            )
            results[annotator] = result
            all_scores.append(result.pq)
            all_results.append(result)
        by_observation[observation] = results
        observation_scores.append(math.fsum(r.pq for r in results.values()) / len(results))
    global_sum_iou = math.fsum(result.sum_iou for result in all_results)
    global_tp = sum(result.tp for result in all_results)
    global_fp = sum(result.fp for result in all_results)
    global_fn = sum(result.fn for result in all_results)
    denominator = global_tp + 0.5 * global_fp + 0.5 * global_fn
    return AnnotationEvaluation(
        official_global_pq=global_sum_iou / denominator if denominator else 0.0,
        global_counts={
            "sum_iou": global_sum_iou,
            "tp": global_tp,
            "fp": global_fp,
            "fn": global_fn,
        },
        macro_pq=math.fsum(all_scores) / len(all_scores),
        observation_macro_pq=math.fsum(observation_scores) / len(observation_scores),
        comparison_count=len(all_scores),
        by_observation=by_observation,
    )

