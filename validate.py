"""Private CPU holdout evaluation; no download, upload or model installation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from baseline import BaselineConfig, segment_image
from dataset import ground_truth_masks, load_coco_index, read_observation
from evaluation import group_by_observation_split, instance_pq


def validate(annotation_path, image_root, config=None, validation_fraction=0.2,
             seed=2026, max_observations=None):
    config = config or BaselineConfig()
    index = load_coco_index(annotation_path)
    # Deliberately repeat filenames for their annotation sets, not image IDs.
    split = group_by_observation_split(
        [record.file_name for record in index.images.values()], validation_fraction, seed)
    selected = split.validation
    if max_observations is not None:
        if isinstance(max_observations, bool) or not isinstance(max_observations, int) or max_observations < 1:
            raise ValueError("max_observations must be a positive integer")
        selected = selected[:max_observations]
    all_scores, observation_scores, records = [], [], []
    global_sum_iou, global_tp, global_fp, global_fn = 0.0, 0, 0, 0
    started = time.perf_counter()
    for observation in selected:
        image = read_observation(index, image_root, observation)
        if image.shape != (2048, 2048):
            raise ValueError("official validation expects 2048x2048 observations")
        result = segment_image(image, config)
        predictions = result.masks()
        comparisons = {}
        for image_id in index.observations[observation]:
            score = instance_pq(predictions, ground_truth_masks(index, image_id),
                                expected_shape=(2048, 2048))
            comparisons[image_id] = score.to_dict()
            all_scores.append(score.pq)
            global_sum_iou += score.sum_iou
            global_tp += score.tp
            global_fp += score.fp
            global_fn += score.fn
        mean_score = math.fsum(item["pq"] for item in comparisons.values()) / len(comparisons)
        observation_scores.append(mean_score)
        records.append({"observation": observation, "prediction_count": len(predictions),
                        "diagnostics": result.diagnostics, "comparisons": comparisons,
                        "mean_pq_across_annotation_sets": mean_score})
    denominator = global_tp + 0.5 * global_fp + 0.5 * global_fn
    return {
        "status": "PRIVATE_LOCAL_VALIDATION_NOT_A_LEADERBOARD_SUBMISSION",
        "config": config.to_dict(), "seed": seed,
        "split_group": "exact shared file_name, never individual annotation image_id",
        "split": split.to_dict(), "evaluated_observations": len(records),
        "annotation_comparisons": len(all_scores),
        "official_global_pq": global_sum_iou / denominator if denominator else 0.0,
        "official_global_counts": {"sum_iou": global_sum_iou, "tp": global_tp,
                                   "fp": global_fp, "fn": global_fn},
        "full_validation_split_evaluated": len(selected) == len(split.validation),
        "subset_warning": None if len(selected) == len(split.validation) else "Smoke subset only; not an estimate from the full holdout",
        "macro_pq_per_annotation_set": math.fsum(all_scores) / len(all_scores),
        "macro_pq_per_observation": math.fsum(observation_scores) / len(observation_scores),
        "aggregation": "Official self-evaluation v6: pool all annotator-image comparison counts, then compute one global PQ. Macro values are diagnostics only.",
        "empty_empty_score": 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", help="Optional local JSON object of BaselineConfig parameters")
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    parameters = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
    report = validate(args.annotations, args.images, BaselineConfig(**parameters),
                      seed=args.seed, max_observations=args.max_observations)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "evaluated_observations", "annotation_comparisons", "official_global_pq", "macro_pq_per_annotation_set",
        "macro_pq_per_observation", "full_validation_split_evaluated", "elapsed_seconds")}, indent=2))
