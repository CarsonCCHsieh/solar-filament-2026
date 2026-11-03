"""Reproducible Solar post-training pipeline. No downloads or uploads.

Dev-only threshold freeze precedes the reserved holdout and test inference.
Only genuine pycocotools is used for native compressed-COCO verification.
The server's acceptance of zero-row images remains unverified by this code.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import zipfile

import cv2
import numpy as np

from dataset import ground_truth_masks, load_coco_index, read_observation, submission_rows
from evaluation import group_by_observation_split, instance_pq
from unet_candidate import UNetCandidate, UNetConfig

THRESHOLDS = (0.45, 0.65, 0.80)
OFFICIAL_SHAPE = (2048, 2048)
EXPECTED_COUNTS = (707, 566, 141, 558, 8, 180)
EMPTY_POLICY = "No CSV row for an image with zero detections; record it in coverage manifest. Offline evaluator supports this; server acceptance unverified."


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _counts():
    return dict(sum_iou=0.0, tp=0, fp=0, fn=0)


def _add_counts(total, score):
    for key in total:
        total[key] += score[key] if isinstance(score, dict) else getattr(score, key)


def _pq(counts):
    denominator = counts["tp"] + 0.5 * counts["fp"] + 0.5 * counts["fn"]
    return counts["sum_iou"] / denominator if denominator else 0.0


def partitions(index, strict=True):
    """Same observation grouping and evenly spaced dev8 as actual training."""
    split = group_by_observation_split([row.file_name for row in index.images.values()], 0.2, 2026)
    development = [split.train[int(i)] for i in np.linspace(0, len(split.train) - 1, 8)]
    training = [name for name in split.train if name not in set(development)]
    heldout = list(split.validation)
    if len(development) != len(set(development)):
        raise ValueError("not enough observations for distinct dev8")
    if set(training) & set(development) or set(training + development) & set(heldout):
        raise ValueError("observation split leakage")
    actual = (len(index.observations), len(split.train), len(heldout), len(training), len(development))
    if strict and actual != EXPECTED_COUNTS[:5]:
        raise ValueError("unexpected official training counts: " + repr(actual))
    return dict(training=training, development=development, holdout=heldout,
                combined_train=list(split.train), split_seed=2026, validation_fraction=0.2)


def validate_training_bundle(bundle_dir, split, expected_epochs=24):
    bundle_dir = Path(bundle_dir)
    config = _read_json(bundle_dir / "config.json")
    manifest = _read_json(bundle_dir / "training-manifest.json")
    if manifest.get("observations") != split["training"]:
        raise ValueError("bundle training observation manifest differs from fixed train558")
    if manifest.get("observation_count") != len(split["training"]):
        raise ValueError("bundle observation count mismatch")
    if config.get("epochs") != expected_epochs or manifest.get("epochs") != expected_epochs:
        raise ValueError("only the completed requested training bundle is accepted")
    history = manifest.get("history", [])
    if [row.get("epoch") for row in history] != list(range(1, expected_epochs + 1)):
        raise ValueError("bundle does not contain every completed training epoch")
    if any(not math.isfinite(row["mean_training_loss"]) for row in history):
        raise ValueError("nonfinite training history")
    return dict(weights_sha256=_hash_file(bundle_dir / "weights.pt"),
                config_sha256=_hash_file(bundle_dir / "config.json"),
                training_manifest_sha256=_hash_file(bundle_dir / "training-manifest.json"),
                training_summary_sha256=_digest(manifest),
                split_sha256=_digest(split), config=config, source_sha256=source_manifest())


def _gt_universe(index, observation):
    # get_overlap_df in organizer v6 enumerates GT CSV filament prefixes.
    # An annotator-image with no filament rows does not enter that table.
    included = [image_id for image_id in index.observations[observation]
                if len(index.segmentations[image_id]) > 0]
    skipped = [image_id for image_id in index.observations[observation] if image_id not in included]
    return included, skipped


def _comparison_counts(index, observation, predictions):
    comparisons, total = {}, _counts()
    included, skipped = _gt_universe(index, observation)
    for image_id in included:
        score = instance_pq(predictions, ground_truth_masks(index, image_id), expected_shape=OFFICIAL_SHAPE)
        comparisons[image_id] = score.to_dict()
        _add_counts(total, score)
    return comparisons, total, skipped


def _masks(labels):
    for number in range(1, int(labels.max()) + 1):
        yield labels == number


def evaluate_development(model, index, image_root, split, output_path):
    """Only used when no real completed dev receipt is available yet."""
    totals = {str(threshold): _counts() for threshold in THRESHOLDS}
    records = []
    for observation in split["development"]:
        image = read_observation(index, image_root, observation)
        probability = model.predict_proba(image)
        row = dict(observation=observation, thresholds={})
        for threshold in THRESHOLDS:
            labels = model.labels_from_probability(image, probability, threshold)
            comparisons, counts, skipped = _comparison_counts(index, observation, list(_masks(labels)))
            _add_counts(totals[str(threshold)], counts)
            row["thresholds"][str(threshold)] = dict(prediction_count=int(labels.max()),
                comparisons=comparisons, zero_gt_sets_excluded=skipped)
        records.append(row)
        print("FINAL_DEV", json.dumps({"observation": observation}), flush=True)
    report = dict(status="DEVELOPMENT_ONLY_TRAIN558_DEV8_DISJOINT_HOLDOUT141_UNTOUCHED",
                  config=model.config.to_dict(), training=split["training"],
                  training_summary=model.training_summary_,
                  development=split["development"], holdout_reserved=len(split["holdout"]),
                  trials=[dict(threshold=t, counts=totals[str(t)], official_global_pq=_pq(totals[str(t)])) for t in THRESHOLDS],
                  records=records)
    _json(output_path, report)
    return report


def _development_totals(report, split, index, source_config):
    if report.get("training") != split["training"] or report.get("development") != split["development"]:
        raise ValueError("development receipt has different train/dev observations")
    if report.get("holdout_reserved") != len(split["holdout"]):
        raise ValueError("development receipt holdout count differs")
    config = dict(report.get("config", {}))
    expected = dict(source_config)
    config.pop("device", None)
    expected.pop("device", None)
    # JSON represents tuples as lists; canonicalization treats both the same.
    if _canonical(config) != _canonical(expected):
        raise ValueError("development receipt model config differs from artifact")
    records = report.get("records", [])
    if [r.get("observation") for r in records] != split["development"]:
        raise ValueError("development receipt must contain exactly the fixed dev8 in order")
    totals = {str(threshold): _counts() for threshold in THRESHOLDS}
    skipped_ids = set()
    for row in records:
        image_ids, skipped = _gt_universe(index, row["observation"])
        skipped_ids.update(skipped)
        if set(row.get("thresholds", {})) != set(totals):
            raise ValueError("development receipt threshold set changed")
        for key in totals:
            comparisons = row["thresholds"][key].get("comparisons", {})
            if not set(image_ids) <= set(comparisons) or not set(comparisons) <= set(index.observations[row["observation"]]):
                raise ValueError("development receipt annotator-image membership differs")
            for image_id in image_ids:
                score = comparisons[image_id]
                if any(not isinstance(score[k], int) or isinstance(score[k], bool) or score[k] < 0 for k in ("tp", "fp", "fn")):
                    raise ValueError("invalid development integer counts")
                if not isinstance(score["sum_iou"], (int, float)) or not math.isfinite(score["sum_iou"]) or score["sum_iou"] < 0:
                    raise ValueError("invalid development overlap sum")
                _add_counts(totals[key], score)
    return totals, sorted(skipped_ids)


def freeze_threshold(output_dir, identity, split, report, index):
    """One deterministic dev-only choice; any incompatible existing freeze fails."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "training_summary_sha256" in identity and _digest(report.get("training_summary")) != identity["training_summary_sha256"]:
        raise ValueError("development receipt training summary differs from completed model bundle")
    totals, excluded = _development_totals(report, split, index, identity["config"])
    trials = [dict(threshold=t, counts=totals[str(t)], official_global_pq=_pq(totals[str(t)])) for t in THRESHOLDS]
    selected = max(trials, key=lambda row: row["official_global_pq"])
    payload = dict(schema_version=1, weights_sha256=identity["weights_sha256"],
        config_sha256=identity["config_sha256"], training_manifest_sha256=identity["training_manifest_sha256"],
        split_sha256=identity["split_sha256"], development_report_sha256=_digest(report),
        source_sha256=identity["source_sha256"],
        thresholds_predefined=list(THRESHOLDS), tie_policy="first in predefined order",
        selected_threshold=selected["threshold"], development_trials=trials,
        zero_gt_sets_excluded=excluded, selection_source="fixed dev8 only, before holdout/test")
    payload["freeze_id"] = _digest(payload)
    path = output_dir / "threshold-freeze.json"
    if path.exists():
        existing = _read_json(path)
        comparable = dict(existing)
        comparable.pop("created_at_utc", None)
        if comparable != payload:
            raise ValueError("existing threshold freeze is incompatible; no overwrite or re-tuning")
        return existing
    payload["created_at_utc"] = _now()
    _json(path, payload)
    return payload


def _journal(path, context_id, allowed_names):
    seen = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            name = row.get("observation")
            if row.get("context_id") != context_id or name not in allowed_names or name in seen:
                raise ValueError("incompatible or duplicate resume journal entry")
            seen[name] = row
    return seen


def _append(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()


def _figure(model, image, labels, index, observation, counts, output_dir, display_figures):
    import matplotlib.pyplot as plt
    union = np.zeros(image.shape, dtype=bool)
    for image_id in index.observations[observation]:
        for mask in ground_truth_masks(index, image_id):
            union |= mask
    # Figures are derived comparison panels for the noncommercial report,
    # not a redistribution of original JPEGs or annotation files.
    size = (768, 768)
    gray = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    target = cv2.resize(union.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    predicted = cv2.resize((labels > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), constrained_layout=True)
    for axis in axes:
        axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
        axis.axis("off")
    axes[0].set_title("H-alpha observation")
    axes[1].set_title("GT union boundary (display only)")
    axes[2].set_title("Prediction union boundary")
    if target.any() and not target.all():
        axes[1].contour(target, levels=[0.5], colors=["#40dd70"], linewidths=0.65)
    if predicted.any() and not predicted.all():
        axes[2].contour(predicted, levels=[0.5], colors=["#ff9f31"], linewidths=0.65)
    figure.suptitle(observation + " | per-observation global PQ=" + format(_pq(counts), ".4f") +
                    " | TP=" + str(counts["tp"]) + " FP=" + str(counts["fp"]) + " FN=" + str(counts["fn"]), fontsize=10)
    figure.text(0.5, 0.005, "MAGFiLO / NSO GONG H-alpha — CC BY-NC 4.0. Fixed first-three holdout examples; GT annotators scored separately.",
                ha="center", fontsize=7)
    stem = "holdout-example-" + Path(observation).stem
    png, pdf = Path(output_dir) / (stem + ".png"), Path(output_dir) / (stem + ".pdf")
    figure.savefig(png, dpi=140, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    if display_figures:
        from IPython.display import display
        display(figure)
    plt.close(figure)
    return dict(observation=observation, selection="first three fixed sorted holdout names, chosen before evaluation",
                per_observation_global_pq=_pq(counts), counts=counts,
                files=[dict(name=p.name, bytes=p.stat().st_size, sha256=_hash_file(p)) for p in (png, pdf)],
                ground_truth_union_is_display_only=True)


def bootstrap_observation_pq(records, resamples=1000, seed=2026):
    """Exploratory cluster bootstrap from existing counts, no new inference."""
    if not records or len({r["observation"] for r in records}) != len(records):
        raise ValueError("bootstrap requires unique observation records")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or not 1 <= resamples <= 10000:
        raise ValueError("bootstrap resamples must be in 1..10000")
    values = np.asarray([[r["counts"][key] for key in ("sum_iou", "tp", "fp", "fn")]
                         for r in records], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("bootstrap counts must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(records), size=(resamples, len(records)))
    total = values[sampled].sum(axis=1)
    denominator = total[:, 1] + 0.5 * total[:, 2] + 0.5 * total[:, 3]
    scores = np.divide(total[:, 0], denominator, out=np.zeros(resamples), where=denominator > 0)
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return dict(method="observation-cluster nonparametric percentile bootstrap",
        cluster="one physical observation with all annotator-image counts kept together",
        resamples=resamples, seed=seed, percentile_interval=[2.5, 97.5],
        interval_95=[float(lower), float(upper)], observation_count=len(records),
        exploratory=True, guaranteed_coverage=False, inference_rerun=False,
        limitations="Hash-based filename split does not establish temporal independence. Correlated observations and small development set limit interpretation; interval coverage is not guaranteed.")


def evaluate_holdout(model, index, image_root, split, freeze, output_dir, display_figures=True):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    names = split["holdout"]
    chosen_figures = names[:3]  # Freeze before looking at any holdout predictions.
    journal = output_dir / "holdout-results.jsonl"
    completed = _journal(journal, freeze["freeze_id"], set(names))
    for number, observation in enumerate(names, start=1):
        if observation in completed:
            continue
        image = read_observation(index, image_root, observation)
        labels = model.predict_labels(image, freeze["selected_threshold"])
        comparisons, counts, excluded = _comparison_counts(index, observation, list(_masks(labels)))
        row = dict(context_id=freeze["freeze_id"], observation=observation,
                   prediction_count=int(labels.max()), comparisons=comparisons,
                   counts=counts, official_global_pq=_pq(counts), zero_gt_sets_excluded=excluded)
        if observation in chosen_figures:
            row["figure"] = _figure(model, image, labels, index, observation, counts, figures_dir, display_figures)
        _append(journal, row)
        completed[observation] = row
        if number <= 3 or number % 10 == 0 or number == len(names):
            print("FINAL_HOLDOUT_PROGRESS", json.dumps(dict(completed=number, total=len(names))), flush=True)
    totals = _counts()
    for observation in names:
        _add_counts(totals, completed[observation]["counts"])
    figures = [completed[name]["figure"] for name in chosen_figures]
    for item in figures:
        for entry in item["files"]:
            path = figures_dir / entry["name"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or _hash_file(path) != entry["sha256"]:
                raise ValueError("completed holdout figure is missing or changed")
    report = dict(status="HELDOUT_EVALUATION_COMPLETE_NO_RETUNING", context_id=freeze["freeze_id"],
        observation_count=len(names), observations=names, threshold=freeze["selected_threshold"],
        counts=totals, official_global_pq=_pq(totals), comparison_count=sum(len(completed[n]["comparisons"]) for n in names),
        zero_gt_sets_excluded=sorted({i for n in names for i in completed[n]["zero_gt_sets_excluded"]}),
        score_universe="Organizer v6 GT-filament-row universe; each nonempty annotator-image scored separately",
        uncertainty=bootstrap_observation_pq([completed[name] for name in names]),
        figures=figures, completed_at_utc=_now(), retuning_permitted=False)
    _json(output_dir / "holdout-results.json", report)
    _json(figures_dir / "figures-manifest.json", dict(attribution="MAGFiLO / NSO GONG H-alpha; CC BY-NC 4.0; noncommercial competition report",
        selection=chosen_figures, selection_before_holdout_evaluation=True, figures=figures))
    return report


def mask_digest(mask):
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool).ravel(), bitorder="big").tobytes()).hexdigest()


def validate_structural_rows(rows, valid_stems):
    seen = set()
    for row in rows:
        if set(row) != {"filament_id", "segmentation_rle"}:
            raise ValueError("CSV columns differ from official contract")
        filament_id, counts = row["filament_id"], row["segmentation_rle"]
        if not isinstance(filament_id, str) or "_" not in filament_id:
            raise ValueError("filament_id lacks a suffix")
        stem, suffix = filament_id.split("_", 1)
        if stem not in valid_stems or not suffix or filament_id in seen:
            raise ValueError("unknown image stem, empty suffix, or duplicate filament_id")
        if not isinstance(counts, str) or not counts or any(c in counts for c in ("'", '"', "\n", "\r")):
            raise ValueError("counts must be a nonempty bare compressed string")
        counts.encode("ascii")
        seen.add(filament_id)
    return seen


def discover_test_images(test_root, expected_count=180):
    root = Path(test_root).resolve()
    paths = sorted(root.glob("*.jpeg"))
    if len(paths) != expected_count or any(p.resolve().parent != root for p in paths):
        raise ValueError("unexpected test image count or unsafe path")
    stems = [path.stem for path in paths]
    if len(set(stems)) != len(stems) or any("_" in stem or not stem for stem in stems):
        raise ValueError("test stem cannot be parsed unambiguously by official first-underscore reader")
    return paths


def infer_test(model, test_root, freeze, output_dir, expected_count=180):
    from pycocotools import mask as mask_api  # Actual installed dependency; no fallback.
    output_dir = Path(output_dir)
    paths = discover_test_images(test_root, expected_count)
    valid_names, valid_stems = {p.name for p in paths}, {p.stem for p in paths}
    journal = output_dir / "test-predictions.jsonl"
    completed = _journal(journal, freeze["freeze_id"], valid_names)
    for number, path in enumerate(paths, start=1):
        if path.name in completed:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != OFFICIAL_SHAPE or image.dtype != np.uint8:
            raise ValueError("test image differs from official grayscale shape/dtype")
        labels = model.predict_labels(image, freeze["selected_threshold"])
        mask_hashes = {}
        def masks():
            for label, mask in enumerate(_masks(labels), start=1):
                mask_hashes[path.stem + "_" + str(label)] = mask_digest(mask)
                yield mask
        rows = submission_rows(path.name, masks(), expected_shape=OFFICIAL_SHAPE, first_instance=1)
        validate_structural_rows(rows, valid_stems)
        row = dict(context_id=freeze["freeze_id"], observation=path.name,
                   prediction_count=len(rows), rows=rows, mask_sha256=mask_hashes,
                   native_coco_encode_decode_roundtrip=True,
                   zero_detection_policy=EMPTY_POLICY if not rows else None)
        _append(journal, row)
        completed[path.name] = row
        if number % 10 == 0 or number == len(paths):
            print("FINAL_TEST_PROGRESS", json.dumps(dict(completed=number, total=len(paths))), flush=True)
    rows = [row for p in paths for row in completed[p.name]["rows"]]
    all_ids = validate_structural_rows(rows, valid_stems)
    if sum(completed[p.name]["prediction_count"] for p in paths) != len(rows):
        raise ValueError("test manifest instance count mismatch")
    csv_path = output_dir / "submission.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        writer.writerows(rows)
    # Verify the actual written CSV and decode every row with native COCO.
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("written CSV header differs")
        read_back = list(reader)
    if read_back != rows:
        raise ValueError("actual CSV serialization roundtrip changed row data")
    validate_structural_rows(read_back, valid_stems)
    for row in read_back:
        stem = row["filament_id"].split("_", 1)[0]
        decoded = mask_api.decode(dict(size=list(OFFICIAL_SHAPE), counts=row["segmentation_rle"].encode("ascii")))
        if decoded.shape != OFFICIAL_SHAPE or not decoded.any():
            raise ValueError("actual CSV RLE decoded to wrong shape or empty foreground")
        expected = completed[stem + ".jpeg"]["mask_sha256"][row["filament_id"]]
        if mask_digest(decoded) != expected:
            raise ValueError("actual CSV RLE differs from original inferred mask")
    zero_names = [p.name for p in paths if not completed[p.name]["rows"]]
    manifest = dict(status="LOCAL_CSV_COMPLETE_NOT_UPLOADED_OR_SERVER_VALIDATED",
        context_id=freeze["freeze_id"], threshold=freeze["selected_threshold"],
        test_images_processed=len(paths), images_with_predictions=len(paths) - len(zero_names),
        zero_detection_images=zero_names, zero_detection_policy=EMPTY_POLICY,
        per_image_prediction_counts={p.name: completed[p.name]["prediction_count"] for p in paths},
        row_count=len(rows), unique_row_ids=len(all_ids), original_mask_shape=list(OFFICIAL_SHAPE),
        actual_csv_reread_equal=True, native_coco_all_rows_roundtripped=True,
        native_coco_roundtripped_mask_count=len(rows),
        csv=dict(name=csv_path.name, bytes=csv_path.stat().st_size, sha256=_hash_file(csv_path)),
        server_uploaded=False, server_acceptance_verified=False, completed_at_utc=_now())
    _json(output_dir / "test-coverage-manifest.json", manifest)
    return manifest


def source_manifest():
    paths = {name + ".py": Path(sys.modules[name].__file__)
             for name in ("dataset", "evaluation", "unet_candidate")}
    paths["pipeline.py"] = Path(__file__)
    return {name: _hash_file(path) for name, path in paths.items()}


def create_delivery_zip(bundle_dir, output_dir, archive_path):
    """Package only explicit own model/results files; never recurse over data."""
    bundle_input, output_input, archive_input = map(Path, (bundle_dir, output_dir, archive_path))
    if any(path.is_symlink() for path in (bundle_input, output_input, archive_input)):
        raise ValueError("delivery roots and archive must not be symlinks")
    bundle_root, output_root = bundle_input.resolve(), output_input.resolve()
    archive = archive_input.resolve()
    if bundle_root == output_root or bundle_root in output_root.parents or output_root in bundle_root.parents:
        raise ValueError("model and output roots must be separate")
    if any(archive == root or root in archive.parents for root in (bundle_root, output_root)):
        raise ValueError("archive must be outside both input roots")
    if archive.suffix.lower() != ".zip":
        raise ValueError("delivery archive must have .zip extension")
    completion = _read_json(output_root / "completion.json")
    if completion.get("status") != "LOCAL_PIPELINE_COMPLETE_NO_OFFICIAL_SUBMISSION":
        raise ValueError("only a completed local pipeline can be packaged")
    files = []
    def include(root, relative, prefix):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe delivery relative filename")
        source = root / relative
        if any(part.is_symlink() for part in (source, *source.parents) if part != root.parent):
            raise ValueError("delivery input must not traverse symlinks")
        resolved = source.resolve()
        if root not in resolved.parents or not source.is_file():
            raise ValueError("delivery file is missing or outside its explicit root")
        files.append((source, prefix + "/" + relative.as_posix()))
    for filename in ("weights.pt", "config.json", "training-manifest.json"):
        include(bundle_root, filename, "model")
    output_files = ("completion.json", "development-results.json", "threshold-freeze.json",
        "holdout-results.json", "holdout-results.jsonl", "test-predictions.jsonl", "submission.csv",
        "test-coverage-manifest.json", "runtime-manifest.json", "split-manifest.json", "zero-gt-universe.json")
    for filename in output_files:
        include(output_root, filename, "outputs")
    include(output_root, "figures/figures-manifest.json", "outputs")
    figures = _read_json(output_root / "figures" / "figures-manifest.json")
    for figure in figures["figures"]:
        for entry in figure["files"]:
            filename = entry["name"]
            if Path(filename).name != filename or Path(filename).suffix not in (".png", ".pdf"):
                raise ValueError("only named derived PNG/PDF report figures can be packaged")
            include(output_root, Path("figures") / filename, "outputs")
            source = output_root / "figures" / filename
            if source.stat().st_size != entry["bytes"] or _hash_file(source) != entry["sha256"]:
                raise ValueError("derived report figure differs from completed manifest")
    names = [name for _, name in files]
    if len(names) != len(set(names)):
        raise ValueError("duplicate delivery entry")
    manifest = dict(schema_version=1, files=[dict(name=name, bytes=source.stat().st_size,
        sha256=_hash_file(source)) for source, name in files], source_sha256=source_manifest(),
        freeze_id=completion["freeze_id"], raw_dataset_included=False,
        official_submission_completed=False, created_at_utc=_now(),
        figure_attribution="MAGFiLO / NSO GONG H-alpha; CC BY-NC 4.0; noncommercial competition report",
        packaging_policy="Explicit model and result allowlist only; no input images, annotations, credentials or private audits")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(".zip.partial")
    if temporary.is_symlink():
        raise ValueError("temporary archive must not be a symlink")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as writer:
        for source, name in files:
            writer.write(source, arcname=name)
        writer.writestr("delivery-manifest.json", json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    with zipfile.ZipFile(temporary) as reader:
        if reader.namelist() != names + ["delivery-manifest.json"] or reader.testzip() is not None:
            raise ValueError("delivery archive integrity check failed")
    temporary.replace(archive)
    receipt = dict(path=str(archive), bytes=archive.stat().st_size, sha256=_hash_file(archive),
                   file_count=len(files) + 1, raw_dataset_included=False, official_submission_completed=False)
    print("FINAL_DELIVERY_ZIP", json.dumps(receipt), flush=True)
    return receipt


def runtime_manifest():
    import torch
    distributions = {}
    for name in ("numpy", "scipy", "opencv-python", "opencv-python-headless", "pandas", "torch", "torchvision",
                 "pycocotools", "matplotlib", "Pillow", "joblib", "scikit-learn", "ipython"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=distributions,
                opencv_runtime_version=cv2.__version__, opencv_build_information=cv2.getBuildInformation(),
                torch_cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                source_sha256=source_manifest())


def run_posttraining(annotation_path, image_root, test_root, bundle_dir, output_dir,
                     development_report_path=None, device="cuda", display_figures=True):
    """Resume only: does not call fit, access networks, or upload anything."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = load_coco_index(annotation_path)
    split = partitions(index)
    identity = validate_training_bundle(bundle_dir, split)
    model = UNetCandidate.load_artifacts(bundle_dir, device=device)
    _json(output_dir / "runtime-manifest.json", runtime_manifest())
    _json(output_dir / "split-manifest.json", split)
    zero_gt = sorted(i for i in index.images if not index.segmentations[i])
    _json(output_dir / "zero-gt-universe.json", dict(zero_gt_annotator_image_ids=zero_gt,
        count=len(zero_gt), evaluation_policy="exclude from organizer GT-filament-row universe; record rather than merge annotators"))
    copied_development = output_dir / "development-results.json"
    if development_report_path is not None:
        report = _read_json(development_report_path)
        if copied_development.exists() and _digest(_read_json(copied_development)) != _digest(report):
            raise ValueError("a different development receipt already exists; no replacement")
        if not copied_development.exists():
            _json(copied_development, report)
    elif copied_development.exists():
        report = _read_json(copied_development)
    else:
        report = evaluate_development(model, index, image_root, split, copied_development)
    freeze = freeze_threshold(output_dir, identity, split, report, index)
    print("FINAL_THRESHOLD_FROZEN", json.dumps(freeze), flush=True)
    holdout = evaluate_holdout(model, index, image_root, split, freeze, output_dir, display_figures)
    print("FINAL_HOLDOUT_SUMMARY", json.dumps(holdout), flush=True)
    test = infer_test(model, test_root, freeze, output_dir)
    _json(output_dir / "completion.json", dict(status="LOCAL_PIPELINE_COMPLETE_NO_OFFICIAL_SUBMISSION",
        freeze_id=freeze["freeze_id"], threshold=freeze["selected_threshold"],
        holdout_official_global_pq=holdout["official_global_pq"], holdout_observations=holdout["observation_count"],
        test_images_processed=test["test_images_processed"], csv=test["csv"],
        zero_detection_images=test["zero_detection_images"], official_submission_completed=False,
        completed_at_utc=_now(), train_annotation_sha256=_hash_file(annotation_path)))
    print("FINAL_PIPELINE_COMPLETE", json.dumps(_read_json(output_dir / "completion.json")), flush=True)
    return dict(freeze=freeze, holdout=holdout, test=test)


def train_from_scratch(annotation_path, image_root, bundle_dir, checkpoint_dir, device="cuda"):
    """Explicit optional reproduction; never called by the resume runner."""
    index = load_coco_index(annotation_path)
    split = partitions(index)
    config = UNetConfig(device=device, epochs=24, batch_size=4, min_area=300)
    model = UNetCandidate(config)
    model.fit_coco(index, image_root, split["training"],
        allowed_training_observations=split["combined_train"],
        forbidden_observations=tuple(split["holdout"]) + tuple(split["development"]),
        progress=lambda row: print("UNET_EPOCH", json.dumps(row), flush=True), checkpoint_dir=checkpoint_dir)
    model.save_artifacts(bundle_dir)
    return model
