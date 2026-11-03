"""New final-pipeline contracts only; no data/weights/COCO decoder substitutes."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import CocoIndex, ImageRecord
from unet_candidate import UNetConfig
from pipeline import (THRESHOLDS, _append, _digest, _journal, _gt_universe, discover_test_images,
                      bootstrap_observation_pq, create_delivery_zip, freeze_threshold, partitions, validate_structural_rows)


def synthetic_index():
    images, observations, segmentations = {}, {}, {}
    for i in range(707):
        name, image_id = "synthetic%04d.jpeg" % i, "ann-synthetic%04d" % i
        images[image_id] = ImageRecord(image_id, name, 2048, 2048)
        observations[name] = (image_id,)
        segmentations[image_id] = ("unused-synthetic-placeholder-not-decoded",)
    return CocoIndex(images, observations, segmentations)


class FinalPipelineContracts(unittest.TestCase):
    def setUp(self):
        self.index = synthetic_index()
        self.split = partitions(self.index)
        self.config = UNetConfig(device="cuda", epochs=24, batch_size=4, min_area=300).to_dict()
        self.identity = dict(weights_sha256="unit-test-identity-no-weight-file",
            config_sha256="unit-test-config", training_manifest_sha256="unit-test-manifest",
            split_sha256="unit-test-split", config=self.config, source_sha256={"pipeline.py": "test-source"})

    def report(self, pq_by_threshold=(0.2, 0.8, 0.5)):
        records = []
        for name in self.split["development"]:
            image_id = self.index.observations[name][0]
            records.append(dict(observation=name, thresholds={str(t): dict(comparisons={image_id:
                dict(sum_iou=score, tp=1, fp=0, fn=0, pq=score)}) for t, score in zip(THRESHOLDS, pq_by_threshold)}))
        return dict(training=self.split["training"], development=self.split["development"],
                    holdout_reserved=141, config=self.config, records=records)

    def test_partition_558_8_141_disjoint_and_order_independent(self):
        self.assertEqual([len(self.split[n]) for n in ("training", "development", "holdout")], [558, 8, 141])
        self.assertFalse(set(self.split["training"]) & set(self.split["development"]))
        self.assertFalse(set(self.split["combined_train"]) & set(self.split["holdout"]))
        self.index.images = dict(reversed(list(self.index.images.items())))
        self.assertEqual(partitions(self.index), self.split)

    def test_freeze_uses_dev_counts_and_rejects_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            report = self.report()
            frozen = freeze_threshold(directory, self.identity, self.split, report, self.index)
            self.assertEqual(frozen["selected_threshold"], 0.65)
            self.assertEqual(freeze_threshold(directory, self.identity, self.split, report, self.index), frozen)
            with self.assertRaises(ValueError):
                freeze_threshold(directory, self.identity, self.split, self.report((0.9, 0.2, 0.1)), self.index)

    def test_threshold_tie_policy_is_predefined_order(self):
        with tempfile.TemporaryDirectory() as directory:
            frozen = freeze_threshold(directory, self.identity, self.split, self.report((0, 0, 0)), self.index)
            self.assertEqual(frozen["selected_threshold"], 0.45)

    def test_development_receipt_training_summary_matches_model_bundle(self):
        report = self.report()
        report["training_summary"] = dict(epochs=24, marker="actual-bundle-fixture")
        identity = dict(self.identity, training_summary_sha256=_digest(report["training_summary"]))
        with tempfile.TemporaryDirectory() as directory:
            freeze_threshold(directory, identity, self.split, report, self.index)
        report["training_summary"]["marker"] = "different-training-run"
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            freeze_threshold(directory, identity, self.split, report, self.index)

    def test_official_gt_row_universe_excludes_zero_gt(self):
        name = self.split["development"][0]
        image_id = self.index.observations[name][0]
        self.index.images["empty-ann"] = ImageRecord("empty-ann", name, 2048, 2048)
        self.index.observations[name] = (image_id, "empty-ann")
        self.index.segmentations["empty-ann"] = ()
        self.assertEqual(_gt_universe(self.index, name), ([image_id], ["empty-ann"]))
        with tempfile.TemporaryDirectory() as directory:
            frozen = freeze_threshold(directory, self.identity, self.split, self.report(), self.index)
            self.assertEqual(frozen["zero_gt_sets_excluded"], ["empty-ann"])

    def test_development_receipt_cannot_contain_holdout_names(self):
        report = self.report()
        report["development"] = [self.split["holdout"][0]] + self.split["development"][1:]
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            freeze_threshold(directory, self.identity, self.split, report, self.index)

    def test_row_contract_membership_unique_suffix_and_empty_policy(self):
        valid = {"imageA", "imageB"}
        row = dict(filament_id="imageA_unique_tail", segmentation_rle="opaque-counts-not-decoded")
        self.assertEqual(validate_structural_rows([row], valid), {"imageA_unique_tail"})
        self.assertEqual(validate_structural_rows([], valid), set())
        for rows in ([row, row], [dict(row, filament_id="unknown_1")],
                     [dict(row, filament_id="imageA_")], [dict(row, segmentation_rle='"quoted"')],
                     [dict(row, segmentation_rle="")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_structural_rows(rows, valid)

    def test_resume_journal_bound_to_freeze_and_unique_known_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            _append(path, dict(context_id="frozen-A", observation="a.jpeg"))
            self.assertEqual(set(_journal(path, "frozen-A", {"a.jpeg"})), {"a.jpeg"})
            with self.assertRaises(ValueError):
                _journal(path, "frozen-B", {"a.jpeg"})
            with self.assertRaises(ValueError):
                _journal(path, "frozen-A", {"b.jpeg"})
            _append(path, dict(context_id="frozen-A", observation="a.jpeg"))
            with self.assertRaises(ValueError):
                _journal(path, "frozen-A", {"a.jpeg"})

    def test_test_coverage_count_and_ambiguous_stem(self):
        with tempfile.TemporaryDirectory() as directory:
            # These filenames are directory fixtures only, not image data.
            (Path(directory) / "known.jpeg").touch()
            self.assertEqual(len(discover_test_images(directory, expected_count=1)), 1)
            with self.assertRaises(ValueError):
                discover_test_images(directory, expected_count=180)
            (Path(directory) / "bad_stem.jpeg").touch()
            with self.assertRaises(ValueError):
                discover_test_images(directory, expected_count=2)

    def test_bootstrap_uses_observation_clusters_and_stays_deterministic(self):
        records = [dict(observation="a", counts=dict(sum_iou=2.0, tp=2, fp=0, fn=0)),
                   dict(observation="b", counts=dict(sum_iou=0.0, tp=0, fp=2, fn=2))]
        interval = bootstrap_observation_pq(records)
        self.assertEqual(interval, bootstrap_observation_pq(records))
        self.assertEqual(interval["interval_95"], [0.0, 1.0])
        self.assertTrue(interval["exploratory"])
        self.assertFalse(interval["inference_rerun"])
        self.assertEqual(interval["observation_count"], 2)
        with self.assertRaises(ValueError):
            bootstrap_observation_pq(records + [records[0]])

    def test_delivery_allowlist_and_actual_zip_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, outputs = root / "model-fixture", root / "outputs-fixture"
            model.mkdir()
            outputs.mkdir()
            # Byte fixtures are not model weights, image data, or actual results.
            for filename in ("weights.pt", "config.json", "training-manifest.json"):
                (model / filename).write_bytes(("synthetic-fixture:" + filename).encode())
            for filename in ("development-results.json", "threshold-freeze.json", "holdout-results.json",
                "holdout-results.jsonl", "test-predictions.jsonl", "submission.csv", "test-coverage-manifest.json",
                "runtime-manifest.json", "split-manifest.json", "zero-gt-universe.json"):
                (outputs / filename).write_text("synthetic-test-fixture", encoding="utf-8")
            (outputs / "completion.json").write_text(json.dumps(dict(
                status="LOCAL_PIPELINE_COMPLETE_NO_OFFICIAL_SUBMISSION", freeze_id="unit-test-only")), encoding="utf-8")
            (outputs / "figures").mkdir()
            (outputs / "figures" / "figures-manifest.json").write_text('{"figures": []}', encoding="utf-8")
            (model / "not-allowed.txt").write_text("excluded-fixture", encoding="utf-8")
            (outputs / "unrelated-private-fixture.txt").write_text("excluded-fixture", encoding="utf-8")
            archive = root / "delivery.zip"
            receipt = create_delivery_zip(model, outputs, archive)
            self.assertEqual(receipt["sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
            with zipfile.ZipFile(archive) as reader:
                self.assertEqual(len(reader.namelist()), receipt["file_count"])
                self.assertNotIn("model/not-allowed.txt", reader.namelist())
                self.assertNotIn("outputs/unrelated-private-fixture.txt", reader.namelist())
                manifest = json.loads(reader.read("delivery-manifest.json"))
                self.assertFalse(manifest["raw_dataset_included"])
                for entry in manifest["files"]:
                    content = reader.read(entry["name"])
                    self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
                    self.assertEqual(len(content), entry["bytes"])
            with self.assertRaises(ValueError):
                create_delivery_zip(model, outputs, outputs / "recursive.zip")


if __name__ == "__main__":
    unittest.main()
