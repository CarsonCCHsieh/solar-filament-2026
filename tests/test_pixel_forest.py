"""Reproducible synthetic-only checks for the independent pixel forest.

Run only this module with:
python -B -m unittest discover -s tests -p test_pixel_forest.py -v

All images, masks and COCO records below are invented. The COCO adapter tests
mock decoding and image reads; they do not exercise pycocotools or official data.
No check establishes real segmentation quality or a competition score.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import index_coco
from evaluation import instance_pq
from pixel_forest import (
    PixelForest, PixelForestConfig, _feature_block, feature_halo,
    iter_coco_training, iter_feature_chunks,
)


# Filled by actual checks, and available to a receipt runner without rerunning.
SYNTHETIC_DIAGNOSTICS = {}


def example(offset, seed=775):
    rng = np.random.default_rng(seed + offset)
    image = np.clip(150 + rng.normal(0, 2, (96, 120)), 0, 255).astype(np.uint8)
    mask = np.zeros(image.shape, dtype=bool)
    mask[22 + offset:28 + offset, 15:102] = True
    mask[55:79, 59 + offset:64 + offset] = True
    image[mask] = rng.integers(30, 50, size=int(mask.sum()), dtype=np.uint8)
    return image, mask


def small_config(**changes):
    config = PixelForestConfig(
        n_estimators=8, max_depth=8, n_jobs=2,
        max_training_pixels=4096, samples_per_image=2048,
        min_samples_leaf=2, chunk_rows=13,
        gaussian_sigmas=(1.0, 3.0), blackhat_sizes=(3, 9), std_sizes=(3, 7),
        limb_margin=3, min_area=12, max_area_fraction=1.0,
        min_elongation=1.0, probability_threshold=0.65,
    )
    return replace(config, **changes)


class FeatureChunkTests(unittest.TestCase):
    def test_whole_and_chunked_features_match_with_full_halos(self):
        rng = np.random.default_rng(9981)
        image = rng.integers(8, 240, (389, 117), dtype=np.uint8)
        image[180:210, 15:100] = 30
        configs = [
            (PixelForestConfig(chunk_rows=37), 40),
            (small_config(chunk_rows=11), 9),
            (PixelForestConfig(chunk_rows=37, gaussian_sigmas=(2.0, 4.0),
                               blackhat_sizes=(15, 61)), 60),
            (PixelForestConfig(chunk_rows=37, gaussian_sigmas=(2.0, 4.0),
                               blackhat_sizes=(15, 121)), 120),
        ]
        diagnostics = []
        for config, expected_halo in configs:
            with self.subTest(halo=expected_halo):
                self.assertEqual(feature_halo(config), expected_halo)
                whole = _feature_block(image, 0, image.shape[0], config)
                chunked = np.concatenate([
                    block for _, _, block in iter_feature_chunks(image, config)
                ], axis=0)
                self.assertEqual(whole.dtype, np.dtype("float32"))
                self.assertTrue(np.isfinite(whole).all())
                np.testing.assert_allclose(whole, chunked, rtol=0, atol=2e-5)
                diagnostics.append({
                    "halo": expected_halo, "chunk_rows": config.chunk_rows,
                    "shape": list(image.shape),
                    "max_abs_error": float(np.max(np.abs(whole - chunked))),
                })
        SYNTHETIC_DIAGNOSTICS["feature_halos"] = diagnostics


class ForestInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = small_config()
        cls.pairs = [example(offset) for offset in (0, 3, 6)]
        cls.model = PixelForest(cls.config).fit(iter(cls.pairs[:2]))
        cls.image = cls.pairs[2][0]

    def test_training_cap_stops_without_requesting_next_image(self):
        consumed = []
        def observations():
            for number in range(3):
                consumed.append(number)
                if number == 2:
                    self.fail("training requested an image after reaching its cap")
                yield example(number * 3)
        model = PixelForest(self.config).fit(observations())
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(model.training_summary_["sampled_pixels"], 4096)
        self.assertTrue(model.training_summary_["sample_cap_reached"])
        self.assertGreater(model.training_summary_["sampled_positive"], 0)
        self.assertGreater(model.training_summary_["sampled_negative"], 0)
        SYNTHETIC_DIAGNOSTICS["bounded_training"] = dict(model.training_summary_)

    def test_fit_accepts_parallel_iterables(self):
        image, mask = self.pairs[0]
        model = PixelForest(small_config(n_estimators=4, max_training_pixels=1024,
                                        samples_per_image=1024))
        model.fit(iter([image]), iter([mask]))
        self.assertEqual(model.training_summary_["sampled_pixels"], 1024)

    def test_fit_accepts_single_image_mask_arrays(self):
        model = PixelForest(small_config(n_estimators=4, max_training_pixels=1024,
                                        samples_per_image=1024))
        self.assertIs(model.fit(*self.pairs[0]), model)
        self.assertIsNotNone(model.model_)

    def test_parallel_iterable_length_mismatch_is_rejected(self):
        image, mask = self.pairs[0]
        model = PixelForest(small_config(samples_per_image=1024))
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            model.fit([image], [mask, mask])
        self.assertIsNone(model.model_)

    def test_single_class_training_is_rejected(self):
        model = PixelForest(self.config)
        with self.assertRaisesRegex(ValueError, "foreground and background"):
            model.fit([(self.image, np.zeros(self.image.shape, dtype=bool))])
        self.assertIsNone(model.model_)

    def test_inference_requires_fit(self):
        with self.assertRaisesRegex(RuntimeError, "fit must complete"):
            PixelForest(self.config).predict_labels(self.image)

    def test_reused_probability_matches_direct_labels_at_three_thresholds(self):
        scores = self.model.predict_proba(self.image)
        original = scores.copy()
        self.assertEqual(scores.dtype, np.dtype("float32"))
        counts = {}
        for threshold in (0.45, 0.65, 0.80):
            with self.subTest(threshold=threshold):
                reused = self.model.labels_from_probability(self.image, scores, threshold)
                direct = self.model.predict_labels(self.image, threshold)
                np.testing.assert_array_equal(reused, direct)
                self.assertEqual(reused.dtype, np.dtype("int32"))
                counts[str(threshold)] = int(reused.max())
        np.testing.assert_array_equal(scores, original)
        with patch.object(self.model, "_positive_probability",
                          side_effect=AssertionError("features recomputed")):
            for threshold in (0.45, 0.65, 0.80):
                self.model.labels_from_probability(self.image, scores, threshold)
        SYNTHETIC_DIAGNOSTICS["reused_thresholds"] = {
            "thresholds": [0.45, 0.65, 0.80],
            "exact_label_equality": True,
            "no_feature_recomputation_by_postprocessing": True,
            "input_score_map_unchanged": True,
            "synthetic_instance_counts": counts,
        }

    def test_whole_and_chunked_inference_are_identical(self):
        chunked = self.model.predict_proba(self.image)
        with patch.object(self.model, "config", replace(self.config, chunk_rows=96)):
            whole = self.model.predict_proba(self.image)
        np.testing.assert_array_equal(chunked, whole)
        SYNTHETIC_DIAGNOSTICS["inference_chunk_max_abs_error"] = float(
            np.max(np.abs(chunked - whole)))

    def test_nonempty_boolean_instances_match_labels_and_pq_interface(self):
        labels = self.model.predict_labels(self.image)
        masks = self.model.predict_instances(self.image)
        self.assertGreater(len(masks), 0)
        self.assertEqual(set(np.unique(labels)), set(range(len(masks) + 1)))
        for mask in masks:
            self.assertEqual(mask.dtype, np.dtype("bool"))
            self.assertEqual(mask.shape, self.image.shape)
            self.assertTrue(mask.any())
        np.testing.assert_array_equal(np.logical_or.reduce(masks), labels > 0)
        # Self-comparison checks only the evaluator interface, not model quality.
        score = instance_pq(masks, masks, expected_shape=self.image.shape)
        self.assertEqual(score.tp, len(masks))
        self.assertEqual(score.pq, 1.0)

    def test_invalid_probability_maps_are_rejected(self):
        invalid = [
            np.zeros((3, 4)), np.full(self.image.shape, np.nan),
            np.full(self.image.shape, np.inf), np.full(self.image.shape, -0.01),
            np.full(self.image.shape, 1.01), np.full(self.image.shape, "x"),
            np.full(self.image.shape, 1j),
        ]
        for number, values in enumerate(invalid):
            with self.subTest(case=number), self.assertRaises(ValueError):
                self.model.labels_from_probability(self.image, values)
        SYNTHETIC_DIAGNOSTICS["invalid_probability_map_cases"] = len(invalid)

    def test_bad_image_dtype_and_hard_training_cap_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model.predict_labels(self.image.astype(np.float32))
        with self.assertRaises(ValueError):
            PixelForestConfig(max_training_pixels=524289)

    def test_postprocessing_preserves_disk_and_instance_safety_cap(self):
        image = np.zeros(self.image.shape, dtype=np.uint8)
        labels = self.model.labels_from_probability(image, np.ones(image.shape, np.float32))
        self.assertFalse(labels.any())
        cap = PixelForest(replace(self.config, limb_margin=0, min_area=1,
                                  min_elongation=1, max_instances=2))
        image = np.full((20, 24), 150, dtype=np.uint8)
        scores = np.zeros(image.shape, dtype=np.float32)
        for y, x in ((3, 3), (3, 10), (10, 3)):
            scores[y, x:x + 2] = 1
        with self.assertRaisesRegex(RuntimeError, "safety cap exceeded"):
            cap.labels_from_probability(image, scores)


class MockCocoTrainingTests(unittest.TestCase):
    def setUp(self):
        self.index = index_coco({
            "images": [
                {"id": "a", "file_name": "same.png", "width": 12, "height": 10,
                 "date": "unused", "url": "unused"},
                {"id": "b", "file_name": "same.png", "width": 12, "height": 10},
                {"id": "c", "file_name": "heldout.png", "width": 12, "height": 10},
            ],
            "annotations": [],
        })
        self.one = np.zeros((10, 12), dtype=bool)
        self.one[2:5, 1:4] = True
        self.two = np.zeros((10, 12), dtype=bool)
        self.two[5:8, 7:10] = True

    def test_mock_adapter_unions_train_annotators_without_touching_holdout(self):
        images_seen, ids_seen = [], []
        def read(index, root, observation):
            images_seen.append(observation)
            return np.full((10, 12), 150, dtype=np.uint8)
        def decode(index, image_id):
            ids_seen.append(image_id)
            self.assertNotEqual(image_id, "c")
            return [{"a": self.one, "b": self.two}[image_id]]
        with patch("dataset.read_observation", side_effect=read), \
             patch("dataset.ground_truth_masks", side_effect=decode):
            pairs = list(iter_coco_training(self.index, "unused", ["same.png"]))
        self.assertEqual(images_seen, ["same.png"])
        self.assertEqual(ids_seen, ["a", "b"])
        self.assertEqual(len(pairs), 1)
        np.testing.assert_array_equal(pairs[0][1], self.one | self.two)
        SYNTHETIC_DIAGNOSTICS["coco_adapter"] = {
            "mode": "mock_image_read_and_mask_decode",
            "one_observation_two_annotators_unioned_for_training": True,
            "heldout_not_read": True, "actual_pycocotools_rasterization": False,
        }

    def test_mock_adapter_rejects_duplicate_training_observations(self):
        with patch("dataset.read_observation", return_value=np.full((10, 12), 150, np.uint8)), \
             patch("dataset.ground_truth_masks", return_value=[self.one]):
            with self.assertRaisesRegex(ValueError, "duplicate training observation"):
                list(iter_coco_training(self.index, "unused", ["same.png", "same.png"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
