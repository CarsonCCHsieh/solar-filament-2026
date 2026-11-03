"""Synthetic boundary tests; no competition data or real score is used."""

import unittest

import numpy as np

from evaluation import (
    evaluate_annotators,
    group_by_observation_split,
    instance_pq,
)


def mask(points, shape=(8, 8)):
    value = np.zeros(shape, dtype=bool)
    for y, x in points:
        value[y, x] = True
    return value


class InstancePQTests(unittest.TestCase):
    def setUp(self):
        self.left = mask([(1, 1), (1, 2), (2, 1)])
        self.right = mask([(5, 5), (5, 6)])

    def test_exact_match(self):
        result = instance_pq([self.left], [self.left.copy()])
        self.assertEqual(result.to_dict(), {"pq": 1.0, "sum_iou": 1.0, "tp": 1, "fp": 0, "fn": 0})

    def test_no_predictions(self):
        result = instance_pq([], [self.left])
        self.assertEqual((result.pq, result.tp, result.fp, result.fn), (0.0, 0, 0, 1))

    def test_no_ground_truth(self):
        result = instance_pq([self.left], [])
        self.assertEqual((result.pq, result.tp, result.fp, result.fn), (0.0, 0, 1, 0))

    def test_both_empty_official_zero_denominator(self):
        result = instance_pq([], [], expected_shape=(8, 8))
        self.assertEqual((result.pq, result.tp, result.fp, result.fn), (0.0, 0, 0, 0))

    def test_iou_exactly_half_does_not_match(self):
        target = mask([(1, 1), (1, 2)])
        prediction = mask([(1, 1)])
        result = instance_pq([prediction], [target])
        self.assertEqual((result.pq, result.tp, result.fp, result.fn), (0.0, 0, 1, 1))

    def test_iou_above_half_matches(self):
        prediction = mask([(1, 1), (1, 2)])
        result = instance_pq([prediction], [self.left])
        self.assertAlmostEqual(result.pq, 2 / 3)
        self.assertEqual((result.tp, result.fp, result.fn), (1, 0, 0))

    def test_extra_false_positive_penalty(self):
        result = instance_pq([self.left, self.right], [self.left])
        self.assertAlmostEqual(result.pq, 2 / 3)
        self.assertEqual((result.tp, result.fp, result.fn), (1, 1, 0))

    def test_missed_ground_truth_penalty(self):
        result = instance_pq([self.left], [self.left, self.right])
        self.assertAlmostEqual(result.pq, 2 / 3)
        self.assertEqual((result.tp, result.fp, result.fn), (1, 0, 1))

    def test_all_hits_not_hungarian(self):
        result = instance_pq([self.left, self.left.copy()], [self.left])
        self.assertEqual((result.pq, result.tp, result.fp, result.fn), (1.0, 2, 0, 0))

    def test_every_qualifying_pair_contributes(self):
        result = instance_pq(
            [self.left, self.left.copy()], [self.left.copy(), self.left.copy()]
        )
        self.assertEqual((result.pq, result.sum_iou, result.tp), (1.0, 4.0, 4))

    def test_disjoint_boxes_are_unmatched(self):
        result = instance_pq([self.left], [self.right])
        self.assertEqual((result.pq, result.fp, result.fn), (0.0, 1, 1))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            instance_pq([self.left], [mask([(1, 1)], shape=(9, 8))])

    def test_explicit_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            instance_pq([self.left], [], expected_shape=(2048, 2048))

    def test_empty_instance_rejected(self):
        with self.assertRaises(ValueError):
            instance_pq([np.zeros((8, 8), dtype=bool)], [])

    def test_non_binary_instance_rejected(self):
        bad = self.left.astype(np.uint8)
        bad[0, 0] = 2
        with self.assertRaises(ValueError):
            instance_pq([bad], [])

    def test_nan_instance_rejected(self):
        bad = self.left.astype(float)
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            instance_pq([], [bad])

    def test_non_2d_instance_rejected(self):
        with self.assertRaises(ValueError):
            instance_pq([np.ones((2, 2, 2), dtype=bool)], [])

    def test_unwrapped_2d_array_rejected(self):
        with self.assertRaises(ValueError):
            instance_pq(self.left, [])

    def test_array_batches_accepted(self):
        result = instance_pq(np.stack([self.left, self.right]), [self.left, self.right])
        self.assertEqual(result.pq, 1.0)

    def test_generators_and_uint_binary_accepted(self):
        result = instance_pq(
            (m.astype(np.uint8) for m in [self.left]), (m for m in [self.left])
        )
        self.assertEqual(result.pq, 1.0)

    def test_inputs_are_not_modified(self):
        before = self.left.copy()
        instance_pq([self.left], [self.left])
        np.testing.assert_array_equal(self.left, before)

    def test_invalid_expected_shapes_rejected(self):
        for shape in [(0, 8), (8,), (8, 8, 1), (True, 8), (8.0, 8)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    instance_pq([], [], expected_shape=shape)


class ObservationSplitTests(unittest.TestCase):
    def test_duplicate_annotators_stay_in_one_subset(self):
        observations = ["sun-a", "sun-a", "sun-b", "sun-c", "sun-c", "sun-d"]
        result = group_by_observation_split(observations, 0.5)
        self.assertFalse(set(result.train) & set(result.validation))
        self.assertEqual(set(result.train) | set(result.validation), set(observations))
        self.assertEqual(len(result.train) + len(result.validation), 4)
        for observation in observations:
            self.assertNotEqual(observation in result.train, observation in result.validation)

    def test_deterministic_and_order_independent(self):
        observations = ["a", "b", "b", "c", "d", "e", "f"]
        self.assertEqual(
            group_by_observation_split(observations, seed=41),
            group_by_observation_split(reversed(observations), seed=41),
        )

    def test_fraction_counts_unique_observations(self):
        observations = ["obs-{}".format(index) for index in range(10)]
        observations += ["obs-0"] * 50
        result = group_by_observation_split(observations, 0.3)
        self.assertEqual((len(result.train), len(result.validation)), (7, 3))

    def test_two_observations_keep_both_subsets_nonempty(self):
        result = group_by_observation_split(["a", "b"], 0.001)
        self.assertEqual((len(result.train), len(result.validation)), (1, 1))

    def test_single_observation_with_many_annotators_rejected(self):
        with self.assertRaises(ValueError):
            group_by_observation_split(["same"] * 20)

    def test_invalid_fraction_rejected(self):
        for fraction in [0, 1, -0.1, 1.1, float("nan"), float("inf"), True]:
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    group_by_observation_split(["a", "b"], fraction)

    def test_invalid_observation_ids_rejected(self):
        for ids in [["a", ""], ["a", 2], ["a", "   "]]:
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError):
                    group_by_observation_split(ids)

    def test_invalid_seed_rejected(self):
        with self.assertRaises(ValueError):
            group_by_observation_split(["a", "b"], seed=1.5)


class AnnotatorEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.left = mask([(1, 1), (1, 2)])
        self.right = mask([(5, 5), (5, 6)])

    def test_annotators_compared_separately_not_merged(self):
        result = evaluate_annotators(
            {"sun": [self.left]},
            {"sun": {"annotator-a": [self.left], "annotator-b": [self.right]}},
        )
        self.assertEqual(result.comparison_count, 2)
        self.assertEqual(result.macro_pq, 0.5)
        self.assertEqual(result.by_observation["sun"]["annotator-a"].pq, 1.0)
        self.assertEqual(result.by_observation["sun"]["annotator-b"].pq, 0.0)

    def test_two_aggregation_weightings_are_exposed(self):
        result = evaluate_annotators(
            {"sun-a": [self.left], "sun-b": []},
            {
                "sun-a": {"one": [self.left], "two": [self.right]},
                "sun-b": {"one": []},
            },
        )
        self.assertAlmostEqual(result.macro_pq, 1 / 3)
        self.assertEqual(result.observation_macro_pq, 0.25)
        self.assertEqual(result.official_global_pq, 0.5)
        self.assertEqual(result.global_counts, {"sum_iou": 1.0, "tp": 1, "fp": 1, "fn": 1})
        self.assertEqual(result.to_dict()["comparison_count"], 3)

    def test_official_global_score_aggregates_counts_before_ratio(self):
        result = evaluate_annotators(
            {"sun-a": [self.left, self.right], "sun-b": [self.left]},
            {
                "sun-a": {"one": [self.left, self.right]},
                "sun-b": {"one": [self.right]},
            },
        )
        self.assertAlmostEqual(result.official_global_pq, 2 / 3)
        self.assertEqual(result.macro_pq, 0.5)
        self.assertEqual(result.global_counts, {"sum_iou": 2.0, "tp": 2, "fp": 1, "fn": 1})
        self.assertEqual(result.to_dict()["official_global_pq"], result.official_global_pq)

    def test_official_global_empty_comparisons_return_zero(self):
        result = evaluate_annotators({}, {"sun": {"one": [], "two": []}})
        self.assertEqual(result.official_global_pq, 0.0)
        self.assertEqual(result.global_counts, {"sum_iou": 0.0, "tp": 0, "fp": 0, "fn": 0})
        self.assertEqual(result.macro_pq, 0.0)

    def test_prediction_generator_reused_for_all_annotators(self):
        result = evaluate_annotators(
            {"sun": (m for m in [self.left])},
            {"sun": {"one": [self.left], "two": [self.left]}},
        )
        self.assertEqual(result.macro_pq, 1.0)

    def test_missing_prediction_means_empty_prediction(self):
        result = evaluate_annotators({}, {"sun": {"one": [self.left]}})
        self.assertEqual(result.macro_pq, 0.0)

    def test_unknown_prediction_observation_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_annotators({"typo": []}, {"sun": {"one": [self.left]}})

    def test_no_annotated_observations_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_annotators({}, {})

    def test_observation_without_annotator_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_annotators({}, {"sun": {}})


if __name__ == "__main__":
    unittest.main()

