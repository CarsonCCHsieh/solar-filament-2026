import unittest

import cv2
import numpy as np

from baseline import BaselineConfig, segment_image, solar_disk_mask


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(self.image, (128, 128), 112, 150, -1)
        self.config = BaselineConfig(limb_margin=8, background_kernel=31,
                                     contrast_threshold=0.10, min_area=20,
                                     max_area_fraction=0.10, min_elongation=1.8)

    def test_uniform_disk_has_no_instances(self):
        result = segment_image(self.image, self.config)
        self.assertEqual(len(result.detections), 0)
        self.assertEqual(result.labels.shape, self.image.shape)

    def test_black_image_has_no_disk_or_instances(self):
        result = segment_image(np.zeros((64, 64), dtype=np.uint8), self.config)
        self.assertEqual(result.diagnostics["disk_area"], 0)
        self.assertEqual(result.masks(), [])

    def test_dark_elongated_instance_is_retained(self):
        self.image[90:156, 99:105] = 70
        result = segment_image(self.image, self.config)
        self.assertEqual(len(result.detections), 1)
        mask = result.masks()[0]
        self.assertTrue(mask[120, 101])
        self.assertGreater(np.count_nonzero(mask & (self.image == 70)), 350)
        self.assertEqual(mask.dtype, np.dtype(bool))

    def test_separate_filaments_are_not_joined(self):
        self.image[90:156, 99:105] = 70
        self.image[90:156, 139:145] = 75
        result = segment_image(self.image, self.config)
        self.assertEqual(len(result.detections), 2)
        self.assertNotEqual(result.labels[120, 101], result.labels[120, 141])
        self.assertEqual(result.labels[120, 123], 0)

    def test_small_fragments_removed(self):
        self.image[120:122, 120:122] = 50
        result = segment_image(self.image, self.config)
        self.assertEqual(len(result.detections), 0)
        self.assertGreaterEqual(result.diagnostics["rejected_small"], 1)

    def test_round_dark_spot_removed(self):
        cv2.circle(self.image, (128, 128), 6, 40, -1)
        result = segment_image(self.image, self.config)
        self.assertEqual(len(result.detections), 0)
        self.assertGreaterEqual(result.diagnostics["rejected_round"], 1)

    def test_gross_component_removed_by_area_limit(self):
        self.image[100:150, 85:165] = 50
        config = BaselineConfig(**{**self.config.to_dict(), "max_area_fraction": 0.001})
        result = segment_image(self.image, config)
        self.assertEqual(len(result.detections), 0)
        self.assertGreaterEqual(result.diagnostics["rejected_large"], 1)

    def test_disk_fills_dark_internal_hole(self):
        self.image[90:156, 99:105] = 0
        mask = solar_disk_mask(self.image, self.config)
        self.assertTrue(mask[120, 101])
        self.assertFalse(mask[0, 0])

    def test_limb_margin_excludes_edge(self):
        mask = solar_disk_mask(self.image, self.config)
        self.assertFalse(mask[128, 17])
        self.assertTrue(mask[128, 40])

    def test_input_contract_rejects_rgb_float_and_empty(self):
        for image in [np.zeros((10, 10, 3), dtype=np.uint8),
                      np.zeros((10, 10), dtype=np.float32),
                      np.zeros((0, 0), dtype=np.uint8)]:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaises(ValueError):
                    segment_image(image, self.config)

    def test_invalid_config_rejected(self):
        for update in [{"background_kernel": 2}, {"contrast_threshold": 0},
                       {"grow_ratio": 1.1}, {"min_area": 0},
                       {"limb_margin": -1}, {"max_area_fraction": 2},
                       {"min_elongation": float("nan")}, {"disk_threshold": True}]:
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    BaselineConfig(**{**self.config.to_dict(), **update})


if __name__ == "__main__":
    unittest.main()
