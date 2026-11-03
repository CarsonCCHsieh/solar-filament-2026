import copy
import importlib.util
import unittest

import numpy as np

from dataset import ground_truth_masks, index_coco, submission_rows
from evaluation import group_by_observation_split


def payload():
    return {
        "images": [
            {"id": "050103-observation-a", "file_name": "observation-a.jpeg", "width": 16, "height": 16,
             "url": "https://example.invalid/do-not-use", "date_captured": "unused"},
            {"id": "050102-observation-a", "file_name": "observation-a.jpeg", "width": 16, "height": 16},
            {"id": "050101-observation-b", "file_name": "observation-b.jpeg", "width": 16, "height": 16}],
        "annotations": [
            {"id": 1, "image_id": "050103-observation-a", "segmentation": [[1, 1, 7, 1, 7, 4, 1, 4]],
             "category_id": "do-not-train-on-this", "spine": "unused"},
            {"id": 2, "image_id": "050102-observation-a", "segmentation": [[2, 2, 8, 2, 8, 5, 2, 5]]}],
    }


class DatasetTests(unittest.TestCase):
    def test_duplicate_observation_annotation_sets_kept_separate(self):
        index = index_coco(payload())
        self.assertEqual(len(index.observations), 2)
        self.assertEqual(len(index.observations["observation-a.jpeg"]), 2)
        self.assertEqual(len(index.segmentations["050103-observation-a"]), 1)
        self.assertNotEqual(index.segmentations["050103-observation-a"], index.segmentations["050102-observation-a"])
        split = group_by_observation_split([record.file_name for record in index.images.values()])
        train_ids = {image_id for file_name in split.train for image_id in index.observations[file_name]}
        valid_ids = {image_id for file_name in split.validation for image_id in index.observations[file_name]}
        self.assertTrue({"050103-observation-a", "050102-observation-a"} <= train_ids or
                        {"050103-observation-a", "050102-observation-a"} <= valid_ids)

    def test_outside_metadata_not_retained_in_feature_index(self):
        index = index_coco(payload())
        text = repr(index)
        for forbidden in ["example.invalid", "date_captured", "do-not-train-on-this", "spine"]:
            self.assertNotIn(forbidden, text)

    def test_duplicate_image_id_rejected(self):
        value = payload()
        value["images"].append(copy.deepcopy(value["images"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate image id"):
            index_coco(value)

    def test_unknown_annotation_image_rejected(self):
        value = payload()
        value["annotations"][0]["image_id"] = "not-present"
        with self.assertRaisesRegex(ValueError, "unknown image_id"):
            index_coco(value)

    def test_inconsistent_observation_shape_rejected(self):
        value = payload()
        value["images"][1]["width"] = 32
        with self.assertRaisesRegex(ValueError, "dimensions"):
            index_coco(value)

    def test_path_traversal_rejected(self):
        for file_name in ["../private.jpeg", "folder/a.jpeg", "folder\\a.jpeg", ".."]:
            value = payload()
            value["images"][0]["file_name"] = file_name
            with self.subTest(file_name=file_name):
                with self.assertRaises(ValueError):
                    index_coco(value)

    def test_malformed_polygons_rejected(self):
        for polygon in [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5, float("nan")]]:
            value = payload()
            value["annotations"][0]["segmentation"] = [polygon]
            with self.subTest(polygon=polygon):
                with self.assertRaises(ValueError):
                    index_coco(value)

    def test_actual_pycoco_required_without_approximate_fallback(self):
        index = index_coco(payload())
        if importlib.util.find_spec("pycocotools") is None:
            with self.assertRaisesRegex(RuntimeError, "pycocotools"):
                ground_truth_masks(index, "050103-observation-a")
            with self.assertRaisesRegex(RuntimeError, "pycocotools"):
                submission_rows("observation-a.jpeg", [])
        else:
            masks = ground_truth_masks(index, "050103-observation-a")
            self.assertEqual(masks[0].shape, (16, 16))
            self.assertEqual(masks[0].dtype, np.dtype(bool))
            rows = submission_rows("observation-a.jpeg", masks, expected_shape=(16, 16))
            self.assertEqual(set(rows[0]), {"filament_id", "segmentation_rle"})


if __name__ == "__main__":
    unittest.main()
