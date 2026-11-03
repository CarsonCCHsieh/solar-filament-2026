"""Tiny, CPU-only behavioral checks; no official data or quality claims."""
import tempfile
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from unet_candidate import (
    UNetCandidate, UNetConfig, SmallUNet, _PairCache, _training_names,
    image_channels, prepare_pair, segmentation_loss,
)


class TestUNetCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, **changes):
        values = dict(input_size=32, channels=(4, 8, 12, 16), epochs=1,
                      batch_size=2, device="cpu", min_area=2,
                      min_elongation=1.0, max_area_fraction=1.0, limb_margin=0)
        values.update(changes)
        return UNetConfig(**values)

    def pair(self):
        image = np.full((48, 64), 160, dtype=np.uint8)
        mask = np.zeros(image.shape, dtype=bool)
        mask[18:22, 10:45] = True
        image[mask] = 60
        return image, mask

    def test_config_guards(self):
        for values in (dict(input_size=33), dict(channels=(16, 32)),
                       dict(epochs=0), dict(cache_bytes=256 * 1024 * 1024 + 1),
                       dict(device="auto"), dict(learning_rate=float("nan"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                UNetConfig(**values)

    def test_split_guard_before_loading(self):
        self.assertEqual(_training_names(["a.jpeg"], ["a.jpeg"], ["b.jpeg"]), ("a.jpeg",))
        for names, allowed, forbidden in ((["a.jpeg"], ["b.jpeg"], []),
                                          (["a.jpeg"], ["a.jpeg"], ["a.jpeg"]),
                                          (["a.jpeg", "a.jpeg"], ["a.jpeg"], []),
                                          (["a.jpeg"], None, [])):
            with self.assertRaises(ValueError):
                _training_names(names, allowed, forbidden)

    def test_prepare_preserves_fractional_thin_mask(self):
        image = np.full((64, 64), 120, dtype=np.uint8)
        mask = np.zeros(image.shape, dtype=bool)
        mask[1, :] = True
        gray, coverage = prepare_pair(image, mask, 32)
        self.assertEqual(gray.shape, (32, 32))
        self.assertEqual(coverage.dtype, np.uint8)
        self.assertTrue(np.any((coverage > 0) & (coverage < 255)))
        channels = image_channels(gray)
        self.assertEqual(channels.shape, (2, 32, 32))
        self.assertEqual(channels.dtype, np.float32)
        self.assertTrue(np.isfinite(channels).all())

    def test_prepare_rejects_invalid_target(self):
        image, mask = self.pair()
        for bad in (mask[:, :-1], np.full(mask.shape, 0.3), np.full(mask.shape, np.nan)):
            with self.assertRaises(ValueError):
                prepare_pair(image, bad, 32)

    def test_bounded_cache_lru_and_no_cache(self):
        seen = []
        def load(name):
            seen.append(name)
            return np.zeros((4, 4), np.uint8), np.zeros((4, 4), np.uint8)
        cache = _PairCache(64)
        for name in ("a", "b", "a", "c"):
            cache.get(name, load)
        self.assertEqual(seen, ["a", "b", "c"])
        self.assertEqual(list(cache.values), ["a", "c"])
        self.assertLessEqual(cache.peak_bytes, 64)
        tiny = _PairCache(0)
        tiny.get("a", load)
        self.assertEqual(tiny.bytes, 0)

    def test_forward_backward_finite(self):
        model = SmallUNet((4, 8, 12, 16))
        logits = model(torch.rand(2, 2, 32, 40))
        self.assertEqual(tuple(logits.shape), (2, 1, 32, 40))
        target = torch.zeros_like(logits)
        target[:, :, 12:14, 6:25] = 1
        loss = segmentation_loss(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_tiny_fit_predict_and_state_dict_roundtrip(self):
        pairs = {"train-a.jpeg": self.pair(), "train-b.jpeg": self.pair()}
        progress = []
        predictor = UNetCandidate(self.config())
        predictor.fit_pairs(pairs, allowed_training_observations=pairs,
                            forbidden_observations=["holdout.jpeg"], progress=progress.append)
        image = pairs["train-a.jpeg"][0]
        score = predictor.predict_proba(image)
        self.assertEqual(score.shape, image.shape)
        self.assertEqual(score.dtype, np.float32)
        self.assertTrue(np.isfinite(score).all())
        self.assertTrue(np.all((0 <= score) & (score <= 1)))
        self.assertEqual(len(progress), 1)
        self.assertEqual(predictor.training_summary_["observation_count"], 2)
        self.assertLessEqual(predictor.training_summary_["cache_peak_bytes"], predictor.config.cache_bytes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-weights.pt"
            predictor.save_weights(path)
            restored = UNetCandidate(self.config()).load_weights(path)
            np.testing.assert_array_equal(score, restored.predict_proba(image))

    def test_probability_contract_and_components(self):
        image, _ = self.pair()
        predictor = UNetCandidate(self.config())
        score = np.zeros(image.shape, dtype=np.float32)
        score[12:15, 5:25] = 0.9
        score[25:28, 30:50] = 0.8
        labels = predictor.labels_from_probability(image, score, 0.7)
        self.assertEqual(labels.dtype, np.int32)
        self.assertEqual(set(np.unique(labels)), {0, 1, 2})
        self.assertEqual(int(predictor.labels_from_probability(image, score, 0.85).max()), 1)
        for bad in (score[:, :-1], score + 1.1, score - 0.1, score * float("nan")):
            with self.assertRaises(ValueError):
                predictor.labels_from_probability(image, bad)

    def test_epoch_checkpoint_and_portable_artifacts(self):
        pairs = {"a.jpeg": self.pair()}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            predictor = UNetCandidate(self.config()).fit_pairs(
                pairs, allowed_training_observations=pairs, checkpoint_dir=checkpoint)
            manifest = json.loads((checkpoint / "training-manifest.json").read_text())
            self.assertEqual(manifest["epochs_completed"], 1)
            self.assertEqual(manifest["observations"], ["a.jpeg"])
            self.assertTrue((checkpoint / "weights.pt").is_file())
            final = Path(directory) / "final"
            predictor.save_artifacts(final)
            restored = UNetCandidate.load_artifacts(final, device="cpu")
            np.testing.assert_array_equal(predictor.predict_proba(pairs["a.jpeg"][0]),
                                          restored.predict_proba(pairs["a.jpeg"][0]))
            self.assertEqual(restored.training_summary_["observation_count"], 1)

    def test_instance_cap_fails_instead_of_truncating(self):
        image, _ = self.pair()
        score = np.zeros(image.shape, dtype=np.float32)
        score[12:15, 5:25] = 0.9
        score[25:28, 30:50] = 0.9
        with self.assertRaises(RuntimeError):
            UNetCandidate(self.config(max_instances=1)).labels_from_probability(image, score)

    def test_unfitted_and_explicit_cuda_guard(self):
        image, _ = self.pair()
        with self.assertRaises(RuntimeError):
            UNetCandidate(self.config()).predict_proba(image)


if __name__ == "__main__":
    unittest.main()
