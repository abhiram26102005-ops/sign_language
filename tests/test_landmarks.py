import unittest

from src.landmarks import FEATURE_COUNT, normalize_landmarks


class LandmarkNormalizationTests(unittest.TestCase):
    def test_normalizes_about_wrist_and_hand_scale(self):
        base = [(index / 20, index / 40, index / 80) for index in range(21)]
        translated_scaled = [
            (x * 4 + 0.3, y * 4 - 0.2, z * 4 + 0.1)
            for x, y, z in base
        ]

        original = normalize_landmarks(base)
        transformed = normalize_landmarks(translated_scaled)

        self.assertEqual(len(original), FEATURE_COUNT)
        for left, right in zip(original, transformed):
            self.assertAlmostEqual(left, right, places=7)

    def test_rejects_wrong_landmark_count(self):
        with self.assertRaises(ValueError):
            normalize_landmarks([(0, 0, 0)])

    def test_returns_none_for_zero_scale(self):
        self.assertIsNone(normalize_landmarks([(0, 0, 0)] * 21))


if __name__ == "__main__":
    unittest.main()
