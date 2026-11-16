import unittest

from eta import estimate, sample_trips, train


class DeliveryModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics = train()

    def test_data_is_reproducible(self):
        self.assertEqual(sample_trips(10), sample_trips(10))

    def test_beats_distance_only_baseline_on_simulated_holdout(self):
        self.assertEqual(self.metrics["holdout_trips"], 400)
        self.assertLess(
            self.metrics["forest_mae_minutes"], self.metrics["distance_baseline_mae_minutes"]
        )

    def test_predicts_more_time_for_heavy_traffic(self):
        light = estimate(self.model, 15, 0, 2, 0)
        heavy = estimate(self.model, 15, 2, 2, 0)
        self.assertGreater(heavy, light)

    def test_rejects_route_outside_simulated_range(self):
        for invalid_distance in (-5, 30, float("nan")):
            with self.subTest(distance=invalid_distance), self.assertRaises(ValueError):
                estimate(self.model, invalid_distance, 1, 0, 0)
