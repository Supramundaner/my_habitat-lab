"""Tests for SR/SPL metric normalization and serialization."""

from __future__ import annotations

import json
import unittest

from reason_navi.navigation.metrics import compute_navigation_metrics


class NavigationMetricsTest(unittest.TestCase):
    def test_success_and_spl(self) -> None:
        metrics = compute_navigation_metrics(
            optimal_distance=4.0,
            distance_to_target=0.3,
            path_length=5.0,
            success_threshold=0.5,
        )
        self.assertTrue(metrics.success)
        self.assertTrue(metrics.reachable)
        self.assertEqual(metrics.sr, 1.0)
        self.assertAlmostEqual(metrics.spl, 0.8)
        self.assertTrue(metrics.as_dict()["reachable"])

    def test_configured_threshold_is_used(self) -> None:
        outside = compute_navigation_metrics(
            optimal_distance=2.0,
            distance_to_target=0.3,
            path_length=2.0,
            success_threshold=0.25,
        )
        inside = compute_navigation_metrics(
            optimal_distance=2.0,
            distance_to_target=0.3,
            path_length=2.0,
            success_threshold=0.5,
        )
        self.assertFalse(outside.success)
        self.assertTrue(inside.success)

    def test_unreachable_start_is_not_a_success(self) -> None:
        metrics = compute_navigation_metrics(
            optimal_distance=float("inf"),
            distance_to_target=0.0,
            path_length=1.0,
            success_threshold=0.25,
        )
        self.assertFalse(metrics.success)
        self.assertFalse(metrics.reachable)
        self.assertEqual(metrics.spl, 0.0)
        artifact = metrics.as_dict()
        self.assertIsNone(artifact["optimal_geodesic_distance"])
        self.assertEqual(artifact["geodesic_distance_to_target"], 0.0)
        json.dumps(artifact, allow_nan=False)

    def test_unreachable_final_pose_is_not_a_success(self) -> None:
        metrics = compute_navigation_metrics(
            optimal_distance=1.0,
            distance_to_target=float("inf"),
            path_length=1.0,
            success_threshold=0.25,
        )
        self.assertFalse(metrics.success)
        artifact = metrics.as_dict()
        self.assertFalse(artifact["reachable"])
        self.assertIsNone(artifact["geodesic_distance_to_target"])
        json.dumps(artifact, allow_nan=False)

    def test_every_non_finite_distance_is_json_null(self) -> None:
        metrics = compute_navigation_metrics(
            optimal_distance=float("nan"),
            distance_to_target=float("-inf"),
            path_length=float("inf"),
            success_threshold=0.25,
        )
        artifact = metrics.as_dict()
        self.assertFalse(artifact["reachable"])
        self.assertIsNone(artifact["geodesic_distance_to_target"])
        self.assertIsNone(artifact["optimal_geodesic_distance"])
        self.assertIsNone(artifact["path_length"])
        json.dumps(artifact, allow_nan=False)

    def test_zero_length_success_has_full_spl(self) -> None:
        metrics = compute_navigation_metrics(
            optimal_distance=0.0,
            distance_to_target=0.0,
            path_length=0.0,
            success_threshold=0.25,
        )
        self.assertEqual(metrics.spl, 1.0)

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "success_threshold"):
            compute_navigation_metrics(
                optimal_distance=1.0,
                distance_to_target=0.0,
                path_length=1.0,
                success_threshold=0.0,
            )

    def test_negative_distances_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimal_distance"):
            compute_navigation_metrics(
                optimal_distance=-1.0,
                distance_to_target=0.0,
                path_length=1.0,
                success_threshold=0.25,
            )


if __name__ == "__main__":
    unittest.main()
