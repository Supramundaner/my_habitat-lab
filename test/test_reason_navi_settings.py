"""Tests for the canonical navigation settings object."""

from __future__ import annotations

import unittest

from reason_navi.navigation.settings import NavigationSettings


class NavigationSettingsTest(unittest.TestCase):
    def test_defaults_preserve_controller_tuning(self) -> None:
        settings = NavigationSettings.from_config({})
        self.assertEqual(settings.a_star_interval, 5)
        self.assertEqual(settings.intermediate_distance, 1.5)
        self.assertEqual(settings.detection_switch_distance, 1.5)
        self.assertEqual(settings.final_stop_threshold, 0.8)
        self.assertEqual(settings.forward_distance, 0.25)
        self.assertEqual(settings.max_iterations, 500)

    def test_runtime_parameters_are_configurable(self) -> None:
        settings = NavigationSettings.from_config(
            {
                "navigation": {
                    "waypoint_distance": 2.0,
                    "destination_distance": 0.5,
                    "a_star_interval": 3,
                    "a_star_max_iterations": 1000,
                    "detected_target_max_iterations": 25,
                }
            }
        )
        self.assertEqual(settings.intermediate_distance, 2.0)
        self.assertEqual(settings.final_stop_threshold, 0.5)
        self.assertEqual(settings.a_star_interval, 3)
        self.assertEqual(settings.a_star_max_iterations, 1000)
        self.assertEqual(settings.detected_target_max_iterations, 25)

    def test_invalid_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            NavigationSettings.from_config(
                {
                    "navigation": {
                        "min_search_radius": 2.0,
                        "max_search_radius": 1.0,
                    }
                }
            )

    def test_boolean_iteration_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive integer"):
            NavigationSettings.from_config(
                {"navigation": {"a_star_interval": True}}
            )


if __name__ == "__main__":
    unittest.main()
