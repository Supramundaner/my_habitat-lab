from __future__ import annotations

import unittest

from reason_navi.navigation.camera import pinhole_intrinsics


class CameraIntrinsicsTest(unittest.TestCase):
    def test_512_square_90_degree_camera_has_256_pixel_focal_length(self) -> None:
        params = pinhole_intrinsics(512, 512, 90)
        self.assertAlmostEqual(params["fx"], 256.0)
        self.assertAlmostEqual(params["fy"], 256.0)
        self.assertAlmostEqual(params["cx"], 255.5)
        self.assertAlmostEqual(params["cy"], 255.5)

    def test_intrinsics_follow_actual_resolution(self) -> None:
        params = pinhole_intrinsics(640, 480, 90)
        self.assertAlmostEqual(params["fx"], 320.0)
        self.assertAlmostEqual(params["cx"], 319.5)
        self.assertAlmostEqual(params["cy"], 239.5)

    def test_invalid_fov_is_rejected(self) -> None:
        for value in (0, 180, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "FOV"):
                    pinhole_intrinsics(512, 512, value)


if __name__ == "__main__":
    unittest.main()
