from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from reason_navi.navigation.contracts import (
    RequestFormat,
    ContractError,
    NavigationRequest,
)


def canonical_request() -> dict:
    return {
        "agent_state": {
            "position": [1, 2, 3],
            "rotation": [0, 2, 0, 2],
        },
        "target_info": {"coordinate": [4, 5], "name": " chair "},
        "wall_mask": "wall_mask.png",
        "map_metadata": "metadata.json",
    }


class NavigationRequestContractTest(unittest.TestCase):
    def test_canonical_action_is_validated_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parsed = NavigationRequest.from_mapping(
                canonical_request(), base_dir=Path(directory)
            )

        self.assertEqual(parsed.input_format, RequestFormat.TARGET_INFO)
        self.assertEqual(parsed.agent_state.position, (1.0, 2.0, 3.0))
        self.assertAlmostEqual(
            math.sqrt(sum(item * item for item in parsed.agent_state.rotation)),
            1.0,
        )
        self.assertEqual(
            parsed.command_groups,
            (
                {
                    "target_info": {
                        "coordinate": [4.0, 5.0],
                        "name": "chair",
                    },
                    "wall_mask": str(Path(directory) / "wall_mask.png"),
                },
            ),
        )
        self.assertEqual(
            parsed.map_metadata, Path(directory) / "metadata.json"
        )

    def test_explicit_artifact_pair_has_precedence(self) -> None:
        parsed = NavigationRequest.from_mapping(
            canonical_request(),
            wall_mask_override="override.png",
            map_metadata_override="override.json",
            base_dir=Path("/tmp/action"),
        )
        self.assertEqual(parsed.wall_mask, Path("/tmp/action/override.png"))
        self.assertEqual(
            parsed.map_metadata, Path("/tmp/action/override.json")
        )

    def test_agent_state_takes_priority_with_legacy_compatibility_state(self) -> None:
        request = canonical_request()
        request["initial_state"] = {"position": [8, 9], "rotation": 45}
        parsed = NavigationRequest.from_mapping(request)
        initial_state, agent_state = parsed.simulator_states()
        self.assertEqual(initial_state, {"position": [8.0, 9.0], "rotation": 45.0})
        self.assertEqual(agent_state["position"], [1.0, 2.0, 3.0])

    def test_zero_quaternion_is_rejected(self) -> None:
        request = canonical_request()
        request["agent_state"]["rotation"] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ContractError, "non-zero quaternion"):
            NavigationRequest.from_mapping(request)

    def test_non_finite_coordinate_is_rejected(self) -> None:
        request = canonical_request()
        request["target_info"]["coordinate"] = [float("inf"), 0]
        with self.assertRaisesRegex(ContractError, "finite number"):
            NavigationRequest.from_mapping(request)

    def test_missing_pose_is_rejected(self) -> None:
        request = canonical_request()
        request.pop("agent_state")
        with self.assertRaisesRegex(ContractError, "agent_state or initial_state"):
            NavigationRequest.from_mapping(request)

    def test_historical_format_precedence_is_explicit(self) -> None:
        request = canonical_request()
        request["action"] = [{"target": "chair"}]
        request["sequence"] = ["move_forward"]
        parsed = NavigationRequest.from_mapping(request)
        self.assertEqual(parsed.input_format, RequestFormat.LEGACY_ACTION)


if __name__ == "__main__":
    unittest.main()
