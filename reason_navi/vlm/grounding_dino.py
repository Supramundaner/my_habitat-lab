# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Optional

import numpy as np
import torch

from .detections import ObjectDetections
from .runtime_config import GROUNDING_DINO_PORT, required_file
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    import torchvision.transforms.functional as F
    from groundingdino.util.inference import load_model, predict
except (ImportError, RuntimeError) as exc:
    F = None
    load_model = None
    predict = None
    _GROUNDING_DINO_IMPORT_ERROR = exc
else:
    _GROUNDING_DINO_IMPORT_ERROR = None

CLASSES = "chair . person . dog ."  # Default classes. Can be overridden at inference.


class GroundingDINO:
    def __init__(
        self,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        caption: str = CLASSES,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: Optional[torch.device] = None,
    ):
        config_file = required_file(
            config_path,
            env_var="GROUNDING_DINO_CONFIG",
            cli_flag="--config",
            description="Grounding DINO model config",
        )
        weights_file = required_file(
            weights_path,
            env_var="GROUNDING_DINO_WEIGHTS",
            cli_flag="--weights",
            description="Grounding DINO model weights",
        )
        if _GROUNDING_DINO_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Grounding DINO server dependencies are unavailable. Install "
                "groundingdino and a compatible torchvision build in the service environment."
            ) from _GROUNDING_DINO_IMPORT_ERROR

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(
            model_config_path=str(config_file),
            model_checkpoint_path=str(weights_file),
        ).to(device)
        self.caption = caption
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(self, image: np.ndarray, caption: Optional[str] = None) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            caption (Optional[str]): A string containing the possible classes
                separated by periods. If not provided, the default classes will be used.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        # Convert image to tensor and normalize from 0-255 to 0-1
        image_tensor = F.to_tensor(image)
        image_transformed = F.normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if caption is None:
            caption_to_use = self.caption
        else:
            caption_to_use = caption
        print("Caption:", caption_to_use)
        with torch.inference_mode():
            boxes, logits, phrases = predict(
                model=self.model,
                image=image_transformed,
                caption=caption_to_use,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
        detections = ObjectDetections(boxes, logits, phrases, image_source=image)

        # Remove detections whose class names do not exactly match the provided classes
        classes = caption_to_use[: -len(" .")].split(" . ")
        detections.filter_by_class(classes)

        return detections


class GroundingDINOClient:
    def __init__(self, port: int = GROUNDING_DINO_PORT):
        self.url = f"http://127.0.0.1:{port}/gdino"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=GROUNDING_DINO_PORT)
    parser.add_argument(
        "--config",
        help="Model config path (or set GROUNDING_DINO_CONFIG)",
    )
    parser.add_argument(
        "--weights",
        help="Model weights path (or set GROUNDING_DINO_WEIGHTS)",
    )
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image, caption=payload["caption"]).to_json()

    gdino = GroundingDINOServer(
        config_path=args.config,
        weights_path=args.weights,
    )
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)
