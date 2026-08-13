# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from .coco_classes import COCO_CLASSES
from .detections import ObjectDetections
from .runtime_config import YOLOV7_PORT, required_directory, required_file
from .server_wrapper import ServerMixin, host_model, send_request, str_to_image


@lru_cache(maxsize=1)
def _load_yolov7_backend(root: str):
    """Load YOLOv7 only when a local server instance is constructed."""
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        experimental = importlib.import_module("models.experimental")
        datasets = importlib.import_module("utils.datasets")
        general = importlib.import_module("utils.general")
        torch_utils = importlib.import_module("utils.torch_utils")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            f"Could not import YOLOv7 from {root}. Set YOLOV7_ROOT to a valid "
            "YOLOv7 checkout and install its dependencies."
        ) from exc

    expected_root = Path(root).resolve()
    for module in (experimental, datasets, general, torch_utils):
        module_file = Path(module.__file__).resolve()
        if os.path.commonpath((str(expected_root), str(module_file))) != str(expected_root):
            raise RuntimeError(
                f"Imported {module.__name__} from {module_file}, outside YOLOV7_ROOT "
                f"{expected_root}. Remove the conflicting module from sys.path."
            )

    return (
        experimental.attempt_load,
        datasets.letterbox,
        general.check_img_size,
        general.non_max_suppression,
        general.scale_coords,
        torch_utils.TracedModel,
    )


class YOLOv7:
    def __init__(
        self,
        weights: Optional[str] = None,
        yolo_root: Optional[str] = None,
        image_size: int = 640,
        half_precision: bool = True,
    ):
        """Loads the model and saves it to a field."""
        root = required_directory(
            yolo_root,
            env_var="YOLOV7_ROOT",
            cli_flag="--root",
            description="YOLOv7 repository",
        )
        weights_file = required_file(
            weights,
            env_var="YOLOV7_WEIGHTS",
            cli_flag="--weights",
            description="YOLOv7 model weights",
        )
        (
            attempt_load,
            self._letterbox,
            check_img_size,
            self._non_max_suppression,
            self._scale_coords,
            TracedModel,
        ) = _load_yolov7_backend(str(root))
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.half_precision = self.device.type != "cpu" and half_precision
        self.model = attempt_load(str(weights_file), map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(image_size, s=stride)  # check img_size
        self.model = TracedModel(self.model, self.device, self.image_size)
        if self.half_precision:
            self.model.half()  # to FP16

        # Warm-up
        if self.device.type != "cpu":
            dummy_img = torch.rand(1, 3, int(self.image_size * 0.7), self.image_size).to(self.device)
            if self.half_precision:
                dummy_img = dummy_img.half()
            for i in range(3):
                self.model(dummy_img)

    def predict(
        self,
        image: np.ndarray,
        caption: Optional[str] = None,  # 保持接口兼容性，但YOLOv7不使用caption
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[str]] = None,
        agnostic_nms: bool = False,
    ) -> ObjectDetections:
        """
        Outputs bounding box and class prediction data for the given image.
        
        Args:
            image (np.ndarray): An RGB image represented as a numpy array.
            caption (Optional[str]): Caption for compatibility with GroundingDINO interface.
                                   If provided, will filter results to only include these classes.
            conf_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IOU threshold for filtering detections.
            classes (list): List of classes to filter by.
            agnostic_nms (bool): Whether to use agnostic NMS.
        """
        orig_shape = image.shape

        # Preprocess image
        img = cv2.resize(
            image,
            (self.image_size, int(self.image_size * 0.7)),
            interpolation=cv2.INTER_AREA,
        )
        img = self._letterbox(img, new_shape=self.image_size)[0]
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half_precision else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.inference_mode():  # Calculating gradients causes a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = self._non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=classes,
            agnostic=agnostic_nms,
        )[0]
        
        # Handle empty predictions
        if pred is None or len(pred) == 0:
            empty_boxes = torch.empty((0, 4))
            empty_logits = torch.empty((0,))
            empty_phrases = []
            return ObjectDetections(empty_boxes, empty_logits, empty_phrases, image_source=image, fmt="xyxy")
        
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = self._scale_coords(img.shape[2:], pred[:, :4], orig_shape).round()
        pred[:, 0] /= orig_shape[1]
        pred[:, 1] /= orig_shape[0]
        pred[:, 2] /= orig_shape[1]
        pred[:, 3] /= orig_shape[0]
        boxes = pred[:, :4]
        logits = pred[:, 4]
        phrases = [COCO_CLASSES[int(i)] for i in pred[:, 5]]

        detections = ObjectDetections(boxes, logits, phrases, image_source=image, fmt="xyxy")
        
        # 如果提供了caption，过滤结果以保持与GroundingDINO的兼容性
        if caption:
            # 解析caption格式 "class1 . class2 . class3 ."
            caption_classes = caption.replace(" .", "").split(" . ") if caption.endswith(" .") else [caption.strip()]
            caption_classes = [c.strip().lower() for c in caption_classes if c.strip()]
            if caption_classes:
                detections.filter_by_class(caption_classes)

        return detections


class YOLOv7Client:
    def __init__(self, port: int = YOLOV7_PORT):
        self.url = f"http://127.0.0.1:{port}/yolov7"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections:
        """
        预测接口，保持与GroundingDINOClient的兼容性
        
        Args:
            image_numpy: 输入图像
            caption: 类别描述，用于过滤检测结果（可选）
        
        Returns:
            ObjectDetections: 检测结果
        """
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=YOLOV7_PORT)
    parser.add_argument(
        "--root",
        help="YOLOv7 repository path (or set YOLOV7_ROOT)",
    )
    parser.add_argument(
        "--weights",
        help="YOLOv7 weights path (or set YOLOV7_WEIGHTS)",
    )
    args = parser.parse_args()

    print("Loading YOLOv7 model...")

    class YOLOv7Server(ServerMixin, YOLOv7):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            caption = payload.get("caption", "")
            return self.predict(image, caption=caption).to_json()

    yolov7 = YOLOv7Server(weights=args.weights, yolo_root=args.root)
    print("YOLOv7 model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolov7, name="yolov7", port=args.port)
