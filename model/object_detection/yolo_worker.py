#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT = Path("/data/2_data_server/cv-07/dice/the Korea Customs Service/project")
YOLO_REPO =  Path("/data/2_data_server/cv-07/dice/the Korea Customs Service/model/yoloV5/1). AI 모델 소스코드/yolov5")
WEIGHT_CANDIDATES = [
    YOLO_REPO / "runs/train/17_super_mapped_yolov5x6_e304/weights/best.pt",
    YOLO_REPO / "runs/train/17_super_mapped_yolov5x6_e303/weights/best.pt",
    YOLO_REPO / "runs/train/231_mapped32_yolov5x6_e30_fresh_save_period1/weights/best.pt",
    YOLO_REPO / "runs/train/231_mapped32_yolov5x6_e30/weights/best.pt",
    YOLO_REPO / "runs/train/231_super_mapped_yolov5x6_e302/weights/best.pt",
    YOLO_REPO / "runs/train/231_super_mapped_no_aerosol32_yolov5x6_e30/weights/best.pt",
]
DEFAULT_WEIGHTS = next((path for path in WEIGHT_CANDIDATES if path.exists()), WEIGHT_CANDIDATES[0])
if not DEFAULT_WEIGHTS.exists():
    raise FileNotFoundError(
        "No YOLO weight file found. Checked: "
        + ", ".join(str(path) for path in WEIGHT_CANDIDATES)
    )
DEFAULT_IMGSZ = 896
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_DEVICE = "0"
HOST = "127.0.0.1"
PORT = 18081

sys.path.insert(0, str(YOLO_REPO))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import check_img_size, non_max_suppression, scale_boxes  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


DEVICE = select_device(DEFAULT_DEVICE)
MODEL = DetectMultiBackend(str(DEFAULT_WEIGHTS), device=DEVICE, fp16=False)
STRIDE, NAMES, PT = MODEL.stride, MODEL.names, MODEL.pt
IMGSZ = check_img_size((DEFAULT_IMGSZ, DEFAULT_IMGSZ), s=STRIDE)
MODEL.warmup(imgsz=(1, 3, *IMGSZ))


def decode_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image.")
    return image


def infer(im0: np.ndarray, conf: float, iou: float, max_det: int) -> dict:
    im = letterbox(im0, IMGSZ, stride=STRIDE, auto=PT)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(DEVICE)
    im = im.float() / 255.0
    if im.ndimension() == 3:
        im = im[None]

    pred = MODEL(im, augment=False, visualize=False)
    pred = non_max_suppression(
        pred,
        conf_thres=conf,
        iou_thres=iou,
        classes=None,
        agnostic=False,
        max_det=max_det,
    )

    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, score, cls in det.tolist():
                cls_idx = int(cls)
                cls_name = NAMES[cls_idx] if isinstance(NAMES, (list, tuple)) else NAMES.get(cls_idx, str(cls_idx))
                detections.append(
                    {
                        "class": str(cls_name),
                        "class_id": cls_idx,
                        "confidence": float(score),
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    }
                )
    return {
        "width": int(im0.shape[1]),
        "height": int(im0.shape[0]),
        "weights": str(DEFAULT_WEIGHTS),
        "detections": detections,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"ok": True, "weights": str(DEFAULT_WEIGHTS), "device": str(DEVICE)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            image = decode_image(payload["image_b64"])
            result = infer(
                image,
                conf=float(payload.get("conf", DEFAULT_CONF)),
                iou=float(payload.get("iou", DEFAULT_IOU)),
                max_det=int(payload.get("max_det", 300)),
            )
            self._send(200, result)
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)


def main() -> None:
    print(f"YOLO worker ready on http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
