#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv5 inference and emit JSON detections.")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=896)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.repo))

    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.torch_utils import select_device

    device = select_device(args.device)
    model = DetectMultiBackend(str(args.weights), device=device, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((args.imgsz, args.imgsz), s=stride)

    im0 = cv2.imread(str(args.image))
    if im0 is None:
        raise FileNotFoundError(args.image)

    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = im.copy()
    im = torch.from_numpy(im).to(device)
    im = im.float() / 255.0
    if im.ndimension() == 3:
        im = im[None]

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(
        pred,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        classes=None,
        agnostic=False,
        max_det=args.max_det,
    )

    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det.tolist():
                cls_idx = int(cls)
                cls_name = names[cls_idx] if isinstance(names, (list, tuple)) else names.get(cls_idx, str(cls_idx))
                detections.append(
                    {
                        "class": str(cls_name),
                        "class_id": cls_idx,
                        "confidence": float(conf),
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    }
                )

    print(
        json.dumps(
            {
                "image": str(args.image),
                "width": int(im0.shape[1]),
                "height": int(im0.shape[0]),
                "weights": str(args.weights),
                "detections": detections,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
