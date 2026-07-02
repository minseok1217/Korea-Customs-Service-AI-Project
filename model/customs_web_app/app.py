from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path("/data/2_data_server/cv-07/dice/the Korea Customs Service")
MODEL_ROOT = PROJECT / "project/model"
YOLO_BASE = PROJECT / "model/yoloV5/1). AI 모델 소스코드"
YOLO_REPO = YOLO_BASE / "yolov5"
YOLO_PY = Path("/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python")
INFER_SCRIPT = MODEL_ROOT / "customs_web_app/yolo_infer_json.py"

DEFAULT_WEIGHT_CANDIDATES = [
    YOLO_REPO / "runs/train/17_super_mapped_yolov5x6_e30/weights/best.pt",
    YOLO_REPO / "runs/train/17_super_mapped_yolov5x6_e30/weights/last.pt",
    YOLO_REPO / "runs/train/231_super_mapped_yolov5x6_e30/weights/best.pt",
    YOLO_REPO / "runs/train/231_super_mapped_yolov5x6_e30/weights/last.pt",
    YOLO_REPO / "weights/yolov5x6.pt",
    YOLO_REPO / "yolov5x6.pt",
]


def existing_weight_options() -> list[str]:
    options = [str(path) for path in DEFAULT_WEIGHT_CANDIDATES if path.exists()]
    for path in sorted((YOLO_REPO / "runs/train").glob("*/weights/*.pt")):
        text = str(path)
        if text not in options:
            options.append(text)
    return options


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    palette = [
        (255, 64, 64),
        (64, 180, 255),
        (80, 220, 120),
        (255, 190, 70),
        (210, 130, 255),
        (255, 105, 180),
    ]
    for idx, det in enumerate(detections):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        label_h = bottom - top + 4
        label_w = right - left + 6
        y_text = max(0, y1 - label_h)
        draw.rectangle([x1, y_text, x1 + label_w, y_text + label_h], fill=color)
        draw.text((x1 + 3, y_text + 2), label, fill=(0, 0, 0), font=font)
    return out


def run_yolo(image_path: Path, weights: Path, conf: float, iou: float, device: str, imgsz: int) -> dict:
    cmd = [
        str(YOLO_PY),
        str(INFER_SCRIPT),
        "--repo",
        str(YOLO_REPO),
        "--weights",
        str(weights),
        "--image",
        str(image_path),
        "--imgsz",
        str(imgsz),
        "--conf-thres",
        str(conf),
        "--iou-thres",
        str(iou),
        "--device",
        device,
    ]
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)
    return json.loads(completed.stdout)


st.set_page_config(page_title="Customs X-ray Object Detection", layout="wide")
st.title("Customs X-ray Object Detection")

weight_options = existing_weight_options()
with st.sidebar:
    st.subheader("YOLOv5")
    if weight_options:
        selected = st.selectbox("Weights", weight_options, index=0)
        custom_weight = st.text_input("Custom weight path", value=selected)
    else:
        custom_weight = st.text_input("Custom weight path", value=str(YOLO_REPO / "weights/yolov5x6.pt"))
    conf = st.slider("Confidence", min_value=0.01, max_value=0.95, value=0.25, step=0.01)
    iou = st.slider("IoU NMS", min_value=0.1, max_value=0.9, value=0.45, step=0.01)
    imgsz = st.select_slider("Image size", options=[640, 768, 896, 1024, 1280], value=896)
    device = st.text_input("CUDA device", value="0")

uploaded = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded is None:
    st.info("Upload an X-ray image to run detection.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
col_a, col_b = st.columns(2)
with col_a:
    st.image(image, caption="Input", use_container_width=True)

if st.button("Run detection", type="primary"):
    weights = Path(custom_weight)
    if not weights.exists():
        st.error(f"Weight file not found: {weights}")
        st.stop()
    with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix or ".png", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = Path(tmp.name)
    try:
        with st.spinner("Running YOLOv5 inference..."):
            result = run_yolo(tmp_path, weights, conf, iou, device, imgsz)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    finally:
        tmp_path.unlink(missing_ok=True)

    detections = result.get("detections", [])
    with col_b:
        st.image(draw_detections(image, detections), caption="Detections", use_container_width=True)

    st.subheader("Detection JSON")
    st.json(detections)
    if detections:
        st.subheader("Detection Table")
        st.dataframe(pd.DataFrame(detections), use_container_width=True)
    else:
        st.warning("No objects detected.")
