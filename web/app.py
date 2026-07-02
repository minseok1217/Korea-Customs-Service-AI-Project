from __future__ import annotations

import json
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path("/data/2_data_server/cv-07/dice/the Korea Customs Service")
ROOT = PROJECT / "project/web"
MODEL_ROOT = PROJECT / "project/model"

CUSTOMS_WEB_PY = Path("/data/2_data_server/cv-07/anaconda3/envs/customs_web/bin/python")
YOLO_PY = Path("/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python")

YOLO_REPO = MODEL_ROOT / "object_detection/yolov5"
YOLO_INFER_SCRIPT = MODEL_ROOT / "object_detection/yolo_infer_json.py"
QWEN_SCRIPT = MODEL_ROOT / "text_classification/qwen_text_classification/classify_text.py"
QWEN_MODEL = MODEL_ROOT / "text_classification/qwen3_4b"
DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IOU = 0.45
DEFAULT_YOLO_IMGSZ = 896
DEFAULT_YOLO_DEVICE = "0"
DEFAULT_QWEN_DEVICE = "auto"
DEFAULT_TIMEOUT_SECONDS = 180

LABELS = [
    "송곳",
    "도끼",
    "배터리",
    "탄환",
    "폭죽",
    "총기",
    "총기 부품",
    "망치",
    "수갑",
    "하드디스크",
    "칼",
    "노트북",
    "라이터",
    "액체",
    "성냥",
    "손톱깎이",
    "휴대용 가스",
    "톱",
    "가위",
    "드라이버",
    "스마트폰",
    "고체연료",
    "스패너",
    "SSD(솔리드 스테이트 드라이브)",
    "보조배터리",
    "태블릿PC",
    "USB",
    "펜치",
    "끌",
    "전자담배",
    "전자담배 액상",
    "투척용 칼",
]

CATEGORY_ROWS = [{"id": idx, "name": name, "supercategory": name} for idx, name in enumerate(LABELS, 1)]

DEFAULT_WEIGHT_CANDIDATES = [
    YOLO_REPO / "runs/train/17_super_mapped_yolov5x6_e304/weights/best.pt",
    YOLO_REPO / "runs/train/17_super_mapped_yolov5x6_e303/weights/best.pt",
    YOLO_REPO / "runs/train/231_mapped32_yolov5x6_e30/weights/best.pt",
    YOLO_REPO / "runs/train/231_super_mapped_yolov5x6_e302/weights/best.pt",
    YOLO_REPO / "weights/yolov5x6.pt",
]


st.set_page_config(
    page_title="Korea Customs Service AI Web",
    page_icon=":package:",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; max-width: 1500px; }
    [data-testid="stSidebar"] { min-width: 320px; }
    [data-testid="stFileUploader"] {
        border: 1px dashed #6b7280;
        border-radius: 8px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.03);
    }
    [data-testid="stFileUploader"] section {
        min-height: 220px;
        align-items: center;
    }
    .empty-output {
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px dashed rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def existing_weight_options() -> list[str]:
    options = [str(path) for path in DEFAULT_WEIGHT_CANDIDATES if path.exists()]
    train_dir = YOLO_REPO / "runs/train"
    if train_dir.exists():
        for path in sorted(train_dir.glob("*/weights/*.pt")):
            text = str(path)
            if text not in options:
                options.append(text)
    return options


def read_image(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file).convert("RGB")
    image.load()
    return image


def image_download_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def normalize_label(label: str | None) -> str:
    return (label or "").strip().lower().replace("_", " ")


def top_detection_label(detections: list[dict]) -> tuple[str | None, float | None]:
    if not detections:
        return None, None
    best = max(detections, key=lambda item: float(item.get("confidence", 0)))
    return str(best.get("class")), float(best.get("confidence", 0))


def run_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def run_yolo(
    image: Image.Image,
    suffix: str,
    weights: str,
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
    timeout: int,
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=suffix or ".png", delete=False) as tmp:
        image.save(tmp.name)
        image_path = Path(tmp.name)

    cmd = [
        str(YOLO_PY),
        str(YOLO_INFER_SCRIPT),
        "--repo",
        str(YOLO_REPO),
        "--weights",
        weights,
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
    try:
        completed = run_command(cmd, timeout)
    finally:
        image_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or "YOLOv5 failed.")
    return json.loads(completed.stdout)


def run_qwen(text: str, model_path: str, device: str, timeout: int) -> dict:
    cmd = [
        str(CUSTOMS_WEB_PY),
        str(QWEN_SCRIPT),
        "--model",
        model_path,
        "--labels",
        ",".join(LABELS),
        "--text",
        text,
        "--device",
        device,
    ]
    completed = run_command(cmd, timeout)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout or "Qwen classification failed.")

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Qwen returned no output.")
    return json.loads(lines[-1])


def render_uploader(key: str) -> tuple[str | None, Image.Image | None]:
    version_key = f"{key}_version"
    bytes_key = f"{key}_bytes"
    name_key = f"{key}_name"
    if version_key not in st.session_state:
        st.session_state[version_key] = 0
    uploader_key = f"{key}_{st.session_state[version_key]}"

    if bytes_key not in st.session_state:
        uploaded = st.file_uploader(
            "Drag and drop file here",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
            label_visibility="collapsed",
            key=uploader_key,
        )
        if uploaded is not None:
            st.session_state[bytes_key] = uploaded.getvalue()
            st.session_state[name_key] = uploaded.name
            st.rerun()
        return None, None

    image = Image.open(BytesIO(st.session_state[bytes_key])).convert("RGB")
    image.load()
    uploaded_name = st.session_state.get(name_key)
    if image is not None:
        clear_col, download_col, _ = st.columns([1, 2, 3])
        with clear_col:
            if st.button("Clear image", key=f"{uploader_key}_clear"):
                st.session_state[version_key] += 1
                st.session_state.pop(bytes_key, None)
                st.session_state.pop(name_key, None)
                for result_key in (
                    "combined_detection",
                    "combined_classification",
                    "detect_result",
                ):
                    st.session_state.pop(result_key, None)
                st.rerun()
        with download_col:
            st.download_button(
                "Download original image",
                image_download_bytes(image),
                file_name="original_image.png",
                mime="image/png",
                key=f"{uploader_key}_download",
            )
        st.image(image, use_container_width=True)
    return uploaded_name, image


def render_agreement(detected_label: str | None, classified_label: str | None) -> None:
    if not detected_label or not classified_label:
        st.metric("Final check", "NO RESULT")
        st.warning("Both task results are required.")
        return

    is_match = normalize_label(detected_label) == normalize_label(classified_label)
    st.metric("Final check", "MATCH" if is_match else "MISMATCH")
    if is_match:
        st.success("Object detection and text classification agree.")
    else:
        st.error("Object detection and text classification do not agree.")


def render_category_summary() -> None:
    st.caption(f"{len(LABELS)} categories")
    for start in range(0, len(LABELS), 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx < len(LABELS):
                col.write(f"{idx + 1}. {LABELS[idx]}")


def render_detection_result(image: Image.Image, result: dict) -> tuple[str | None, float | None]:
    detections = result.get("detections", [])
    st.image(draw_detections(image, detections), caption="YOLOv5 detections", use_container_width=True)
    detected_label, confidence = top_detection_label(detections)
    st.metric("Top detected label", detected_label or "None")
    st.metric("Confidence", f"{confidence:.2%}" if confidence is not None else "-")
    with st.expander("Detection JSON"):
        st.json(detections)
    return detected_label, confidence


def run_detection_button(uploaded_name: str | None, image: Image.Image | None, prefix: str) -> dict | None:
    if image is None:
        st.warning("Upload an image first.")
        return None
    if not Path(yolo_weight).exists():
        st.error(f"Weight file not found: {yolo_weight}")
        return None

    if st.button("Run object detection", type="primary", key=f"{prefix}_run_yolo"):
        with st.spinner("Running YOLOv5 in yolov5 env..."):
            return run_yolo(
                image=image,
                suffix=Path(uploaded_name).suffix if uploaded_name else ".png",
                weights=yolo_weight,
                conf=yolo_conf,
                iou=yolo_iou,
                device=yolo_device,
                imgsz=yolo_imgsz,
                timeout=timeout_seconds,
            )
    return None


def run_classification_button(text: str, prefix: str) -> dict | None:
    if not text.strip():
        st.warning("Enter text first.")
        return None

    if st.button("Run text classification", type="primary", key=f"{prefix}_run_qwen"):
        with st.spinner("Running Qwen in customs_web env..."):
            return run_qwen(
                text=text,
                model_path=qwen_model_path,
                device=qwen_device,
                timeout=timeout_seconds,
            )
    return None


def render_combined_task() -> None:
    st.title("Object Detection + Text Classification")
    st.caption("Default setting: detect the object from the image, classify the declared item name, then compare both categories.")

    image_col, text_col = st.columns([1.1, 0.9], gap="large")
    with image_col:
        st.subheader("Original Image")
        uploaded_name, image = render_uploader("combined_image")
    with text_col:
        st.subheader("Text")
        text = st.text_area(
            "Detailed declared item name",
            placeholder="예: 리튬이온 보조배터리, 휴대용 라이터, 스테인리스 주방용 칼\n여러 품명을 쉼표나 줄바꿈으로 입력해도 됩니다.",
            height=260,
            key="combined_text",
        )

    if st.button("Run both tasks", type="primary"):
        if image is None or not text.strip():
            st.warning("Upload an image and enter text first.")
            return
        st.session_state.combined_detection = None
        st.session_state.combined_classification = None
        try:
            with st.spinner("Running YOLOv5 in yolov5 env..."):
                st.session_state.combined_detection = run_yolo(
                    image=image,
                    suffix=Path(uploaded_name).suffix if uploaded_name else ".png",
                    weights=yolo_weight,
                    conf=yolo_conf,
                    iou=yolo_iou,
                    device=yolo_device,
                    imgsz=yolo_imgsz,
                    timeout=timeout_seconds,
                )
        except Exception as exc:
            st.error(f"YOLOv5 failed: {exc}")
        try:
            with st.spinner("Running Qwen in customs_web env..."):
                st.session_state.combined_classification = run_qwen(
                    text=text,
                    model_path=qwen_model_path,
                    device=qwen_device,
                    timeout=timeout_seconds,
                )
        except Exception as exc:
            st.error(f"Qwen failed: {exc}")

    st.divider()
    det_col, cls_col, agree_col = st.columns(3, gap="large")
    detected_label = None
    classified_label = None

    with det_col:
        st.subheader("Object Detection")
        result = st.session_state.get("combined_detection")
        if image is not None and result:
            detected_label, _ = render_detection_result(image, result)
        else:
            st.markdown("<div class='empty-output'>No detection result.</div>", unsafe_allow_html=True)

    with cls_col:
        st.subheader("Text Classification")
        result = st.session_state.get("combined_classification")
        if result:
            classified_label = str(result.get("label", ""))
            st.metric("Classified label", classified_label or "None")
            confidence = result.get("confidence")
            st.metric("Confidence", "-" if confidence is None else str(confidence))
            with st.expander("Qwen raw output"):
                st.json(result)
        else:
            st.markdown("<div class='empty-output'>No classification result.</div>", unsafe_allow_html=True)

    with agree_col:
        st.subheader("Agreement")
        render_agreement(detected_label, classified_label)


def render_detection_task() -> None:
    st.title("Object Detection")
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.subheader("Original Image")
        uploaded_name, image = render_uploader("detect_image")
        try:
            result = run_detection_button(uploaded_name, image, "detect")
            if result:
                st.session_state.detect_result = result
        except Exception as exc:
            st.error(str(exc))

    with right_col:
        st.subheader("Detection Result")
        result = st.session_state.get("detect_result")
        if image is not None and result:
            render_detection_result(image, result)
        else:
            st.markdown("<div class='empty-output'>No detection result.</div>", unsafe_allow_html=True)


def render_classification_task() -> None:
    st.title("Text Classification")
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.subheader("Text")
        text = st.text_area(
            "Detailed declared item name",
            placeholder="예: 리튬이온 보조배터리, 휴대용 라이터, 스테인리스 주방용 칼\n여러 품명을 쉼표나 줄바꿈으로 입력해도 됩니다.",
            height=260,
            key="classify_text",
        )
        try:
            result = run_classification_button(text, "classify")
            if result:
                st.session_state.classify_result = result
        except Exception as exc:
            st.error(str(exc))

    with right_col:
        st.subheader("Classification Result")
        result = st.session_state.get("classify_result")
        if result:
            st.metric("Classified label", str(result.get("label", "")) or "None")
            confidence = result.get("confidence")
            st.metric("Confidence", "-" if confidence is None else str(confidence))
            with st.expander("Qwen raw output"):
                st.json(result)
        else:
            st.markdown("<div class='empty-output'>No classification result.</div>", unsafe_allow_html=True)


weight_options = existing_weight_options()
yolo_weight = weight_options[0] if weight_options else str(YOLO_REPO / "weights/yolov5x6.pt")
yolo_conf = DEFAULT_YOLO_CONF
yolo_iou = DEFAULT_YOLO_IOU
yolo_imgsz = DEFAULT_YOLO_IMGSZ
yolo_device = DEFAULT_YOLO_DEVICE
qwen_model_path = str(QWEN_MODEL)
qwen_device = DEFAULT_QWEN_DEVICE
timeout_seconds = DEFAULT_TIMEOUT_SECONDS

with st.sidebar:
    st.header("Task")
    task = st.radio(
        "Select task",
        [
            "Object Detection + Text Classification",
            "Object Detection",
            "Text Classification",
        ],
        label_visibility="collapsed",
    )
    with st.expander("Text classification categories"):
        render_category_summary()

    st.divider()
    st.caption("Customs AI demo")


if task == "Object Detection":
    render_detection_task()
elif task == "Text Classification":
    render_classification_task()
else:
    render_combined_task()
