from __future__ import annotations

import base64
import html
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
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
YOLO_CANDIDATE_CONF = 0.01
YOLO_CANDIDATE_NMS_IOU = 0.90
BOTTOM_STRIP_FILTER_SIZE = (1920, 1080)
BOTTOM_STRIP_FILTER_PX = 80
YOLO_WORKER_URL = "http://127.0.0.1:18081/infer"
QWEN_WORKER_URL = "http://127.0.0.1:18082/classify"
LLAMA_WORKER_URL = "http://127.0.0.1:18083/classify"
TEXT_MODEL_WORKER_URLS = {
    "Qwen": QWEN_WORKER_URL,
    "Llama": LLAMA_WORKER_URL,
}

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

QWEN_LABEL_VERSIONS = {
    "DG 버전": LABELS,
    "관세청 버전": [
        "도검류",
        "총기류",
        "조준경",
        "총기부품",
        "보이스피싱",
        "비정형농산물",
        "CBD오일",
        "스프레이",
        "유리실린더",
        "안경케이스",
        "플라스틱병",
        "플라스틱통",
        "노트북",
        "휴대폰",
        "금속통",
        "금속병",
        "숟가락",
        "구두금속",
        "텀블러",
        "지퍼",
        "체인백",
        "키보드",
        "데스크탑",
        "시계",
        "전선",
        "골프채",
        "휴대용 배터리",
        "하드디스크",
        "전동 드라이버",
        "금속접시",
        "조준경 오탐제거용",
        "멀티툴 오탐제거용",
        "부엌칼 오탐제거용",
        "총알 오탐제거용",
        "소총 오탐제거용",
        "농산물 오탐제거용",
        "전기충격기 오탐제거용",
    ],
}

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
    [data-testid="stTextArea"] textarea {
        font-size: 1.15rem;
        line-height: 1.65;
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
    .agreement-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
    }
    .agreement-table th,
    .agreement-table td {
        padding: 0.65rem 0.75rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        text-align: left;
    }
    .agreement-table th {
        color: #64748b;
        font-weight: 700;
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
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    palette = [
        (255, 0, 0),
        (0, 180, 255),
        (0, 220, 90),
        (255, 180, 0),
        (210, 80, 255),
        (255, 70, 170),
        (255, 255, 0),
    ]
    for idx, det in enumerate(detections):
        color = palette[idx % len(palette)]
        raw_bbox = det.get("bbox", [])
        if len(raw_bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
        x1 = max(0, min(out.width - 1, x1))
        y1 = max(0, min(out.height - 1, y1))
        x2 = max(0, min(out.width - 1, x2))
        y2 = max(0, min(out.height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        label = f"#{idx} {det['class']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
        left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
        label_h = bottom - top + 8
        label_w = right - left + 10
        y_text = max(0, y1 - label_h)
        if x1 + label_w > out.width:
            x1 = max(0, out.width - label_w)
        draw.rectangle([x1, y_text, x1 + label_w, y_text + label_h], fill=color)
        draw.text((x1 + 5, y_text + 4), label, fill=(0, 0, 0), font=font)
    return out


def normalize_label(label: str | None) -> str:
    return (label or "").strip().lower().replace("_", " ")


def top_detection_label(detections: list[dict]) -> tuple[str | None, float | None]:
    if not detections:
        return None, None
    best = max(detections, key=lambda item: float(item.get("confidence", 0)))
    return str(best.get("class")), float(best.get("confidence", 0))


def bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def apply_local_nms(detections: list[dict], iou_threshold: float) -> list[dict]:
    by_class: dict[str, list[dict]] = {}
    for det in detections:
        by_class.setdefault(str(det.get("class", "")), []).append(det)

    kept: list[dict] = []
    for class_detections in by_class.values():
        candidates = sorted(class_detections, key=lambda item: float(item.get("confidence", 0)), reverse=True)
        class_kept: list[dict] = []
        for det in candidates:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            if all(bbox_iou_xyxy(bbox, kept_det.get("bbox", [])) <= iou_threshold for kept_det in class_kept):
                class_kept.append(det)
        kept.extend(class_kept)
    return sorted(kept, key=lambda item: float(item.get("confidence", 0)), reverse=True)


def is_bottom_strip_detection(det: dict, image_size: tuple[int, int] | None) -> bool:
    if image_size != BOTTOM_STRIP_FILTER_SIZE:
        return False
    bbox = det.get("bbox", [])
    if len(bbox) != 4:
        return False
    _, y1, _, y2 = [float(value) for value in bbox]
    box_center_y = (y1 + y2) / 2.0
    return box_center_y >= image_size[1] - BOTTOM_STRIP_FILTER_PX


def filter_detection_result(
    result: dict | None,
    conf_threshold: float,
    iou_threshold: float,
    image_size: tuple[int, int] | None = None,
) -> dict | None:
    if not result:
        return None
    filtered = dict(result)
    candidates = result.get("candidate_detections", result.get("detections", []))
    candidates = [
        det
        for det in candidates
        if float(det.get("confidence", 0.0)) >= float(conf_threshold)
        and not is_bottom_strip_detection(det, image_size)
    ]
    filtered["candidate_detections"] = candidates
    filtered["detections"] = apply_local_nms(candidates, iou_threshold)
    filtered["conf_threshold"] = conf_threshold
    filtered["nms_iou"] = iou_threshold
    filtered["candidate_nms_iou"] = result.get("candidate_nms_iou", YOLO_CANDIDATE_NMS_IOU)
    return filtered


def run_command(cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)


def post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Worker is not reachable at {url}: {exc.reason}") from exc


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
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return post_json(
        YOLO_WORKER_URL,
        {
            "image_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "conf": conf,
            "iou": iou,
            "max_det": 300,
        },
        timeout=timeout,
    )


def run_yolo_subprocess(
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


def run_text_classifier(text: str, model_name: str, timeout: int, labels: list[str]) -> dict:
    url = TEXT_MODEL_WORKER_URLS[model_name]
    return post_json(
        url,
        {
            "text": text,
            "labels": ",".join(labels),
        },
        timeout=timeout,
    )


def run_qwen_subprocess(text: str, model_path: str, device: str, timeout: int, labels: list[str]) -> dict:
    cmd = [
        str(CUSTOMS_WEB_PY),
        str(QWEN_SCRIPT),
        "--model",
        model_path,
        "--labels",
        ",".join(labels),
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
            if st.button("X Clear image", key=f"{uploader_key}_clear"):
                st.session_state[version_key] += 1
                st.session_state.pop(bytes_key, None)
                st.session_state.pop(name_key, None)
                for result_key in (
                    "combined_detection",
                    "combined_detection_conf",
                    "combined_detection_iou",
                    "combined_detection_candidates",
                    "combined_classification",
                    "detect_result",
                    "detect_result_conf",
                    "detect_result_iou",
                    "detect_result_candidates",
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


def get_classification_counts(result: dict | None) -> dict[str, int]:
    if not result:
        return {}
    counts = result.get("counts")
    if isinstance(counts, dict):
        parsed = {}
        for label, value in counts.items():
            try:
                count = int(value)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                parsed[str(label)] = count
        return parsed
    label = result.get("label")
    return {str(label): 1} if label else {}


def get_detection_counts(result: dict | None) -> dict[str, int]:
    if not result:
        return {}
    counts: dict[str, int] = {}
    for det in result.get("detections", []):
        label = str(det.get("class", "")).strip()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts


def render_count_lines(counts: dict[str, int], empty_text: str = "No result.") -> None:
    if not counts:
        st.markdown(f"<div class='empty-output'>{empty_text}</div>", unsafe_allow_html=True)
        return
    for label, count in counts.items():
        st.write(f"{label} : {count}")


def render_conf_slider(key: str) -> float:
    return float(
        st.slider(
            "Detection confidence",
            min_value=0.01,
            max_value=0.95,
            value=float(DEFAULT_YOLO_CONF),
            step=0.01,
            key=key,
        )
    )


def render_agreement(detection_counts: dict[str, int], classified_counts: dict[str, int]) -> None:
    if not detection_counts or not classified_counts:
        st.metric("Final check", "NO RESULT")
        st.caption("Both task results are required.")
        return

    normalized_detection = {normalize_label(label): (label, count) for label, count in detection_counts.items()}
    normalized_classification = {normalize_label(label): (label, count) for label, count in classified_counts.items()}
    all_keys = sorted(set(normalized_detection) | set(normalized_classification))

    rows = []
    exact_match = bool(all_keys)
    any_label_overlap = False
    for key in all_keys:
        det_label, det_count = normalized_detection.get(key, ("", 0))
        cls_label, cls_count = normalized_classification.get(key, ("", 0))
        label = det_label or cls_label
        if det_count > 0 and cls_count > 0:
            any_label_overlap = True
        if det_count == cls_count and det_count > 0:
            status = "MATCH"
            color = "#15803d"
            bg = "rgba(21, 128, 61, 0.14)"
        elif cls_count > det_count:
            status = "TEXT > OBJECT"
            color = "#b45309"
            bg = "rgba(180, 83, 9, 0.14)"
            sort_order = 1
            exact_match = False
        else:
            status = "OBJECT > TEXT"
            color = "#b91c1c"
            bg = "rgba(185, 28, 28, 0.14)"
            sort_order = 2
            exact_match = False
        if det_count == cls_count and det_count > 0:
            sort_order = 0
        rows.append((sort_order, label, det_count, cls_count, status, color, bg))

    st.metric("Final check", "MATCH" if exact_match and any_label_overlap else "MISMATCH")
    rows = sorted(rows, key=lambda row: (row[0], normalize_label(row[1])))
    table_rows = "\n".join(
        f"""
        <tr style="background:{bg};">
            <td>{html.escape(str(label))}</td>
            <td>{det_count}</td>
            <td>{cls_count}</td>
        </tr>
        """
        for _, label, det_count, cls_count, status, color, bg in rows
    )
    st.markdown(
        f"""
        <table class="agreement-table">
            <thead>
                <tr>
                    <th>Label</th>
                    <th>Object Detection</th>
                    <th>Text Classification</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_category_summary(labels: list[str]) -> None:
    st.caption(f"{len(labels)} categories")
    for start in range(0, len(labels), 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx < len(labels):
                col.write(f"{idx + 1}. {labels[idx]}")


def render_detection_result(image: Image.Image, result: dict) -> dict[str, int]:
    detections = result.get("detections", [])
    candidate_detections = result.get("candidate_detections", detections)
    conf_threshold = result.get("conf_threshold")
    nms_iou = result.get("nms_iou")
    candidate_count = len(candidate_detections)
    caption = "YOLOv5 detections"
    if conf_threshold is not None:
        caption = f"YOLOv5 detections | confidence threshold {float(conf_threshold):.2f}"
    st.image(draw_detections(image, detections), caption=caption, use_container_width=True)
    st.caption(f"Showing {len(detections)} boxes from {candidate_count} stored candidate boxes.")
    detection_counts = get_detection_counts(result)
    render_count_lines(detection_counts, empty_text="No detection result.")
    with st.expander("Detection raw output"):
        st.markdown("Filtered detections")
        st.json(detections)
        st.markdown("Stored candidate detections")
        st.json(candidate_detections)
    return detection_counts


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
                conf=YOLO_CANDIDATE_CONF,
                iou=YOLO_CANDIDATE_NMS_IOU,
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
        with st.spinner(f"Running {text_model_name} in customs_web env..."):
            return run_text_classifier(
                text=text,
                timeout=timeout_seconds,
                labels=qwen_labels,
                model_name=text_model_name,
            )
    return None


def render_combined_task() -> None:
    st.title("AI 기반 신고-실물 품목 대조 시스템")
    # st.caption("Default setting: detect the object from the image, classify the declared item name, then compare both categories.")

    image_col, text_col = st.columns([1.1, 0.9], gap="large")
    with image_col:
        st.subheader("Original Image")
        uploaded_name, image = render_uploader("combined_image")
    with text_col:
        title_col, model_col, version_col = st.columns([1.0, 1.0, 1.2], gap="small")
        with title_col:
            st.subheader("Text")
        with model_col:
            text_model_name = st.radio(
                "Text model",
                list(TEXT_MODEL_WORKER_URLS.keys()),
                horizontal=True,
                key="combined_text_model",
            )
        with version_col:
            qwen_label_version = st.radio(
                "Qwen class version",
                list(QWEN_LABEL_VERSIONS.keys()),
                horizontal=True,
                key="combined_qwen_label_version",
            )
        qwen_labels = QWEN_LABEL_VERSIONS[qwen_label_version]
        text = st.text_area(
            "Detailed declared item name",
            placeholder="Example: lithium ion power bank, portable lighter, stainless kitchen knife\nMultiple item names can be separated by commas or new lines.",
            height=260,
            key="combined_text",
        )
        with st.expander("Text classification categories"):
            render_category_summary(qwen_labels)

    if st.button("Run both tasks", type="primary"):
        if image is None or not text.strip():
            st.warning("Upload an image and enter text first.")
            return
        st.session_state.combined_detection = None
        st.session_state.combined_detection_iou = None
        st.session_state.combined_detection_candidates = None
        st.session_state.combined_classification = None
        try:
            with st.spinner("Running YOLOv5 in yolov5 env..."):
                detection_result = run_yolo(
                    image=image,
                    suffix=Path(uploaded_name).suffix if uploaded_name else ".png",
                    weights=yolo_weight,
                    conf=YOLO_CANDIDATE_CONF,
                    iou=YOLO_CANDIDATE_NMS_IOU,
                    device=yolo_device,
                    imgsz=yolo_imgsz,
                    timeout=timeout_seconds,
                )
                detection_result["candidate_detections"] = detection_result.get("detections", [])
                detection_result["candidate_nms_iou"] = YOLO_CANDIDATE_NMS_IOU
                st.session_state.combined_detection_candidates = detection_result
        except Exception as exc:
            st.error(f"YOLOv5 failed: {exc}")
        try:
            with st.spinner(f"Running {text_model_name} in customs_web env..."):
                st.session_state.combined_classification = run_text_classifier(
                    text=text,
                    timeout=timeout_seconds,
                    labels=qwen_labels,
                    model_name=text_model_name,
                )
        except Exception as exc:
            st.error(f"Qwen failed: {exc}")

    st.divider()
    det_col, cls_col, agree_col = st.columns(3, gap="large")
    detection_counts: dict[str, int] = {}
    classified_counts: dict[str, int] = {}

    with det_col:
        st.subheader("객체탐지기")
        combined_yolo_conf = render_conf_slider("combined_conf_threshold")
        result = filter_detection_result(
            st.session_state.get("combined_detection_candidates"),
            combined_yolo_conf,
            DEFAULT_YOLO_IOU,
            image.size if image is not None else None,
        )
        st.session_state.combined_detection = result
        st.session_state.combined_detection_conf = combined_yolo_conf
        st.session_state.combined_detection_iou = DEFAULT_YOLO_IOU
        if image is not None and result:
            detection_counts = render_detection_result(image, result)
        else:
            st.markdown("<div class='empty-output'>No detection result.</div>", unsafe_allow_html=True)

    with cls_col:
        st.subheader("신고품목 분류기")
        result = st.session_state.get("combined_classification")
        if result:
            classified_counts = get_classification_counts(result)
            render_count_lines(classified_counts)
            with st.expander("Qwen raw output"):
                st.json(result)
        else:
            st.markdown("<div class='empty-output'>No classification result.</div>", unsafe_allow_html=True)

    with agree_col:
        st.subheader("Agreement")
        render_agreement(detection_counts, classified_counts)


def render_detection_task() -> None:
    st.title("Object Detection")
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        st.subheader("Original Image")
        uploaded_name, image = render_uploader("detect_image")
        try:
            result = run_detection_button(uploaded_name, image, "detect")
            if result:
                result["candidate_detections"] = result.get("detections", [])
                result["candidate_nms_iou"] = YOLO_CANDIDATE_NMS_IOU
                st.session_state.detect_result_candidates = result
        except Exception as exc:
            st.error(str(exc))

    with right_col:
        st.subheader("Detection Result")
        detect_yolo_conf = render_conf_slider("detect_conf_threshold")
        result = filter_detection_result(
            st.session_state.get("detect_result_candidates"),
            detect_yolo_conf,
            DEFAULT_YOLO_IOU,
            image.size if image is not None else None,
        )
        st.session_state.detect_result = result
        st.session_state.detect_result_conf = detect_yolo_conf
        st.session_state.detect_result_iou = DEFAULT_YOLO_IOU
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
            placeholder="Example: lithium ion power bank, portable lighter, stainless kitchen knife\nMultiple item names can be separated by commas or new lines.",
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
            render_count_lines(get_classification_counts(result))
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

render_combined_task()
