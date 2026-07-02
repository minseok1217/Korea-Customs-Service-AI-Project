# Customs Web App

This Streamlit app runs X-ray object detection using the existing YOLOv5 training/inference environment.

## Design

- Streamlit/web env: Python >= 3.10, contains `streamlit`, `pandas`, `pillow`.
- YOLO model env: existing `/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python`.
- The web app calls `yolo_infer_json.py` through subprocess, so the Streamlit env and YOLO env can be different.

This avoids modifying or breaking the YOLO training environment.

## Paths

```bash
APP_DIR="/data/2_data_server/cv-07/dice/the Korea Customs Service/project/model/customs_web_app"
YOLO_REPO="/data/2_data_server/cv-07/dice/the Korea Customs Service/model/yoloV5/1). AI 모델 소스코드/yolov5"
YOLO_PY="/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python"
```

## Create Web/Qwen Env

Recommended:

```bash
conda create -n customs_web python=3.10 -y
conda activate customs_web
python -m pip install -U pip
python -m pip install -r "$APP_DIR/requirements_web.txt"
```

For Qwen text classification in the same env:

```bash
python -m pip install -U \
  "transformers>=4.51.0" \
  accelerate \
  safetensors \
  sentencepiece \
  huggingface_hub \
  datasets \
  scikit-learn
```

Optional for LoRA fine-tuning:

```bash
python -m pip install -U peft trl bitsandbytes
```

## Run Streamlit

```bash
conda activate customs_web
cd "$APP_DIR"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open:

```text
http://SERVER_IP:8501
```

## Weight Path Examples

```bash
/data/2_data_server/cv-07/dice/the Korea Customs Service/model/yoloV5/1). AI 모델 소스코드/yolov5/runs/train/17_super_mapped_yolov5x6_e30/weights/best.pt
/data/2_data_server/cv-07/dice/the Korea Customs Service/model/yoloV5/1). AI 모델 소스코드/yolov5/runs/train/231_super_mapped_yolov5x6_e30/weights/best.pt
```

If training is still running, use `last.pt` only for quick checks. Use `best.pt` for final testing.
