#!/usr/bin/env bash
set -u

PROJECT="/data/2_data_server/cv-07/dice/the Korea Customs Service/project"
CUSTOMS_WEB_PY="/data/2_data_server/cv-07/anaconda3/envs/customs_web/bin/python"
YOLO_PY="/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python"

cd "$PROJECT"

pgrep -af "$PROJECT/model/object_detection/yolo_worker.py" >/dev/null || \
  nohup "$YOLO_PY" "$PROJECT/model/object_detection/yolo_worker.py" > "$PROJECT/web/yolo_worker.log" 2>&1 &

pgrep -af "$PROJECT/model/text_classification/qwen_worker.py" >/dev/null || \
  nohup "$CUSTOMS_WEB_PY" "$PROJECT/model/text_classification/qwen_worker.py" > "$PROJECT/web/qwen_worker.log" 2>&1 &

pgrep -af "streamlit run $PROJECT/web/app.py" >/dev/null || \
  nohup "$CUSTOMS_WEB_PY" -m streamlit run "$PROJECT/web/app.py" \
    --server.address 0.0.0.0 \
    --server.port 8501 > "$PROJECT/web/streamlit.log" 2>&1 &

echo "Started. Run: bash $PROJECT/web/check.sh"
