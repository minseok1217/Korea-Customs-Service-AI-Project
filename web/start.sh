#!/usr/bin/env bash
set -u

PROJECT="/data/2_data_server/cv-07/dice/the Korea Customs Service/project"
CUSTOMS_WEB_PY="/data/2_data_server/cv-07/anaconda3/envs/customs_web/bin/python"
YOLO_PY="/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python"
LLAMA_MODEL_DIR="${LLAMA_MODEL_DIR:-$PROJECT/model/text_classification/llama4_scout_17b_16e_instruct}"

cd "$PROJECT"

pgrep -af "$PROJECT/model/object_detection/yolo_worker.py" >/dev/null || \
  nohup "$YOLO_PY" "$PROJECT/model/object_detection/yolo_worker.py" > "$PROJECT/web/yolo_worker.log" 2>&1 &

pgrep -af "$PROJECT/model/text_classification/qwen_worker.py" >/dev/null || \
  nohup "$CUSTOMS_WEB_PY" "$PROJECT/model/text_classification/qwen_worker.py" > "$PROJECT/web/qwen_worker.log" 2>&1 &

if [ -f "$LLAMA_MODEL_DIR/config.json" ]; then
  pgrep -af "$PROJECT/model/text_classification/llama_worker.py" >/dev/null || \
    LLAMA_MODEL_DIR="$LLAMA_MODEL_DIR" nohup "$CUSTOMS_WEB_PY" "$PROJECT/model/text_classification/llama_worker.py" > "$PROJECT/web/llama_worker.log" 2>&1 &
else
  echo "Llama model not found at $LLAMA_MODEL_DIR; skipping llama_worker."
fi

pgrep -af "streamlit run $PROJECT/web/app.py" >/dev/null || \
  nohup "$CUSTOMS_WEB_PY" -m streamlit run "$PROJECT/web/app.py" \
    --server.address 0.0.0.0 \
    --server.port 8501 > "$PROJECT/web/streamlit.log" 2>&1 &

echo "Started. Run: bash $PROJECT/web/check.sh"
