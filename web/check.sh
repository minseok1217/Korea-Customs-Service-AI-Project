#!/usr/bin/env bash
PROJECT="/data/2_data_server/cv-07/dice/the Korea Customs Service/project"

echo "--- processes ---"
pgrep -af "$PROJECT/model/object_detection/yolo_worker.py" || echo "yolo_worker: not running"
pgrep -af "$PROJECT/model/text_classification/qwen_worker.py" || echo "qwen_worker: not running"
pgrep -af "streamlit run $PROJECT/web/app.py" || echo "streamlit: not running"

echo "--- health ---"
curl -fsS http://127.0.0.1:18081/health || true
echo
curl -fsS http://127.0.0.1:18082/health || true
echo
curl -I --max-time 5 http://127.0.0.1:8501 2>/dev/null | head -1 || true

echo "--- logs ---"
for f in "$PROJECT/web/yolo_worker.log" "$PROJECT/web/qwen_worker.log" "$PROJECT/web/streamlit.log"; do
  echo "### $f"
  tail -30 "$f" 2>/dev/null || true
done
