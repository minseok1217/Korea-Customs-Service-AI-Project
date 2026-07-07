#!/usr/bin/env bash
PROJECT="/data/2_data_server/cv-07/dice/the Korea Customs Service/project"

pkill -f "$PROJECT/model/object_detection/yolo_worker.py" || true
pkill -f "$PROJECT/model/text_classification/qwen_worker.py" || true
pkill -f "$PROJECT/model/text_classification/llama_worker.py" || true
pkill -f "streamlit run $PROJECT/web/app.py" || true

echo "Stopped project workers and Streamlit."
