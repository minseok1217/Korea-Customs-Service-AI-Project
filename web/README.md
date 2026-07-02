# Korea Customs Service Streamlit Project

This directory contains a small Streamlit dashboard for category review and project notes.

## Run

```bash
cd "/data/2_data_server/cv-07/dice/the Korea Customs Service/project"
cd web
/data/2_data_server/cv-07/anaconda3/envs/customs_web/bin/python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

The Streamlit app runs in `customs_web`. YOLOv5 inference is called through the `yolov5` env from inside the app.

## Files

- `app.py`: Streamlit web page
- `requirements.txt`: Python packages needed by the app
- `.streamlit/config.toml`: Streamlit server/theme defaults
