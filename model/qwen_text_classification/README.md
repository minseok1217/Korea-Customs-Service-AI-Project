# Qwen Text Classification

Target path:

```bash
/data/2_data_server/cv-07/dice/the Korea Customs Service/project/model/qwen_text_classification
```

Python env requested by user:

```bash
/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python
```

Important: the current `yolov5` env uses a Python version too old for recent Qwen3/Qwen3.5 models requiring `transformers>=4.51` and Python >= 3.9. Do not upgrade Python inside `yolov5` unless you are willing to risk breaking YOLO training.

## Install Runtime Packages

For Qwen3/Qwen3.5:

```bash
PY=/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python
$PY -m pip install -U "transformers>=4.51.0" accelerate safetensors sentencepiece huggingface_hub datasets scikit-learn
```

If this fails due Python version, use a newer env such as a Qwen-specific env, or clone `yolov5` to a separate Python >=3.10 env.

## Download Model

Default uses `Qwen/Qwen3-4B`. If you have the exact Qwen3.5 4B HF id, replace `MODEL_ID`.

```bash
cd "/data/2_data_server/cv-07/dice/the Korea Customs Service/project/model/qwen_text_classification"
PY=/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python
MODEL_ID=Qwen/Qwen3-4B

$PY download_model.py \
  --model-id "$MODEL_ID" \
  --output-dir "/data/2_data_server/cv-07/dice/the Korea Customs Service/project/model/qwen3_4b"
```

## Single Text Classification

```bash
PY=/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python

$PY classify_text.py \
  --model "/data/2_data_server/cv-07/dice/the Korea Customs Service/project/model/qwen3_4b" \
  --labels "prohibited,non-prohibited,uncertain" \
  --text "The baggage scan contains a knife-shaped metallic object."
```

## CSV Classification

Input CSV must contain a `text` column by default.

```bash
PY=/data/2_data_server/cv-07/anaconda3/envs/yolov5/bin/python

$PY classify_text.py \
  --model "/data/2_data_server/cv-07/dice/the Korea Customs Service/project/model/qwen3_4b" \
  --labels "prohibited,non-prohibited,uncertain" \
  --input-csv input.csv \
  --text-column text \
  --output-jsonl outputs/predictions.jsonl
```
