# Intent Classification NLU — Transformers + MLflow + FastAPI

A production-ready Spanish NLU project for intent classification built on Hugging Face Transformers. It includes:
- Training with Optuna hyperparameter tuning
- Experiment tracking, model registry, and deployment with MLflow
- A FastAPI inference service with robust preprocessing
- Utilities for dataset validation, augmentation/expansion, and evaluation
- Optional Docker image for serving


## Table of Contents
- Overview
- Project Structure
- Setup and Installation
- Configuration (config.yaml)
- Data Format
- Quickstart: Common Commands
  - Expand dataset
  - Train (baseline and enhanced)
  - Run pipelines
  - Serve API (FastAPI)
  - Evaluate and error analysis
  - MLflow UI
  - Run tests
  - Docker build and run
- Troubleshooting
- License


## Overview
This repository provides an end-to-end workflow to train and serve a Spanish intent classifier. Models are fine-tuned from a transformer checkpoint, tracked in MLflow, and can be served as a REST API. The service performs enhanced text preprocessing (normalization, contraction expansion, typo handling) to improve robustness.


## Project Structure
- config.yaml — Central configuration (datasets, experiment name, HPO ranges, data expansion).
- src/
  - intent_classifier.py — Training wrapper for a Transformer classifier.
  - data_utils.py — Dataset loading with Pydantic schema validation.
  - text_preprocessing.py — Enhanced text normalization/typo correction and augmentation helpers.
  - dataset_expansion.py — Dataset expansion engine used by the CLI script.
- scripts/
  - train.py — Baseline HPO training with Optuna and MLflow registration.
  - enhanced_train.py — Enhanced training with expanded HPO space and options.
  - error_analysis.py — Validation error report against MLflow-logged model.
  - generate_expanded_dataset.py — CLI to expand the dataset using templates/heuristics.
- pipelines/
  - 1_prepare_data.py — Prepare and split dataset (artifacts in data/processed/).
  - 2_run_hp_tuning.py — Run Optuna study, log to MLflow, save best params.
  - 3_train_final_model.py — Train final model with best hyperparams and register in MLflow.
  - 4_evaluate_on_blind_set.py — Evaluate a registered model on a blind set.
- serve.py — FastAPI app. Loads model from MLflow Registry (stage alias) with local fallback.
- text_classifier_wrapper.py — MLflow PyFunc wrapper for tokenization + inference.
- enhanced_text_classifier_wrapper.py — Removed (use text_classifier_wrapper.py).
- models/ — Saved local model artifacts (e.g., models/best_transformer/ for fallback).
- mlruns/ — MLflow tracking artifacts (local DB in mlflow.db by default).
- results/ — Training outputs/checkpoints.
- tests/ — Pytest-based unit tests for core utilities.


## Setup and Installation
Requirements: Python 3.9+

1) Create and activate a virtual environment
- python -m venv .venv
- source .venv/bin/activate    # Windows: .venv\Scripts\activate

2) Install dependencies
- pip install -r requirements.txt

3) (Optional) Set MLflow environment variables
- export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
- export MLFLOW_REGISTRY_URI=sqlite:///mlflow.db
- export MLFLOW_MODEL_NAME=intent-classifier-svc
- export MLFLOW_MODEL_STAGE=Production

Windows (PowerShell) equivalents:
- setx MLFLOW_TRACKING_URI "sqlite:///mlflow.db"
- setx MLFLOW_REGISTRY_URI "sqlite:///mlflow.db"
- setx MLFLOW_MODEL_NAME "intent-classifier-svc"
- setx MLFLOW_MODEL_STAGE "Production"


## Configuration (config.yaml)
Key sections:
- base.project_name: MLflow experiment/registered model name.
- data.training_set: Path to training dataset JSON.
- data.blind_test_set: Path to blind test dataset JSON.
- data.test_split_ratio: Validation split ratio for training.
- transformer_params: HPO search space for epochs, lr, and batch size.
- enhanced_params: Extended search space and features for enhanced training.
- data_expansion: Parameters for dataset expansion CLI (input/output/report, etc.).

Edit config.yaml to fit your data locations and preferences.


## Data Format
Training and blind test sets are JSON arrays of objects with Spanish keys:
[
  {"texto": "Quiero agendar una cita", "intencion": "agendar_cita"},
  {"texto": "Necesito cambiar mi turno", "intencion": "reprogramar_cita"}
]

Validation occurs with Pydantic to ensure non-empty texto fields.


## Quickstart: Common Commands
All commands assume the current working directory is the project root.

1) Expand the dataset (CLI)
- python scripts/generate_expanded_dataset.py --config-path config.yaml --output-path data/training_data_expanded.json --log-level INFO
This will write the expanded dataset and an expansion report as configured in config.yaml (data_expansion section).

2) Train (baseline)
- python scripts/train.py
Performs Optuna tuning, logs runs to MLflow, retrains best model, and registers a PyFunc model under base.project_name.

3) Train (enhanced)
- python scripts/enhanced_train.py
Uses an expanded search space and optional features (e.g., augmentation).

4) Run pipelines (decomposed workflow)
Run steps individually:
- python pipelines/1_prepare_data.py
- python pipelines/2_run_hp_tuning.py
- python pipelines/3_train_final_model.py
- python pipelines/4_evaluate_on_blind_set.py
Most scripts accept --config-path; run with -h to see options.

5) Serve the model (FastAPI)
Start the API (loads MLflow model by stage alias, else falls back to models/best_transformer):
- uvicorn serve:app --host 0.0.0.0 --port 8000

Health check:
- curl http://localhost:8000/health

Predict:
- curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Quiero agendar una cita para mañana"}'

Environment variables to control serving:
- MLFLOW_TRACKING_URI, MLFLOW_REGISTRY_URI, MLFLOW_MODEL_NAME, MLFLOW_MODEL_STAGE

6) Evaluate and error analysis
- python models/evaluate.py --config-path config.yaml --model-path models/intent_model.joblib
- python scripts/error_analysis.py

7) MLflow UI
Launch the UI locally (sees mlflow.db by default if configured):
- mlflow ui --host 0.0.0.0 --port 5000
Open http://localhost:5000 to view experiments and registered models.

8) Run tests
- pip install -r requirements.txt  # ensure dev deps installed
- pytest -q


## Docker (optional)
Build the image:
- docker build -t intent-svc:latest .

Run the API container (exposes port 8000):
- docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db} \
  -e MLFLOW_REGISTRY_URI=${MLFLOW_REGISTRY_URI:-sqlite:///mlflow.db} \
  -e MLFLOW_MODEL_NAME=${MLFLOW_MODEL_NAME:-intent-classifier-svc} \
  -e MLFLOW_MODEL_STAGE=${MLFLOW_MODEL_STAGE:-Production} \
  intent-svc:latest

Then call the API as shown in the Predict example.


## Troubleshooting
- MLflow model not found: Ensure you have trained and registered a model (scripts/train.py). The server will fall back to models/best_transformer if present.
- CUDA/MPS warnings: Training forces TOKENIZERS_PARALLELISM=false and disables pin_memory; these warnings are safe.
- JSON dataset errors: Validate your JSON matches the Data Format section; run tests/test_data_utils.py locally with pytest.
- Port already in use: Change --port for uvicorn or stop the conflicting process.


## License
This project is provided as-is for educational and internal use. Adjust as needed for your environment.



## FAQ: What are files prefixed with enhanced_ used for?
These files provide an “enhanced” training and evaluation path that adds stronger preprocessing, optional data augmentation, and extra training options on top of the baseline pipeline. They are optional and coexist with the baseline so you can compare results.

- src/enhanced_intent_classifier.py
  - A drop-in alternative to src/intent_classifier.py with extras:
    - EnhancedTextPreprocessor integration (normalization, contraction expansion, typo correction)
    - Optional data augmentation via DataAugmentationEngine
    - Optional Focal Loss, weight decay, and warmup steps
    - Saves enhanced_metadata.json alongside label_mappings.json for richer model metadata
  - Used by the enhanced training scripts below.

- scripts/enhanced_train.py
  - HPO training using the enhanced model and the enhanced_params section of config.yaml.
  - Logs runs to MLflow and registers a model named "<project_name>-enhanced" (e.g., intent-classifier-svc-enhanced).
  - Outputs enhanced artifacts under models/best_enhanced_transformer.
  - Run: python scripts/enhanced_train.py

- scripts/simple_enhanced_train.py
  - A quick, fixed-parameter training path to try the enhanced features without running Optuna.
  - Registers a model named "<project_name>-enhanced-simple" (e.g., intent-classifier-svc-enhanced-simple).
  - Outputs artifacts under models/simple_enhanced_transformer.
  - Run: python scripts/simple_enhanced_train.py

- enhanced_text_classifier_wrapper.py [removed]
  - Removed to avoid duplication. Use text_classifier_wrapper.py instead.
  - The FastAPI server already applies EnhancedTextPreprocessor at request time, so no special wrapper is needed.
  - If you previously referenced EnhancedTextClassifierWrapper, import TextClassifierWrapper; behavior remains compatible.

- scripts/enhanced_error_analysis.py
  - Error analysis focused on the latest enhanced model; prints confusion patterns and shows preprocessing effects on difficult examples.
  - Run: python scripts/enhanced_error_analysis.py

Serving enhanced models
- The FastAPI server (serve.py) already applies EnhancedTextPreprocessor at request time, regardless of which wrapper was used during registration.
- To serve an enhanced model from the MLflow Registry, set the model name env var to the enhanced one before starting uvicorn, for example:
  - export MLFLOW_MODEL_NAME=intent-classifier-svc-enhanced
  - uvicorn serve:app --host 0.0.0.0 --port 8000
- Alternatively, you can serve the "enhanced-simple" model by setting MLFLOW_MODEL_NAME=intent-classifier-svc-enhanced-simple.

Note on naming
- The correct prefix is enhanced_. If you see references to "enchanced_", that is a misspelling — all relevant files in this repo use enhanced_.
