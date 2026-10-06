"""Model storage, serialization, and code generation engine for AutoOptML."""
import os
import uuid
from typing import Dict, Any, Optional
import joblib

# In-memory session store for trained models
MODEL_STORE: Dict[str, Dict[str, Any]] = {}
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)


def save_trained_pipeline(
    pipeline: Any,
    model_name: str,
    target_col: str,
    feature_names: list[str],
    task_type: str,
    metrics: dict,
    label_encoder: Optional[Any] = None
) -> str:
    """Saves a fitted pipeline and metadata, returning a unique model ID."""
    model_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(EXPORT_DIR, f"model_{model_id}.joblib")
    
    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "target_col": target_col,
        "feature_names": feature_names,
        "task_type": task_type,
        "metrics": metrics,
        "label_encoder": label_encoder
    }
    joblib.dump(bundle, file_path)
    
    MODEL_STORE[model_id] = {
        "model_id": model_id,
        "file_path": file_path,
        "bundle": bundle
    }
    return model_id


def get_model_file_path(model_id: str) -> Optional[str]:
    """Retrieves file path for download."""
    if model_id in MODEL_STORE:
        return MODEL_STORE[model_id]["file_path"]
    
    # Check filesystem
    disk_path = os.path.join(EXPORT_DIR, f"model_{model_id}.joblib")
    if os.path.exists(disk_path):
        return disk_path
    return None


def generate_python_inference_code(model_id: str) -> str:
    """Generates a complete, ready-to-run Python script for inference with the exported model."""
    entry = MODEL_STORE.get(model_id)
    if not entry:
        return "# Model not found or session expired."
        
    bundle = entry["bundle"]
    model_name = bundle["model_name"]
    target_col = bundle["target_col"]
    feature_names = bundle["feature_names"]
    task_type = bundle["task_type"]
    
    features_repr = ",\n    ".join([f"'{f}'" for f in feature_names])
    sample_dict_items = ",\n        ".join([f"'{f}': [None]" for f in feature_names[:5]])
    
    code = f'''"""
AutoOptML Standalone Inference Script
Model: {model_name}
Target: {target_col} ({task_type})
Generated automatically by Conditional Search-Space Optimiser
"""
import joblib
import pandas as pd
import numpy as np

# 1. Load the trained model bundle
BUNDLE_PATH = "model_{model_id}.joblib"
print(f"Loading trained model bundle from {{BUNDLE_PATH}}...")
bundle = joblib.load(BUNDLE_PATH)
pipeline = bundle["pipeline"]
label_encoder = bundle.get("label_encoder")

# Expected input features
EXPECTED_FEATURES = [
    {features_repr}
]

print(f"Model successfully loaded: {{bundle['model_name']}}")
print(f"Target column: {{bundle['target_col']}}")

# 2. Example inference with new data
# Replace this sample dictionary with your actual raw data:
sample_data = pd.DataFrame({{
    {sample_dict_items}
}})

# Ensure all expected columns are present
for col in EXPECTED_FEATURES:
    if col not in sample_data.columns:
        sample_data[col] = np.nan

# 3. Predict using the end-to-end preprocessing + model pipeline
print("\\nRunning inference...")
raw_predictions = pipeline.predict(sample_data[EXPECTED_FEATURES])

if label_encoder is not None:
    # Decode integer classes back to human labels
    final_predictions = label_encoder.inverse_transform(raw_predictions)
else:
    final_predictions = raw_predictions

print(f"Predictions: {{final_predictions}}")

# If probabilities are supported:
if hasattr(pipeline, "predict_proba"):
    try:
        probabilities = pipeline.predict_proba(sample_data[EXPECTED_FEATURES])
        print(f"Class Probabilities:\\n{{probabilities}}")
    except Exception:
        pass
'''
    return code
