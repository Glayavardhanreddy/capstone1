"""Test FastAPI HTTP endpoints using starlette TestClient."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_endpoints():
    print("Testing GET /")
    res = client.get("/")
    assert res.status_code == 200
    assert "AutoOptML" in res.text
    print("  GET / OK")

    print("Testing GET /api/sample-datasets")
    res = client.get("/api/sample-datasets")
    assert res.status_code == 200
    datasets = res.json()["datasets"]
    assert len(datasets) >= 5
    print(f"  GET /api/sample-datasets OK ({len(datasets)} datasets found)")

    print("Testing POST /api/load-sample")
    res = client.post("/api/load-sample", json={"dataset_id": "iris"})
    assert res.status_code == 200
    data = res.json()
    session_id = data["session_id"]
    profile = data["profile"]
    assert profile["summary"]["inferred_task"] == "multiclass_classification"
    print(f"  POST /api/load-sample OK (session={session_id})")

    print("Testing POST /api/optimize-search-space")
    res = client.post("/api/optimize-search-space", json={
        "session_id": session_id,
        "meta_features": profile["meta_features"]
    })
    assert res.status_code == 200
    opt_data = res.json()
    assert len(opt_data["selected_algorithms"]) > 0
    print(f"  POST /api/optimize-search-space OK (selected {len(opt_data['selected_algorithms'])} algorithms)")

    print("Testing POST /api/train-and-evaluate")
    res = client.post("/api/train-and-evaluate", json={
        "session_id": session_id,
        "target_col": profile["summary"]["target_column"],
        "task_type": profile["summary"]["inferred_task"],
        "selected_algorithms": opt_data["selected_algorithms"][:2],
        "primary_metric": "accuracy",
        "cv_folds": 3,
        "effective_features": profile["meta_features"]["effective_features"],
        "numeric_cols": profile["meta_features"]["numeric_cols"],
        "categorical_cols": profile["meta_features"]["categorical_cols"]
    })
    assert res.status_code == 200
    train_data = res.json()
    best_model_id = train_data["best_model"]["model_id"]
    print(f"  POST /api/train-and-evaluate OK (best model={train_data['best_model']['algorithm_name']}, id={best_model_id})")

    print(f"Testing GET /api/export-code/{best_model_id}")
    res = client.get(f"/api/export-code/{best_model_id}")
    assert res.status_code == 200
    assert "AutoOptML Standalone Inference Script" in res.text
    print("  GET /api/export-code OK")

    print(f"Testing GET /api/download-model/{best_model_id}")
    res = client.get(f"/api/download-model/{best_model_id}")
    assert res.status_code == 200
    assert len(res.content) > 100
    print("  GET /api/download-model OK")

    print("\n>>> ALL API ENDPOINTS PASSED VERIFICATION! <<<")

if __name__ == "__main__":
    test_endpoints()
