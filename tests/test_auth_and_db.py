"""Test suite verifying User Authentication, SQLite Database persistence, and Experiment History."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal, Base, engine
from app.models import User, Experiment

client = TestClient(app)

def test_auth_and_database_persistence():
    print("\n--- 1. Testing Default Demo User Initialization ---")
    res = client.post("/api/auth/demo-login")
    assert res.status_code == 200
    demo_data = res.json()
    assert "access_token" in demo_data
    demo_token = demo_data["access_token"]
    demo_user = demo_data["user"]
    print(f"  Demo login successful: {demo_user['username']} ({demo_user['email']})")

    print("\n--- 2. Testing User Registration ---")
    reg_payload = {
        "username": "TestScientist99",
        "email": "scientist99@example.com",
        "password": "securepassword123"
    }
    # Clean up if already exists from prior run
    db = SessionLocal()
    existing = db.query(User).filter(User.email == reg_payload["email"]).first()
    if existing:
        db.delete(existing)
        db.commit()
    db.close()

    res = client.post("/api/auth/register", json=reg_payload)
    assert res.status_code == 200
    reg_data = res.json()
    user_token = reg_data["access_token"]
    user_info = reg_data["user"]
    assert user_info["username"] == "TestScientist99"
    print(f"  Registered new user: id={user_info['id']}, username={user_info['username']}")

    print("\n--- 3. Testing User Login ---")
    login_res = client.post("/api/auth/login", json={
        "login_identifier": "scientist99@example.com",
        "password": "securepassword123"
    })
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()
    print("  User login successful with verified password hash.")

    print("\n--- 4. Testing Authenticated Profile Route ---")
    me_res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {user_token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["username"] == "TestScientist99"
    print(f"  GET /api/auth/me OK (experiments={me_data['experiments_count']})")

    print("\n--- 5. Testing Automated Experiment Saving to SQLite Database ---")
    # Load iris dataset
    sample_res = client.post("/api/load-sample", json={"dataset_id": "iris"})
    assert sample_res.status_code == 200
    s_data = sample_res.json()
    session_id = s_data["session_id"]
    profile = s_data["profile"]

    # Run search-space optimization
    opt_res = client.post("/api/optimize-search-space", json={
        "session_id": session_id,
        "meta_features": profile["meta_features"]
    })
    assert opt_res.status_code == 200
    opt_data = opt_res.json()

    # Train models with user's auth token
    train_res = client.post("/api/train-and-evaluate", json={
        "session_id": session_id,
        "dataset_name": "Iris Benchmark Test",
        "target_col": profile["summary"]["target_column"],
        "task_type": profile["summary"]["inferred_task"],
        "selected_algorithms": opt_data["selected_algorithms"][:2],
        "primary_metric": "accuracy",
        "cv_folds": 3,
        "effective_features": profile["meta_features"]["effective_features"],
        "numeric_cols": profile["meta_features"]["numeric_cols"],
        "categorical_cols": profile["meta_features"]["categorical_cols"]
    }, headers={"Authorization": f"Bearer {user_token}"})
    if train_res.status_code != 200:
        print(f"Error {train_res.status_code}: {train_res.text}")
    assert train_res.status_code == 200
    train_data = train_res.json()
    assert train_data.get("experiment_saved") == True
    exp_id = train_data.get("experiment_id")
    print(f"  Training benchmark finished and saved to SQLite! Experiment ID: {exp_id}")

    print("\n--- 6. Testing Retrieving User Experiments from Database ---")
    exp_list_res = client.get("/api/experiments", headers={"Authorization": f"Bearer {user_token}"})
    assert exp_list_res.status_code == 200
    experiments = exp_list_res.json()["experiments"]
    assert len(experiments) >= 1
    matched = [e for e in experiments if e["id"] == exp_id]
    assert len(matched) == 1
    print(f"  Retrieved {len(experiments)} experiment(s) from SQLite. Matched title: '{matched[0]['title']}'")

    print(f"\n--- 7. Testing Restoring Full Experiment Details ---")
    det_res = client.get(f"/api/experiments/{exp_id}", headers={"Authorization": f"Bearer {user_token}"})
    assert det_res.status_code == 200
    det_data = det_res.json()
    assert len(det_data["leaderboard"]) > 0
    assert "diagnostics" in det_data
    print(f"  Restored complete leaderboard ({len(det_data['leaderboard'])} models) & diagnostics from SQLite.")

    print(f"\n--- 8. Testing Experiment Deletion ---")
    del_res = client.delete(f"/api/experiments/{exp_id}", headers={"Authorization": f"Bearer {user_token}"})
    assert del_res.status_code == 200
    print("  Deleted experiment record successfully.")

    print("\n>>> ALL AUTHENTICATION AND DATABASE TESTS PASSED! <<<")

if __name__ == "__main__":
    test_auth_and_database_persistence()
