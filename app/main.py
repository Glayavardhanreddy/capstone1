"""AutoOptML - FastAPI Application Entrypoint.

The Conditional Search-Space Optimiser for Algorithm and Architecture Selection.
Featuring User Authentication, SQLite Database Persistence, and Experiment History.
"""
import io
import os
import json
import uuid
from typing import Dict, Any, Optional, List
import pandas as pd
from pydantic import BaseModel
from sqlalchemy.orm import Session

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.database import engine, Base, get_db
from app.models import User, Experiment
from app.auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, get_optional_current_user, init_default_demo_user
)
from app.analyzer import profile_dataset
from app.conditional_optimizer import optimize_search_space
from app.trainer import train_and_evaluate_models
from app.sample_datasets import get_available_datasets, load_sample_dataset
from app.model_storage import get_model_file_path, generate_python_inference_code

# Initialize SQLite database schema
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AutoOptML - Conditional Search-Space Optimiser",
    description="Automated algorithm selection, architecture optimization, authentication, and persistent database storage.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store for active dataset dataframes
SESSION_DATASETS: Dict[str, pd.DataFrame] = {}
SESSION_PROFILES: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
def on_startup():
    """Seeds the default demo account on server start."""
    from app.database import SessionLocal
    db = SessionLocal()
    try:
        init_default_demo_user(db)
    finally:
        db.close()


# ----------------------------------------------------
# Request & Response Schemas
# ----------------------------------------------------
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    login_identifier: str  # Can be username or email
    password: str


class SampleLoadRequest(BaseModel):
    dataset_id: str


class ReprofileRequest(BaseModel):
    session_id: str
    target_col: str
    override_task: Optional[str] = None


class OptimizeRequest(BaseModel):
    session_id: str
    meta_features: Dict[str, Any]


class TrainRequest(BaseModel):
    session_id: str
    dataset_name: Optional[str] = "Uploaded Dataset"
    target_col: str
    task_type: str
    selected_algorithms: List[Dict[str, Any]]
    primary_metric: str
    cv_folds: int = 5
    effective_features: Optional[List[str]] = None
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None


# ----------------------------------------------------
# Authentication Endpoints
# ----------------------------------------------------
@app.post("/api/auth/register")
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    """Registers a new user and issues a session token."""
    username_clean = req.username.strip()
    email_clean = req.email.strip().lower()

    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    if db.query(User).filter(User.username == username_clean).first():
        raise HTTPException(status_code=400, detail="Username is already taken.")

    if db.query(User).filter(User.email == email_clean).first():
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    new_user = User(
        username=username_clean,
        email=email_clean,
        hashed_password=hash_password(req.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(new_user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "created_at": new_user.created_at.isoformat()
        }
    }


@app.post("/api/auth/login")
def login_user(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticates a user by username/email and password."""
    ident = req.login_identifier.strip().lower()
    user = db.query(User).filter(
        (User.email == ident) | (User.username.ilike(req.login_identifier.strip()))
    ).first()

    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat()
        }
    }


@app.post("/api/auth/demo-login")
def demo_login(db: Session = Depends(get_db)):
    """1-click instant login as pre-seeded demo user."""
    user = init_default_demo_user(db)
    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_demo": True
        }
    }


@app.get("/api/auth/me")
def get_user_profile(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Returns profile and experiment count of current user."""
    exp_count = db.query(Experiment).filter(Experiment.user_id == user.id).count()
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at.isoformat(),
        "experiments_count": exp_count
    }


# ----------------------------------------------------
# Experiment History Endpoints
# ----------------------------------------------------
@app.get("/api/experiments")
def list_experiments(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Lists all historical experiments for the authenticated user."""
    experiments = (
        db.query(Experiment)
        .filter(Experiment.user_id == user.id)
        .order_by(Experiment.created_at.desc())
        .all()
    )

    records = []
    for exp in experiments:
        records.append({
            "id": exp.id,
            "title": exp.title,
            "dataset_name": exp.dataset_name,
            "task_type": exp.task_type,
            "target_column": exp.target_column,
            "n_samples": exp.n_samples,
            "n_features": exp.n_features,
            "primary_metric": exp.primary_metric,
            "best_model_name": exp.best_model_name,
            "best_config_name": exp.best_config_name,
            "best_score": exp.best_score,
            "created_at": exp.created_at.strftime("%Y-%m-%d %H:%M")
        })
    return {"experiments": records}


@app.get("/api/experiments/{experiment_id}")
def get_experiment_details(
    experiment_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves full experiment details to restore historical leaderboard and charts."""
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == user.id
    ).first()

    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")

        diag_raw = json.loads(exp.diagnostics_json) if exp.diagnostics_json else {}
        rec_report = diag_raw.pop("recommendation_report", None)
        return {
            "id": exp.id,
            "title": exp.title,
            "dataset_name": exp.dataset_name,
            "task_type": exp.task_type,
            "target_column": exp.target_column,
            "n_samples": exp.n_samples,
            "n_features": exp.n_features,
            "primary_metric": exp.primary_metric,
            "best_model_name": exp.best_model_name,
            "best_config_name": exp.best_config_name,
            "best_score": exp.best_score,
            "leaderboard": json.loads(exp.leaderboard_json),
            "meta_features": json.loads(exp.meta_features_json),
            "diagnostics": diag_raw,
            "recommendation_report": rec_report,
            "created_at": exp.created_at.strftime("%Y-%m-%d %H:%M")
        }


@app.delete("/api/experiments/{experiment_id}")
def delete_experiment(
    experiment_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deletes an experiment record from the database."""
    exp = db.query(Experiment).filter(
        Experiment.id == experiment_id,
        Experiment.user_id == user.id
    ).first()

    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    db.delete(exp)
    db.commit()
    return {"message": "Experiment deleted successfully."}


# ----------------------------------------------------
# Dataset & Core AutoML Endpoints
# ----------------------------------------------------
@app.get("/api/sample-datasets")
async def list_sample_datasets():
    """Returns available benchmark sample datasets."""
    return {"datasets": get_available_datasets()}


@app.post("/api/load-sample")
async def load_sample(req: SampleLoadRequest):
    """Loads a built-in benchmark dataset and returns its profile."""
    try:
        df, default_target = load_sample_dataset(req.dataset_id)
        session_id = str(uuid.uuid4())[:8]
        profile = profile_dataset(df, target_col=default_target)
        
        SESSION_DATASETS[session_id] = df
        SESSION_PROFILES[session_id] = profile

        return {
            "session_id": session_id,
            "dataset_name": req.dataset_id,
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a CSV or Excel dataset, validates it, and generates its diagnostic profile."""
    filename = file.filename.lower()
    content = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx file.")

        if df.empty or len(df.columns) < 2:
            raise HTTPException(status_code=400, detail="Dataset must have at least 2 columns and 1 row.")

        session_id = str(uuid.uuid4())[:8]
        profile = profile_dataset(df)

        SESSION_DATASETS[session_id] = df
        SESSION_PROFILES[session_id] = profile

        return {
            "session_id": session_id,
            "dataset_name": file.filename,
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {str(e)}")


@app.post("/api/reprofile")
async def reprofile(req: ReprofileRequest):
    """Re-analyzes the dataset with a user-selected target column or task override."""
    df = SESSION_DATASETS.get(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset session expired or not found. Please upload again.")

    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' not found in dataset.")

    profile = profile_dataset(df, target_col=req.target_col)
    
    if req.override_task and req.override_task in ["binary_classification", "multiclass_classification", "regression"]:
        profile["summary"]["inferred_task"] = req.override_task
        profile["meta_features"]["task_type"] = req.override_task
        profile["target_analysis"]["inferred_task"] = req.override_task

    SESSION_PROFILES[req.session_id] = profile
    return {"session_id": req.session_id, "profile": profile}


@app.post("/api/optimize-search-space")
async def run_conditional_optimization(req: OptimizeRequest):
    """Executes the Conditional Search-Space Optimiser on meta-features."""
    try:
        result = optimize_search_space(req.meta_features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


@app.post("/api/train-and-evaluate")
async def train_models(
    req: TrainRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """Trains and benchmarks candidate configurations, returning the leaderboard and saving experiment to DB."""
    df = SESSION_DATASETS.get(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset session not found. Please reload or re-upload dataset.")

    try:
        benchmark_results = train_and_evaluate_models(
            df=df,
            target_col=req.target_col,
            task_type=req.task_type,
            selected_algorithms=req.selected_algorithms,
            primary_metric=req.primary_metric,
            cv_folds=req.cv_folds,
            effective_features=req.effective_features,
            numeric_cols=req.numeric_cols,
            categorical_cols=req.categorical_cols
        )

        # Automatically persist experiment in SQLite database if user is authenticated or demo user
        best = benchmark_results["best_model"]
        exp_id = best.get("model_id") or str(uuid.uuid4())[:8]

        # Use logged in user, or fallback to default demo user if unauthenticated
        owner_id = current_user.id if current_user else None
        if not owner_id:
            demo_user = init_default_demo_user(db)
            owner_id = demo_user.id

        diag_bundle = benchmark_results.get("diagnostics", {}).copy()
        if "recommendation_report" in benchmark_results:
            diag_bundle["recommendation_report"] = benchmark_results["recommendation_report"]

        experiment_record = Experiment(
            id=exp_id,
            user_id=owner_id,
            title=f"{req.dataset_name} ({req.task_type.replace('_', ' ').title()})",
            dataset_name=req.dataset_name or "Custom Dataset",
            task_type=req.task_type,
            target_column=req.target_col,
            n_samples=len(df),
            n_features=len(req.effective_features or []),
            primary_metric=req.primary_metric,
            best_model_name=best["algorithm_name"],
            best_config_name=best["config_name"],
            best_score=float(best["score"]),
            leaderboard_json=json.dumps(benchmark_results["leaderboard"]),
            meta_features_json=json.dumps({
                "n_samples": len(df),
                "n_features": len(req.effective_features or []),
                "task_type": req.task_type,
                "target_col": req.target_col
            }),
            diagnostics_json=json.dumps(diag_bundle),
            model_file_path=get_model_file_path(exp_id)
        )
        db.add(experiment_record)
        db.commit()

        benchmark_results["experiment_saved"] = True
        benchmark_results["experiment_id"] = exp_id

        return benchmark_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training pipeline error: {str(e)}")


@app.get("/api/download-model/{model_id}")
async def download_model(model_id: str):
    """Downloads the serialized model bundle (.joblib)."""
    file_path = get_model_file_path(model_id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found or expired.")

    return FileResponse(
        path=file_path,
        filename=f"autooptml_model_{model_id}.joblib",
        media_type="application/octet-stream"
    )


@app.get("/api/export-code/{model_id}", response_class=PlainTextResponse)
async def export_code(model_id: str):
    """Returns standalone Python inference code for the model."""
    code = generate_python_inference_code(model_id)
    return code


# Static files mount
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serves the main application SPA."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>AutoOptML</h1><p>UI loading...</p>")
