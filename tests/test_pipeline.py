"""Unit test suite verifying all pipeline components of AutoOptML."""
import sys
import os
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.sample_datasets import load_sample_dataset, get_available_datasets
from app.analyzer import profile_dataset
from app.conditional_optimizer import optimize_search_space
from app.trainer import train_and_evaluate_models
from app.model_storage import generate_python_inference_code


def test_classification_flow():
    print("\n--- Testing Classification Flow (Breast Cancer) ---")
    df, target = load_sample_dataset("breast_cancer")
    assert isinstance(df, pd.DataFrame)
    print(f"Loaded dataset: {df.shape}, target={target}")

    # 1. Profile
    profile = profile_dataset(df, target_col=target)
    assert profile["summary"]["inferred_task"] == "binary_classification"
    assert profile["meta_features"]["n_features"] == 30
    print(f"Profiling successful: task={profile['summary']['inferred_task']}, features={profile['summary']['effective_features_count']}")

    # 2. Optimize Search Space
    opt_result = optimize_search_space(profile["meta_features"])
    assert len(opt_result["selected_algorithms"]) > 0
    print(f"Conditional Optimiser: Retained={opt_result['optimization_summary']['retained_count']}, Pruned={opt_result['optimization_summary']['pruned_count']}")
    for r in opt_result["triggered_rules"]:
        print(f"  Triggered Rule: [{r['code']}] {r['title']}")

    # 3. Train & Evaluate
    print("Training candidate models...")
    res = train_and_evaluate_models(
        df=df,
        target_col=target,
        task_type="binary_classification",
        selected_algorithms=opt_result["selected_algorithms"][:3], # test first 3 for speed
        primary_metric=opt_result["optimization_summary"]["recommended_primary_metric"],
        cv_folds=3,
        effective_features=profile["meta_features"]["effective_features"],
        numeric_cols=profile["meta_features"]["numeric_cols"],
        categorical_cols=profile["meta_features"]["categorical_cols"]
    )

    assert len(res["leaderboard"]) > 0
    best = res["best_model"]
    print(f"Winner Model: {best['algorithm_name']} ({best['config_name']}) - Score: {best['score']}")
    assert best["model_id"] is not None

    # 4. Code Generation
    code = generate_python_inference_code(best["model_id"])
    assert "joblib.load" in code
    print("Generated Inference Code verified.")


def test_regression_flow():
    print("\n--- Testing Regression Flow (Diabetes) ---")
    df, target = load_sample_dataset("diabetes")
    profile = profile_dataset(df, target_col=target)
    assert profile["summary"]["inferred_task"] == "regression"

    opt_result = optimize_search_space(profile["meta_features"])
    print(f"Conditional Optimiser Regression: Retained={opt_result['optimization_summary']['retained_count']}")

    res = train_and_evaluate_models(
        df=df,
        target_col=target,
        task_type="regression",
        selected_algorithms=opt_result["selected_algorithms"][:3],
        primary_metric="rmse",
        cv_folds=3,
        effective_features=profile["meta_features"]["effective_features"],
        numeric_cols=profile["meta_features"]["numeric_cols"],
        categorical_cols=profile["meta_features"]["categorical_cols"]
    )
    assert len(res["leaderboard"]) > 0
    print(f"Regression Best Model: {res['best_model']['algorithm_name']} - RMSE: {res['best_model']['score']}")


if __name__ == "__main__":
    test_classification_flow()
    test_regression_flow()
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
