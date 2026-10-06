"""Dataset Profiler & Meta-Feature Extraction Engine for AutoOptML."""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def infer_column_type(series: pd.Series, col_name: str, total_rows: int) -> str:
    """Infers the semantic type of a column."""
    name_lower = col_name.lower()
    
    # Check for ID-like columns
    if any(id_keyword in name_lower for id_keyword in ["id", "uuid", "guid", "index"]) and series.nunique() >= total_rows * 0.9:
        return "id"
    
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Check boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
        
    # Check numeric
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.nunique()
        # If very few integer values, it could be treated as discrete/categorical
        if unique_vals <= 5 and pd.api.types.is_integer_dtype(series):
            return "categorical_discrete"
        return "numeric"
        
    # Check strings/objects
    unique_vals = series.nunique()
    if unique_vals <= 50 or (unique_vals / max(total_rows, 1)) < 0.2:
        return "categorical"
    elif unique_vals >= total_rows * 0.95 and total_rows > 30:
        return "id"
    else:
        return "text_high_cardinality"


def infer_task_type(target_series: pd.Series) -> str:
    """Infers whether the machine learning problem is classification or regression."""
    target_clean = target_series.dropna()
    unique_count = target_clean.nunique()
    total_count = len(target_clean)

    if total_count == 0:
        return "binary_classification"

    if pd.api.types.is_bool_dtype(target_clean):
        return "binary_classification"

    if pd.api.types.is_object_dtype(target_clean) or pd.api.types.is_string_dtype(target_clean) or not pd.api.types.is_numeric_dtype(target_clean):
        if unique_count == 2:
            return "binary_classification"
        else:
            return "multiclass_classification"

    if pd.api.types.is_numeric_dtype(target_clean):
        # If float or high unique values, it's regression
        if pd.api.types.is_float_dtype(target_clean):
            # Check if floats are actually integers (e.g. 0.0, 1.0)
            if (target_clean == target_clean.round()).all() and unique_count <= 10:
                return "binary_classification" if unique_count == 2 else "multiclass_classification"
            return "regression"
        else:
            # Integer type
            if unique_count <= 2:
                return "binary_classification"
            elif unique_count <= 10 or (unique_count / total_count < 0.05 and unique_count <= 20):
                return "multiclass_classification"
            else:
                return "regression"

    return "regression"


def profile_dataset(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes dataset properties, extracts meta-features, and generates
    diagnostic health metrics and warnings for the Conditional Search-Space Optimiser.
    """
    total_rows, total_cols = df.shape
    
    # Auto-detect target column if not provided
    if not target_col or target_col not in df.columns:
        # Search for obvious target column candidates
        common_targets = ["target", "label", "class", "churn", "churned", "price", "outcome", "y", "status"]
        matched_target = None
        for col in df.columns:
            if col.lower() in common_targets:
                matched_target = col
                break
        target_col = matched_target if matched_target else df.columns[-1]

    # Column typing
    column_profiles = []
    numeric_cols = []
    categorical_cols = []
    id_cols = []
    text_cols = []
    datetime_cols = []
    
    total_missing_cells = int(df.isna().sum().sum())
    total_cells = total_rows * total_cols
    missing_ratio_overall = float(total_missing_cells / max(total_cells, 1))
    
    missing_per_column = {}
    
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        pct_missing = round(float((n_missing / max(total_rows, 1)) * 100), 2)
        missing_per_column[col] = {"count": n_missing, "percentage": pct_missing}
        
        inferred_type = infer_column_type(series, col, total_rows)
        
        # Categorize
        if col != target_col:
            if inferred_type in ["numeric"]:
                numeric_cols.append(col)
            elif inferred_type in ["categorical", "categorical_discrete", "boolean"]:
                categorical_cols.append(col)
            elif inferred_type == "id":
                id_cols.append(col)
            elif inferred_type == "datetime":
                datetime_cols.append(col)
            else:
                text_cols.append(col)

        sample_vals = series.dropna().unique()[:4].tolist()
        # Convert non-serializable objects to string
        sample_vals = [str(v) if not isinstance(v, (int, float, bool, str)) else v for v in sample_vals]

        column_profiles.append({
            "name": col,
            "inferred_type": inferred_type,
            "data_type": str(series.dtype),
            "is_target": (col == target_col),
            "missing_count": n_missing,
            "missing_percentage": pct_missing,
            "unique_count": int(series.nunique()),
            "sample_values": sample_vals
        })

    # Target variable analysis
    target_series = df[target_col]
    task_type = infer_task_type(target_series)
    
    target_clean = target_series.dropna()
    target_meta: Dict[str, Any] = {
        "column_name": target_col,
        "inferred_task": task_type,
        "unique_count": int(target_clean.nunique()),
        "missing_count": int(target_series.isna().sum())
    }
    
    is_imbalanced = False
    imbalance_ratio = 1.0
    
    if task_type in ["binary_classification", "multiclass_classification"]:
        val_counts = target_clean.value_counts()
        dist = {}
        for k, v in val_counts.items():
            dist[str(k)] = {
                "count": int(v),
                "percentage": round(float((v / len(target_clean)) * 100), 2)
            }
        target_meta["distribution"] = dist
        
        if len(val_counts) >= 2:
            majority_count = float(val_counts.iloc[0])
            minority_count = float(val_counts.iloc[-1])
            imbalance_ratio = round(majority_count / max(minority_count, 1), 2)
            is_imbalanced = (imbalance_ratio >= 2.5)
            target_meta["imbalance_ratio"] = imbalance_ratio
            target_meta["is_imbalanced"] = is_imbalanced
            target_meta["majority_class"] = str(val_counts.index[0])
            target_meta["minority_class"] = str(val_counts.index[-1])
    else:
        # Regression target stats
        target_meta["distribution"] = None
        try:
            target_numeric = pd.to_numeric(target_clean)
            target_meta["mean"] = round(float(target_numeric.mean()), 3) if len(target_numeric) > 0 else 0
            target_meta["std"] = round(float(target_numeric.std()), 3) if len(target_numeric) > 0 else 0
            target_meta["median"] = round(float(target_numeric.median()), 3) if len(target_numeric) > 0 else 0
            target_meta["min"] = round(float(target_numeric.min()), 3) if len(target_numeric) > 0 else 0
            target_meta["max"] = round(float(target_numeric.max()), 3) if len(target_numeric) > 0 else 0
            skewness = float(target_numeric.skew()) if len(target_numeric) > 2 else 0.0
            target_meta["skewness"] = round(skewness, 3)
            target_meta["is_skewed"] = abs(skewness) > 1.0
        except Exception:
            target_meta["mean"] = 0
            target_meta["std"] = 0
            target_meta["median"] = 0
            target_meta["min"] = 0
            target_meta["max"] = 0
            target_meta["skewness"] = 0.0
            target_meta["is_skewed"] = False

    # Effective features count (excluding target and ID columns)
    effective_features = [c for c in df.columns if c != target_col and c not in id_cols]
    n_features = len(effective_features)
    ratio_n_p = round(float(total_rows / max(n_features, 1)), 2)

    # Automated Diagnosis & Warnings
    warnings = []
    if id_cols:
        warnings.append({
            "type": "info",
            "title": "Identifier Columns Detected",
            "message": f"Columns {id_cols} have high cardinality/ID patterns and will be excluded from training to prevent data leakage."
        })
    if total_missing_cells > 0:
        warnings.append({
            "type": "warning",
            "title": "Missing Values Present",
            "message": f"{total_missing_cells} missing values ({round(missing_ratio_overall*100, 2)}% of cells) detected. AutoOptML will dynamically inject robust median/most-frequent imputation."
        })
    if is_imbalanced:
        warnings.append({
            "type": "warning",
            "title": "Class Imbalance Alert",
            "message": f"Class distribution ratio is {imbalance_ratio}:1. Search-Space Optimiser will condition models with balanced weighting and prioritize Macro F1 / ROC-AUC scoring."
        })
    if total_rows < 300:
        warnings.append({
            "type": "caution",
            "title": "Small Dataset Size",
            "message": f"Sample size (N={total_rows}) is low. Deep neural architectures and high-variance ensembles will be conditioned with tighter regularization to prevent overfitting."
        })
    elif total_rows > 10000:
        warnings.append({
            "type": "info",
            "title": "Large Dataset Scale",
            "message": f"Sample size (N={total_rows}) is large. Slow O(N^2) algorithms (such as standard RBF SVM) will be pruned in favor of Histogram-based Boosted Trees and SGD models."
        })
    if n_features > 50 or n_features > total_rows:
        warnings.append({
            "type": "warning",
            "title": "High Dimensionality Detected",
            "message": f"Feature space (P={n_features}) is wide relative to sample size. L1/L2 Regularization and tree feature subsampling will be enforced."
        })

    # Extract meta-features bundle for the conditional optimizer
    meta_features = {
        "n_samples": total_rows,
        "n_features": n_features,
        "ratio_samples_to_features": ratio_n_p,
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "n_id_cols": len(id_cols),
        "n_text_cols": len(text_cols),
        "missing_ratio": missing_ratio_overall,
        "has_missing_values": (total_missing_cells > 0),
        "is_high_dimensional": (n_features > 40 or n_features > total_rows),
        "is_small_dataset": (total_rows < 400),
        "is_large_dataset": (total_rows > 10000),
        "is_imbalanced": is_imbalanced,
        "imbalance_ratio": imbalance_ratio,
        "task_type": task_type,
        "target_col": target_col,
        "effective_features": effective_features,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "id_cols": id_cols
    }

    # Generate preview (first 10 rows sanitized)
    preview_df = df.head(10).copy()
    # Fill NaN with None for JSON compliance
    preview_records = preview_df.replace({np.nan: None}).to_dict(orient="records")

    return {
        "summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "effective_features_count": n_features,
            "total_missing_cells": total_missing_cells,
            "missing_percentage": round(missing_ratio_overall * 100, 2),
            "ratio_samples_to_features": ratio_n_p,
            "target_column": target_col,
            "inferred_task": task_type
        },
        "target_analysis": target_meta,
        "column_profiles": column_profiles,
        "meta_features": meta_features,
        "warnings": warnings,
        "preview_data": {
            "columns": list(df.columns),
            "rows": preview_records
        }
    }
