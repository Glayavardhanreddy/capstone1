"""Conditional Search-Space Optimiser for Algorithm & Architecture Selection.

Implements rule-based and meta-feature-conditioned pruning and hyperparameter
architecture configuration to reduce unnecessary model training, minimize manual trial-and-error,
and select the best tailored configurations for any tabular dataset.
"""
from typing import Dict, Any, List


def optimize_search_space(meta_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes dataset meta-features and produces:
    1. Selected algorithms conditioned on data properties.
    2. Pruned algorithms with explicit scientific rationale.
    3. Conditionally generated hyperparameter configurations / search grid.
    4. Diagnostic rules triggered during search-space optimization.
    5. Recommended evaluation metrics and CV strategy.
    """
    n_samples = meta_features.get("n_samples", 100)
    n_features = meta_features.get("n_features", 10)
    ratio_np = meta_features.get("ratio_samples_to_features", 10.0)
    n_categorical = meta_features.get("n_categorical", 0)
    has_missing = meta_features.get("has_missing_values", False)
    is_imbalanced = meta_features.get("is_imbalanced", False)
    imbalance_ratio = meta_features.get("imbalance_ratio", 1.0)
    task_type = meta_features.get("task_type", "binary_classification")
    is_regression = (task_type == "regression")
    
    selected_algorithms = []
    pruned_algorithms = []
    triggered_rules = []
    
    # Base algorithm registry
    if not is_regression:
        candidate_pool = [
            {"id": "logistic_regression", "name": "Regularized Logistic Regression", "family": "Linear Model"},
            {"id": "random_forest", "name": "Random Forest Classifier", "family": "Tree Ensemble (Bagging)"},
            {"id": "hist_gradient_boosting", "name": "Hist-Gradient Boosting Classifier", "family": "Tree Ensemble (Boosting)"},
            {"id": "extra_trees", "name": "Extra Trees Classifier", "family": "Extremely Randomized Trees"},
            {"id": "svm", "name": "Support Vector Machine (SVC)", "family": "Kernel Method"},
            {"id": "knn", "name": "K-Nearest Neighbors (KNN)", "family": "Instance-based / Non-parametric"},
            {"id": "decision_tree", "name": "Decision Tree Classifier", "family": "Interpretable Tree Baseline"},
            {"id": "mlp", "name": "Multi-Layer Perceptron (Neural Net)", "family": "Feed-Forward Neural Architecture"}
        ]
    else:
        candidate_pool = [
            {"id": "ridge_regression", "name": "Ridge / ElasticNet Regressor", "family": "Penalized Linear Model"},
            {"id": "random_forest_reg", "name": "Random Forest Regressor", "family": "Tree Ensemble (Bagging)"},
            {"id": "hist_gradient_boosting_reg", "name": "Hist-Gradient Boosting Regressor", "family": "Tree Ensemble (Boosting)"},
            {"id": "extra_trees_reg", "name": "Extra Trees Regressor", "family": "Extremely Randomized Trees"},
            {"id": "svr", "name": "Support Vector Regressor (SVR)", "family": "Kernel Method"},
            {"id": "knn_reg", "name": "K-Nearest Neighbors Regressor", "family": "Instance-based / Non-parametric"},
            {"id": "decision_tree_reg", "name": "Decision Tree Regressor", "family": "Interpretable Tree Baseline"},
            {"id": "mlp_reg", "name": "Multi-Layer Perceptron Regressor", "family": "Feed-Forward Neural Architecture"}
        ]

    # Evaluate Conditions & Apply Pruning Rules

    # Condition 1: Scale Pruning (Sample size N)
    prune_svm = False
    prune_knn = False
    if n_samples > 8000:
        prune_svm = True
        pruned_algorithms.append({
            "id": "svr" if is_regression else "svm",
            "name": "Support Vector Machine / Regressor",
            "reason": f"Sample size (N={n_samples}) causes quadratic/cubic computational complexity O(N^2) to O(N^3). Training and kernel matrix inversion would be bottlenecked.",
            "rule": "SCALE_PRUNE_QUADRATIC_KERNELS"
        })
        prune_knn = True
        pruned_algorithms.append({
            "id": "knn_reg" if is_regression else "knn",
            "name": "K-Nearest Neighbors",
            "reason": f"Sample size (N={n_samples}) causes slow O(N*P) inference lookup per sample without sufficient tree structure efficiency.",
            "rule": "SCALE_PRUNE_INSTANCE_BASED"
        })
        triggered_rules.append({
            "code": "SCALE_LARGE_N",
            "title": "Large Dataset Scale Filtering",
            "description": f"Detected N={n_samples} (>8,000). Pruned computationally prohibitive kernel and instance methods (SVM, KNN) to guarantee fast training and high throughput."
        })

    # Condition 2: Deep MLP Sample Starvation Pruning
    prune_mlp = False
    if n_samples < 250:
        prune_mlp = True
        pruned_algorithms.append({
            "id": "mlp_reg" if is_regression else "mlp",
            "name": "Multi-Layer Perceptron (Neural Net)",
            "reason": f"Sample size (N={n_samples}) is critically low for neural network optimization. Gradient descent on small tabular data is prone to overfitting and representation starvation.",
            "rule": "SAMPLE_STARVATION_PRUNE_NEURAL_NET"
        })
        triggered_rules.append({
            "code": "SAMPLE_STARVATION_MLP",
            "title": "Neural Architecture Pruned",
            "description": f"Sample size N={n_samples} is below statistical threshold for deep/feedforward networks. Pruned MLP to protect against high variance."
        })

    # Condition 3: High Dimensionality & Curse of Dimensionality
    if (n_features > 40 or ratio_np < 3.0) and not prune_knn:
        # Distance metrics collapse in high dimensions
        prune_knn = True
        pruned_algorithms.append({
            "id": "knn_reg" if is_regression else "knn",
            "name": "K-Nearest Neighbors",
            "reason": f"Feature count (P={n_features}) relative to N={n_samples} triggers the curse of dimensionality. Euclidean distance metrics lose discriminative variance in high dimensions.",
            "rule": "CURSE_OF_DIMENSIONALITY_PRUNE_KNN"
        })
        triggered_rules.append({
            "code": "HIGH_DIMENSIONALITY_KNN",
            "title": "Curse of Dimensionality Pruning",
            "description": f"Features P={n_features} relative to samples triggers distance metric degeneration. Pruned KNN in favor of tree feature-subsampling and penalized linear models."
        })

    # Condition 4: Categorical Dominance
    if n_categorical > 0 and (n_categorical / max(n_features, 1)) > 0.4:
        triggered_rules.append({
            "code": "CATEGORICAL_DOMINANCE",
            "title": "High Categorical Feature Ratio",
            "description": f"{round((n_categorical/n_features)*100)}% of features are categorical. Conditioned Tree Ensembles to prioritize split frequency and adjusted linear penalty parameters."
        })

    # Condition 5: Class Imbalance
    if is_imbalanced and not is_regression:
        triggered_rules.append({
            "code": "CLASS_IMBALANCE_COMPENSATION",
            "title": "Imbalanced Learning Conditioning",
            "description": f"Imbalance ratio {imbalance_ratio}:1 detected. Injected class_weight='balanced' across applicable models and conditioned metric optimization toward Macro-F1 / ROC-AUC."
        })

    # Condition 6: Small Sample Regularization
    if n_samples < 400:
        triggered_rules.append({
            "code": "SMALL_SAMPLE_REGULARIZATION",
            "title": "Small Sample Complexity Constraints",
            "description": f"Sample size (N={n_samples}) is small. Constrained ensemble trees (n_estimators=60, max_depth=5..7) and enforced tighter regularization to avoid memorization."
        })

    # Build condition-specific hyperparameter configurations for admitted candidates
    pruned_ids = {p["id"] for p in pruned_algorithms}
    
    for cand in candidate_pool:
        cid = cand["id"]
        if cid in pruned_ids:
            continue
            
        configs = []
        
        # --- Classification Models ---
        if cid == "logistic_regression":
            cw = "balanced" if is_imbalanced else None
            if ratio_np < 5.0 or n_features > 30:
                # High dimension: stronger regularization
                configs.append({"name": "Strong L2 Penalty (C=0.1)", "params": {"C": 0.1, "solver": "lbfgs", "class_weight": cw, "max_iter": 1000}})
                configs.append({"name": "Standard L2 Penalty (C=1.0)", "params": {"C": 1.0, "solver": "lbfgs", "class_weight": cw, "max_iter": 1000}})
            else:
                configs.append({"name": "Standard L2 (C=1.0)", "params": {"C": 1.0, "solver": "lbfgs", "class_weight": cw, "max_iter": 1000}})
                configs.append({"name": "Moderate L2 (C=5.0)", "params": {"C": 5.0, "solver": "lbfgs", "class_weight": cw, "max_iter": 1000}})

        elif cid == "random_forest":
            cw = "balanced" if is_imbalanced else None
            if n_samples < 400:
                configs.append({"name": "Shallow Forest (50 trees, depth 5)", "params": {"n_estimators": 50, "max_depth": 5, "min_samples_split": 5, "class_weight": cw, "random_state": 42}})
                configs.append({"name": "Moderate Forest (80 trees, depth 8)", "params": {"n_estimators": 80, "max_depth": 8, "min_samples_split": 4, "class_weight": cw, "random_state": 42}})
            else:
                configs.append({"name": "Standard Forest (100 trees, depth 10)", "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 3, "class_weight": cw, "random_state": 42}})
                configs.append({"name": "Deep Forest (150 trees, unconstrained)", "params": {"n_estimators": 150, "max_depth": None, "min_samples_split": 2, "class_weight": cw, "random_state": 42}})

        elif cid == "hist_gradient_boosting":
            cw = "balanced" if is_imbalanced else None
            if n_samples < 400:
                configs.append({"name": "Conservative Boosting (lr=0.05, max_iter=60)", "params": {"learning_rate": 0.05, "max_iter": 60, "max_depth": 4, "class_weight": cw, "random_state": 42}})
            else:
                configs.append({"name": "Standard Boosting (lr=0.1, max_iter=100)", "params": {"learning_rate": 0.1, "max_iter": 100, "max_depth": 6, "class_weight": cw, "random_state": 42}})
                configs.append({"name": "Deep Boosting (lr=0.08, max_iter=150)", "params": {"learning_rate": 0.08, "max_iter": 150, "max_depth": 8, "class_weight": cw, "random_state": 42}})

        elif cid == "extra_trees":
            cw = "balanced" if is_imbalanced else None
            n_est = 60 if n_samples < 400 else 100
            m_depth = 6 if n_samples < 400 else 12
            configs.append({"name": f"Extra Trees ({n_est} estimators, depth {m_depth})", "params": {"n_estimators": n_est, "max_depth": m_depth, "class_weight": cw, "random_state": 42}})

        elif cid == "svm":
            cw = "balanced" if is_imbalanced else None
            configs.append({"name": "RBF Kernel (C=1.0, scale)", "params": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": cw, "probability": True, "random_state": 42}})
            if n_features > 20:
                configs.append({"name": "Linear Kernel (C=0.5)", "params": {"C": 0.5, "kernel": "linear", "class_weight": cw, "probability": True, "random_state": 42}})

        elif cid == "knn":
            k = 5 if n_samples > 200 else 3
            configs.append({"name": f"Uniform KNN (k={k})", "params": {"n_neighbors": k, "weights": "uniform"}})
            configs.append({"name": f"Distance-weighted KNN (k={k+2})", "params": {"n_neighbors": k + 2, "weights": "distance"}})

        elif cid == "decision_tree":
            cw = "balanced" if is_imbalanced else None
            max_d = 4 if n_samples < 400 else 6
            configs.append({"name": f"Interpretable Shallow Tree (depth {max_d})", "params": {"max_depth": max_d, "min_samples_leaf": 4, "class_weight": cw, "random_state": 42}})

        elif cid == "mlp":
            if n_samples < 800:
                configs.append({"name": "Compact Architecture (32 units, L2 alpha=0.01)", "params": {"hidden_layer_sizes": (32,), "activation": "relu", "alpha": 0.01, "max_iter": 350, "random_state": 42}})
            else:
                configs.append({"name": "Deep Architecture (64-32 units, L2 alpha=0.001)", "params": {"hidden_layer_sizes": (64, 32), "activation": "relu", "alpha": 0.001, "max_iter": 400, "random_state": 42}})

        # --- Regression Models ---
        elif cid == "ridge_regression":
            configs.append({"name": "Ridge Regularization (alpha=1.0)", "params": {"alpha": 1.0, "random_state": 42}})
            if ratio_np < 5.0:
                configs.append({"name": "High Penalty Ridge (alpha=10.0)", "params": {"alpha": 10.0, "random_state": 42}})

        elif cid == "random_forest_reg":
            if n_samples < 400:
                configs.append({"name": "Constrained Forest (50 trees, depth 6)", "params": {"n_estimators": 50, "max_depth": 6, "min_samples_split": 4, "random_state": 42}})
            else:
                configs.append({"name": "Standard Forest (100 trees, depth 10)", "params": {"n_estimators": 100, "max_depth": 10, "min_samples_split": 3, "random_state": 42}})

        elif cid == "hist_gradient_boosting_reg":
            configs.append({"name": "Histogram Boosting (lr=0.1, max_iter=100)", "params": {"learning_rate": 0.1, "max_iter": 100, "max_depth": 6, "random_state": 42}})

        elif cid == "extra_trees_reg":
            configs.append({"name": "Extra Trees Regressor (80 trees)", "params": {"n_estimators": 80, "max_depth": 8, "random_state": 42}})

        elif cid == "svr":
            configs.append({"name": "SVR RBF Kernel (C=1.0, epsilon=0.1)", "params": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"}})

        elif cid == "knn_reg":
            configs.append({"name": "KNN Regressor (k=5)", "params": {"n_neighbors": 5, "weights": "distance"}})

        elif cid == "decision_tree_reg":
            configs.append({"name": "Decision Tree Regressor (depth 5)", "params": {"max_depth": 5, "min_samples_leaf": 4, "random_state": 42}})

        elif cid == "mlp_reg":
            configs.append({"name": "MLP Regressor (48 units)", "params": {"hidden_layer_sizes": (48,), "max_iter": 350, "random_state": 42}})

        selected_algorithms.append({
            "id": cand["id"],
            "name": cand["name"],
            "family": cand["family"],
            "configurations": configs,
            "config_count": len(configs),
            "selection_rationale": f"Admitted based on meta-feature compatibility with sample size N={n_samples} and P={n_features} features."
        })

    # Determine recommended evaluation metrics
    if not is_regression:
        if is_imbalanced:
            primary_metric = "f1_macro"
            metric_label = "Macro F1-Score (Adjusted for Imbalance)"
            secondary_metrics = ["roc_auc", "balanced_accuracy", "precision_macro", "recall_macro"]
        else:
            primary_metric = "accuracy"
            metric_label = "Classification Accuracy"
            secondary_metrics = ["f1_weighted", "precision_weighted", "recall_weighted"]
    else:
        primary_metric = "rmse"
        metric_label = "Root Mean Squared Error (RMSE)"
        secondary_metrics = ["r2", "mae", "mse"]

    cv_folds = 3 if n_samples > 12000 else 5

    return {
        "selected_algorithms": selected_algorithms,
        "pruned_algorithms": pruned_algorithms,
        "triggered_rules": triggered_rules,
        "optimization_summary": {
            "total_candidates_considered": len(candidate_pool),
            "retained_count": len(selected_algorithms),
            "pruned_count": len(pruned_algorithms),
            "pruning_efficiency_percent": round((len(pruned_algorithms) / len(candidate_pool)) * 100, 1),
            "recommended_primary_metric": primary_metric,
            "metric_label": metric_label,
            "secondary_metrics": secondary_metrics,
            "cv_folds": cv_folds
        }
    }
