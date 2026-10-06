"""Sample datasets provider for AutoOptML."""
from typing import Dict, Any
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes, fetch_california_housing


def get_available_datasets() -> list[dict]:
    """Returns metadata for available sample datasets."""
    return [
        {
            "id": "breast_cancer",
            "name": "Breast Cancer Wisconsin",
            "task": "binary_classification",
            "samples": 569,
            "features": 30,
            "description": "High-dimensional binary classification task for malignant vs benign diagnosis.",
            "target_col": "target"
        },
        {
            "id": "iris",
            "name": "Iris Flower Classification",
            "task": "multiclass_classification",
            "samples": 150,
            "features": 4,
            "description": "Classic multiclass dataset with 3 flower species and continuous features.",
            "target_col": "species"
        },
        {
            "id": "wine",
            "name": "Wine Recognition",
            "task": "multiclass_classification",
            "samples": 178,
            "features": 13,
            "description": "Chemical analysis of wines grown in the same region in Italy derived from three cultivars.",
            "target_col": "cultivar"
        },
        {
            "id": "diabetes",
            "name": "Diabetes Progression",
            "task": "regression",
            "samples": 442,
            "features": 10,
            "description": "Predict disease progression one year after baseline from physiological variables.",
            "target_col": "disease_progression"
        },
        {
            "id": "california_housing",
            "name": "California Housing Prices",
            "task": "regression",
            "samples": 1000,  # sampled for fast interactive demonstration
            "features": 8,
            "description": "Predict median house values in California districts from census metrics.",
            "target_col": "median_house_value"
        },
        {
            "id": "customer_churn",
            "name": "Customer Churn (Mixed Types & Imbalanced)",
            "task": "binary_classification",
            "samples": 800,
            "features": 10,
            "description": "Realistic business tabular dataset with categorical, numeric, and missing values.",
            "target_col": "churned"
        }
    ]


def load_sample_dataset(dataset_id: str) -> tuple[pd.DataFrame, str]:
    """Loads a sample dataset into a pandas DataFrame and returns (df, default_target_col)."""
    if dataset_id == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        # map targets to human labels
        df["target"] = df["target"].map({0: "malignant", 1: "benign"})
        return df, "target"

    elif dataset_id == "iris":
        data = load_iris(as_frame=True)
        df = data.frame.copy()
        df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
        df = df.drop(columns=["target"])
        return df, "species"

    elif dataset_id == "wine":
        data = load_wine(as_frame=True)
        df = data.frame.copy()
        df["cultivar"] = df["target"].map({0: "Cultivar_A", 1: "Cultivar_B", 2: "Cultivar_C"})
        df = df.drop(columns=["target"])
        return df, "cultivar"

    elif dataset_id == "diabetes":
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df = df.rename(columns={"target": "disease_progression"})
        return df, "disease_progression"

    elif dataset_id == "california_housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame.sample(n=1000, random_state=42).reset_index(drop=True)
        df = df.rename(columns={"MedHouseVal": "median_house_value"})
        return df, "median_house_value"

    elif dataset_id == "customer_churn":
        # Generate a realistic mixed tabular dataset with missing values and imbalance
        import numpy as np
        np.random.seed(42)
        n = 800
        ages = np.random.randint(18, 75, size=n)
        tenure = np.random.randint(0, 72, size=n)
        monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n), 2)
        total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 15, size=n), 2)
        total_charges = np.maximum(total_charges, 0)
        contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
        payment_method = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n)
        tech_support = np.random.choice(["Yes", "No", "No internet"], size=n, p=[0.3, 0.5, 0.2])
        internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.4, 0.4, 0.2])
        
        # calculate churn probability based on features
        logits = (
            -1.2 
            + (contract == "Month-to-month") * 1.3 
            + (monthly_charges > 70) * 0.8 
            - (tenure > 24) * 0.9
            + (tech_support == "No") * 0.5
        )
        probs = 1 / (1 + np.exp(-logits))
        churn = (np.random.rand(n) < probs).astype(int)
        
        df = pd.DataFrame({
            "CustomerID": [f"USR_{10000+i}" for i in range(n)],
            "Age": ages,
            "TenureMonths": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "TechSupport": tech_support,
            "InternetService": internet_service,
            "churned": ["Yes" if c == 1 else "No" for c in churn]
        })
        
        # introduce realistic missing values in ~3% of TotalCharges and TechSupport
        mask_miss_num = np.random.rand(n) < 0.04
        df.loc[mask_miss_num, "TotalCharges"] = np.nan
        mask_miss_cat = np.random.rand(n) < 0.03
        df.loc[mask_miss_cat, "TechSupport"] = np.nan
        
        return df, "churned"

    else:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")
