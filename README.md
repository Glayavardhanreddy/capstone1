# AutoOptML: Conditional Search-Space Optimiser for Algorithm & Architecture Selection

An AI-based web application that automates machine learning algorithm selection and hyperparameter configuration conditioned on dataset meta-features.

---

## 📌 Project Abstract
> The Conditional Search-Space Optimiser for Algorithm and Architecture Selection is an AI-based web application that helps users choose the most suitable machine learning algorithm and configuration for their dataset. The user uploads a dataset, and the system automatically analyses its features, data types, missing values, size, and problem type such as classification or regression. Based on this analysis, it selects suitable algorithms and generates different configurations for testing. These models are trained and evaluated using appropriate performance metrics, and their results are compared to recommend the best-performing model. The system uses Python, Scikit-learn, and FastAPI to provide an easy-to-use platform for automated model selection. It reduces unnecessary model testing, minimizes manual effort, and helps users make faster and more reliable machine learning decisions.

---

## 🚀 Key Features

1. **Dataset Ingestion & Auto-Detection**:
   - Supports **CSV** and **Excel** (`.xlsx`, `.xls`) file uploads.
   - Built-in instant benchmark datasets: Breast Cancer (Binary, high-dim), Customer Churn (Mixed types, imbalanced), California Housing (Regression), Iris (Multiclass), Wine Cultivars, and Diabetes.
   - Interactive data preview table showing the first 10 rows and attribute profiles.
   - Intelligent target variable and task type auto-detection (Binary Classification, Multiclass Classification, Regression) with user overrides.

2. **Automated Meta-Feature Extraction & Profiling**:
   - Computes dimensional properties ($N$ samples, $P$ features, $N/P$ ratio).
   - Inferred column roles (Numeric, Categorical, ID/Key, Datetime, High-cardinality text).
   - Missing value percentage & cell counts with automated imputation strategies.
   - Target balance ratio ($M_{maj} / M_{min}$) and skewness analysis.
   - Automated diagnostic alerts (e.g. data leakage warnings on IDs, curse of dimensionality warnings).

3. **Conditional Search-Space Optimiser (Core Innovation)**:
   - **Scale Pruning**: Automatically prunes $O(N^2)$ to $O(N^3)$ algorithms (e.g. standard RBF SVM, KNN) on large datasets ($N > 8,000$) to eliminate training bottlenecks.
   - **Neural Starvation Pruning**: Prunes Multi-Layer Perceptrons on small datasets ($N < 250$) where representation starvation leads to overfitting.
   - **Curse of Dimensionality Rule**: Prunes Euclidean distance methods when $P > 40$ or $N/P < 3.0$ in favor of tree feature-subsampling and penalized linear models.
   - **Imbalance Conditioning**: Injects `class_weight='balanced'` and automatically retargets metric optimization to **Macro F1** and **ROC-AUC**.
   - **Dynamic Architecture Generation**: Rather than blind brute-force grid searches, conditioned hyperparameter sets (shallow vs deep ensembles, regularized vs unregularized penalties) are generated based on the dataset profile.

4. **Automated Scikit-Learn Training & Benchmarking Engine**:
   - Builds complete end-to-end `ColumnTransformer` pipelines with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.
   - Stratified $K$-Fold cross-validation across all admitted models.
   - Comparative Leaderboard ranked by primary metric with "Top Performer" and "Fastest" badges.
   - Interactive Chart.js visualizations:
     - Performance Score vs Training Time Trade-off bar chart.
     - Confusion Matrix Doughnut breakdown (for classification) or Actual vs Predicted Residual scatter plot (for regression).
     - Top Feature Importance ranking bar chart (MDI / absolute coefficients).

5. **Production Model Export & Inference Code Generator**:
   - 1-Click serialized pipeline download (`.joblib`).
   - Automatically generated copy-pasteable Python script to load the saved bundle and run inference on raw data.

---

## 🛠️ Technology Stack
- **Backend**: Python 3.11, FastAPI, Uvicorn, Scikit-learn, Pandas, NumPy, Joblib.
- **Frontend**: Single Page Application, Tailwind CSS, Chart.js, Lucide Icons.

---

## 💻 How to Run the Application

1. Open a terminal in the project directory:
   ```bash
   cd c:\capstone
   ```

2. Run the application launcher:
   ```bash
   py run.py
   ```
   Or run directly with Uvicorn:
   ```bash
   py -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

---

## 🧪 Running Automated Tests
Run the unit test suites to verify data profiling, search-space pruning, training, and API endpoints:
```bash
py tests/test_pipeline.py
py tests/test_api.py
```
