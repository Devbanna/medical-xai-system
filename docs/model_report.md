# Technical Analysis: Heart Disease Prediction Model

This report details the machine learning methodology, evaluation metrics, and clinical reasoning used to develop the Cardio-Aura predictive engine.

## 1. Dataset Overview
The model is trained on an expanded Heart Disease dataset consisting of **1,025 patient records** with 13 clinical features and 1 target variable.

### Preprocessing Pipeline:
- **Feature Scaling:** We applied `StandardScaler` to ensure that features with different magnitudes (e.g., Cholesterol vs. ST Depression) contribute equally to the model's decision-making.
- **Stratified Split:** Data was partitioned using an **80/20 train-test split**. We utilized **stratification** to ensure that the proportion of diseased vs. healthy cases remained consistent across both subsets, preventing bias.

## 2. Model Selection: Random Forest Classifier
A Random Forest was selected as the primary estimator. Its ensemble nature (bagging) provides robust generalization, which is crucial for medical diagnostics where high variance can lead to misdiagnosis.

## 3. Evaluation Metrics
The model was tested on a hold-out set of **205 samples**. The results indicate a highly optimized fit for this specific dataset.

| Metric | Score | Clinical Significance |
| :--- | :--- | :--- |
| **Accuracy** | 100% | The percentage of total correct predictions. |
| **Precision** | 100% | Zero False Positives; no healthy patients were incorrectly flagged as high-risk. |
| **Recall** | 100% | **Critical:** Zero False Negatives; the model successfully identified every single sick patient. |
| **F1-Score** | 100% | The perfect balance between Precision and Recall. |

> **Note:** While a 100% score is achieved on this dataset, real-world clinical deployment would require further testing on external, non-augmented data to ensure robustness against edge cases.

## 4. Feature Importance & Interpretability
During the training phase (documented in `notebooks/01_data_exploration.ipynb`), we focused on identifying the primary clinical drivers. The dashboard uses **SHAP (Shapley Additive Explanations)** to explain how these global features impact an individual patient's risk score in real-time.

---
*Developed by Dev Pratap Singh Tomar*