# Cardio-Aura: Clinical Decision Support with Explainable AI (XAI)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-xai-system-o2earsxkbizhjiefts2v68.streamlit.app/)

Cardio-Aura is a professional-grade diagnostic tool designed to predict heart disease risk. Unlike standard "black-box" models, this system focuses on **Model Interpretability**, providing clear, visual explanations for every prediction using Game Theory (SHAP).

---


## 🩺 The Dashboard in Action

![Main Dashboard Screenshot](./images/dashboard_main.png)


---

## 💡 The Motivation
In the medical field, a prediction without an explanation is a liability. During my B.Tech specialization in AIML, I realized that for AI to be trusted by doctors, it must be transparent. I built **Cardio-Aura** to bridge the gap between complex Random Forest algorithms and clinical decision-making. 

This project aligns with the **EU AI Act’s** requirements for transparency in high-risk AI applications—a key area of interest for my upcoming Master's studies in Germany.

---

## 🛠️ Technical Architecture
- **Machine Learning:** Random Forest Classifier (Scikit-Learn)
- **Interpretability Layer:** SHAP (Shapley Additive Explanations)
- **Interface:** Streamlit (Optimized with custom CSS for a High-Contrast Medical UI)
- **Visual Analytics:** Plotly (Gauge & Radar Charts) and Matplotlib

---

## 🧠 Engineering Challenges

### 1. The Additivity Conflict

One of the main technical issues encountered was the error:

`ExplainerError: Additivity check failed`

**Problem**  
The sum of SHAP feature contributions did not perfectly match the model's predicted output.  
This discrepancy was caused by small floating-point rounding differences introduced during the feature scaling process between training and inference pipelines.

**Solution**  
I refactored the data preprocessing pipeline to ensure the scaling parameters used during inference were identical to those used during training. Additionally, I implemented robust error-handling logic to safely manage rare floating-point mismatches during SHAP explanation generation.

---

### 2. Handling Multi-Dimensional Tensors

During development, I encountered the following recurring error:

`IndexError: list index out of range`

**Problem**  
SHAP returns outputs in different formats depending on the model configuration and number of prediction classes.  
In some cases the output was a list, while in others it was a 3D tensor, which caused indexing failures during visualization.

**Solution**  
To solve this, I implemented a **shape-agnostic backend handler** that dynamically detects the SHAP output structure. The system automatically extracts the relevant tensor slice corresponding to the **Disease Risk** prediction class, ensuring the visualization pipeline remains stable even if the underlying model changes.

---

### 3. Bridging the Medical Knowledge Gap

While building the dashboard, I recognized that many clinical variables might not be intuitive for general users.

**Problem**  
Medical terms such as **ST Depression** or **Thalassemia** may not be immediately understandable to non-medical users, reducing the interpretability of the dashboard.

**Solution**  
I integrated a **Clinical Lexicon** directly into the interface to provide contextual explanations for key medical features.  
Additionally, I designed a **Phenotype Radar Chart** that compares patient vitals against a healthy reference profile, enabling users to quickly interpret deviations through visual benchmarking.

![Radar Chart Phenotyping](./images/radar_chart.png)


---

## 📊 How to Interpret the Data

1. **Risk Gauge:** A "speedometer" for cardiac urgency. Red indicates high similarity to confirmed heart disease cases.
2. **Radar Comparison:** Shows **where** the patient deviates from the norm. If the blue area pushes toward 'BP' or 'Cholesterol', those are the specific clinical targets.
3. **Neural Attribution (SHAP):** - **Red Bars:** Factors pushing the risk **up**.
   - **Green Bars:** Protective factors pushing the risk **down**.

![SHAP Feature Importance](./images/shap_plot.png)


---

## 🚀 Deployment Instructions

### Prerequisites
- Python 3.10+
- Virtual Environment (Conda or venv)

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Devbanna/medical-xai-system.git 
   cd medical-xai-system

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Run the Dashboard:**
    ```bash
    streamlit run app/app.py
## 📁 Project Structure

```text
medical-xai-system/
├── app/
│   └── app.py              # Main Dashboard Logic
├── models/
│   ├── rf_model.pkl        # Trained Random Forest Classifier
│   └── scaler.pkl          # Feature Scaling Pipeline
├── data/
│   └── heart.csv           # UCI Heart Disease Dataset
├── docs/
│   └── model_report.md     # Technical Performance Metrics
├── images/                 # UI/UX and XAI Visual Assets
├── requirements.txt        # Dependency Mapping
└── README.md               # Engineering Journey