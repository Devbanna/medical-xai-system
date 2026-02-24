import streamlit as st
import joblib
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
import shap
import pandas as pd

st.set_page_config(
    page_title="Cardio-Aura | High-Res XAI Suite",
    page_icon="🧬",
    layout="wide"
)

# --- Resource Loading ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
explainer = shap.TreeExplainer(model)

FEATURES = [
    "Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
    "Fasting BS", "Rest ECG", "Max HR",
    "Exercise Angina", "ST Depression", "Slope",
    "Major Vessels", "Thalassemia"
]

# --- Enhanced Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; border-right: 1px solid #30363d; }
    
    .metric-container { 
        background: #1c2128; 
        border: 2px solid #30363d; 
        padding: 40px 20px; 
        border-radius: 15px; 
        text-align: center;
    }
    
    .risk-label { color: #8b949e; font-size: 1.2rem; font-weight: bold; letter-spacing: 2px; }
    .risk-value-high { color: #ff3131; font-size: 5rem; font-weight: 900; line-height: 1; }
    .risk-value-low { color: #00ff41; font-size: 5rem; font-weight: 900; line-height: 1; }
    
    .logic-card { 
        background: rgba(88, 166, 255, 0.08); 
        border-left: 5px solid #58a6ff; 
        padding: 20px; 
        border-radius: 8px; 
        margin: 15px 0 25px 0;
        font-size: 1.05rem;
    }
    h1, h2, h3 { color: #58a6ff !important; margin-top: 40px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 Cardio-Aura Intelligence")
st.markdown("#### Clinical Decision Support System with Integrated Transparency Reports")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Patient Input Portal")
    age = st.slider("Age", 20, 80, 50)
    sex = st.radio("Biological Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    
    with st.expander("Vitals & Lab Data", expanded=True):
        trestbps = st.number_input("Resting BP (mmHg)", 90, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        thalach = st.slider("Max Heart Rate", 70, 210, 150)
    
    with st.expander("Diagnostic Imaging", expanded=False):
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical", "Non-anginal", "Asymptomatic"][x])
        exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed", "Reversible"][x-1])
        fbs, restecg, slope = 0, 0, 1

# --- Prediction Engine ---
input_raw = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
input_scaled = scaler.transform(input_raw)
prob = model.predict_proba(input_scaled)[0][1]

# --- Top Large Metrics ---
m1, m2 = st.columns(2)
with m1:
    risk_class = "risk-value-high" if prob > 0.5 else "risk-value-low"
    st.markdown(f"<div class='metric-container'><p class='risk-label'>CARDIAC RISK SCORE</p><p class='{risk_class}'>{prob*100:.1f}%</p></div>", unsafe_allow_html=True)
with m2:
    status = "CRITICAL" if prob > 0.5 else "STABLE"
    status_class = "risk-value-high" if prob > 0.5 else "risk-value-low"
    st.markdown(f"<div class='metric-container'><p class='risk-label'>PATIENT STATUS</p><p class='{status_class}'>{status}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# --- CHART 1: GAUGE METER ---
st.subheader("🚀 Clinical Risk Meter")
st.markdown("""
<div class='logic-card'>
<b>Clinical Interpretation:</b> This meter translates the AI's internal probability calculation into a standard medical warning system. 
<ul>
    <li><b>0-30% (Green):</b> Indicators align with healthy cardiovascular baseline.</li>
    <li><b>30-70% (Yellow):</b> Mild correlation with high-risk phenotypes; further non-invasive testing suggested.</li>
    <li><b>70-100% (Red):</b> Significant similarity to confirmed heart disease cases; immediate intervention prioritized.</li>
</ul>
</div>
""", unsafe_allow_html=True)

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number", value=prob * 100,
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "white"},
           'steps': [{'range': [0, 30], 'color': "#00ff41"}, 
                     {'range': [30, 70], 'color': "#ffab00"}, 
                     {'range': [70, 100], 'color': "#ff3131"}]}
))
fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=450)
st.plotly_chart(fig_gauge, use_container_width=True)

# --- CHART 2: RADAR CHART ---
st.subheader("📊 Phenotype Comparison Profile")
st.markdown("""
<div class='logic-card'>
<b>Functional Interpretation:</b> Unlike a single score, this radar plot shows <i>where</i> the patient is deviating. 
The <b>Blue Shape</b> (Patient) is overlaid on a <b>Gray Baseline</b> (Healthy Average). 
If the blue area pushes toward 'ST Stress' or 'BP', it identifies the heart's functional weak points.
</div>
""", unsafe_allow_html=True)

categories = ['Age', 'BP', 'Cholesterol', 'Heart Rate', 'ST Stress']
patient_vitals = [age/80, trestbps/200, chol/600, thalach/210, oldpeak/6]
dataset_avg = [0.6, 0.65, 0.45, 0.7, 0.15]
fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(r=patient_vitals, theta=categories, fill='toself', name='Patient', line_color='#58a6ff'))
fig_radar.add_trace(go.Scatterpolar(r=dataset_avg, theta=categories, fill='toself', name='Healthy Avg', line_color='#8b949e', opacity=0.5))
fig_radar.update_layout(polar=dict(bgcolor='rgba(255,255,255,0.05)', radialaxis=dict(visible=False)),
                        paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=600)
st.plotly_chart(fig_radar, use_container_width=True)

# --- CHART 3: SHAP CHART ---
st.subheader("🧠 Transparency Report: Neural Attribution")
st.markdown("""
<div class='logic-card'>
<b>Explainable AI (XAI) Logic:</b> This chart breaks the 'Black Box' of the model. 
It uses Game Theory (SHAP) to show exactly which clinical biomarkers (Age, BP, etc.) pushed the risk score up or down.
<ul>
    <li><b>Red Bars (Right):</b> Biomarkers that <b>increased</b> patient risk.</li>
    <li><b>Green Bars (Left):</b> Biomarkers that acted as <b>protective</b> factors.</li>
</ul>
</div>
""", unsafe_allow_html=True)

raw_shap = explainer.shap_values(input_scaled)
contributions = raw_shap[1].flatten() if isinstance(raw_shap, list) else raw_shap.flatten()
fig_xai = px.bar(x=contributions[:13], y=FEATURES, orientation='h', 
                 color=contributions[:13], color_continuous_scale='RdYlGn_r',
                 template="plotly_dark", labels={'x': 'Impact Magnitude', 'y': 'Biomarker'})
fig_xai.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=650)
st.plotly_chart(fig_xai, use_container_width=True)

# Footer Glossary
with st.expander("📖 Medical Glossary & Input Guide"):
    st.write("**Resting BP:** Tension in the arteries while the heart is relaxed. Chronic high BP leads to structural heart damage.")
    st.write("**ST Depression (Oldpeak):** A critical ECG marker; indicates if the heart muscle is oxygen-starved during exertion.")
    st.write("**Thalassemia:** An inherited blood disorder that affects how oxygen is carried in the blood, influencing heart workload.")