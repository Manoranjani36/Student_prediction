import streamlit as st
import pandas as pd
import joblib, os

st.set_page_config(page_title="Student Prediction",page_icon="🎓")
st.title("🎓 Student Prediction")
st.write("Enter student details to generate a prediction.")

if not os.path.exists("models/student_prediction_model.pkl"):
    st.error("Model not found. Run train_model.py first.")
    st.stop()

model=joblib.load("models/student_prediction_model.pkl")
meta=joblib.load("models/metadata.pkl")
data=pd.read_csv("student_dataset_10000_rows.csv")

st.info(f"Task: {meta['problem_type']} | Target: {meta['target']} | Model: {meta['best_model']}")

values={}
for col in meta["feature_columns"]:
    label=col.replace("_"," ").title()
    if col in meta["categorical_features"]:
        opts=data[col].dropna().astype(str).unique().tolist()
        values[col]=st.selectbox(label,opts or ["Unknown"])
    else:
        s=pd.to_numeric(data[col],errors="coerce").dropna()
        mn=float(s.min()) if len(s) else 0.0
        mx=float(s.max()) if len(s) else 100.0
        med=float(s.median()) if len(s) else 0.0
        values[col]=st.number_input(label,min_value=mn,max_value=mx,value=med)

if st.button("🔮 Predict",use_container_width=True):
    pred=model.predict(pd.DataFrame([values]))[0]
    st.success(f"Prediction: {pred}")
    st.info("Educational machine-learning project.")

st.caption("Student Prediction • Machine Learning Project")
