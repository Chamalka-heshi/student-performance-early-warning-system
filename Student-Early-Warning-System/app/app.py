import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import shap

st.set_page_config(
    page_title="Student Early Warning System",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/students_cleaned.csv")

@st.cache_resource
def load_model_and_scaler():
    model = load_model("models/neural_network.keras")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

@st.cache_resource
def load_explainer(_model, background_data):
    return shap.KernelExplainer(_model.predict, background_data)

FEATURE_COLUMNS = [
    'Age', 'Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
    'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music',
    'Volunteering', 'HighAbsenceFlag', 'Ethnicity_0', 'Ethnicity_1',
    'Ethnicity_2', 'Ethnicity_3'
]

st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Risk", "Analytics"])

if page == "Home":
    st.title("🎓 Student Performance Early Warning System")
    st.write("An ML-powered system to identify students at risk of academic failure before final exams.")

    df = load_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(df))
    col2.metric("Avg. GPA", round(df['GPA'].mean(), 2))
    col3.metric("Avg. Absences", round(df['Absences'].mean(), 1))

    high_risk_count = len(df[df['GradeClass'] == 4.0])
    high_risk_pct = round(high_risk_count / len(df) * 100, 1)
    col4.metric("High Risk Students", f"{high_risk_count} ({high_risk_pct}%)")

    st.subheader("Grade Class Distribution")
    grade_counts = df['GradeClass'].value_counts().sort_index()
    st.bar_chart(grade_counts)

    st.caption("GradeClass: 0 = Best performing, 4 = Highest risk")

elif page == "Predict Risk":
    st.title("🔍 Predict Student Risk")
    st.write("Enter student information below to get a risk prediction with explanation.")

    model, scaler = load_model_and_scaler()

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 15, 18, 16)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3], format_func=lambda x: f"Group {x}")
        parental_education = st.slider("Parental Education Level (0=None, 4=Highest)", 0, 4, 2)
        study_time = st.slider("Weekly Study Time (hours)", 0.0, 20.0, 10.0)
        absences = st.slider("Number of Absences", 0, 29, 10)

    with col2:
        tutoring = st.selectbox("Tutoring", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        parental_support = st.slider("Parental Support Level (0=None, 4=Highest)", 0, 4, 2)
        extracurricular = st.selectbox("Extracurricular Activities", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        sports = st.selectbox("Sports", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        music = st.selectbox("Music", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        volunteering = st.selectbox("Volunteering", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    if st.button("Predict Risk", type="primary"):
        high_absence_flag = 1 if absences > 15 else 0
        ethnicity_onehot = [1 if ethnicity == i else 0 for i in range(4)]

        input_data = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'ParentalEducation': parental_education,
            'StudyTimeWeekly': study_time, 'Absences': absences, 'Tutoring': tutoring,
            'ParentalSupport': parental_support, 'Extracurricular': extracurricular,
            'Sports': sports, 'Music': music, 'Volunteering': volunteering,
            'HighAbsenceFlag': high_absence_flag,
            'Ethnicity_0': ethnicity_onehot[0], 'Ethnicity_1': ethnicity_onehot[1],
            'Ethnicity_2': ethnicity_onehot[2], 'Ethnicity_3': ethnicity_onehot[3]
        }])[FEATURE_COLUMNS]

        input_scaled = scaler.transform(input_data)
        prediction_probs = model.predict(input_scaled)[0]
        predicted_class = int(np.argmax(prediction_probs))

        risk_labels = {0: "Excellent (Lowest Risk)", 1: "Good", 2: "Moderate Risk", 3: "High Risk", 4: "Highest Risk"}

        st.subheader("Prediction Result")
        if predicted_class >= 3:
            st.error(f"⚠️ {risk_labels[predicted_class]} — GradeClass {predicted_class}")
        elif predicted_class == 2:
            st.warning(f"⚠️ {risk_labels[predicted_class]} — GradeClass {predicted_class}")
        else:
            st.success(f"✅ {risk_labels[predicted_class]} — GradeClass {predicted_class}")

        st.write("**Prediction confidence per class:**")
        prob_df = pd.DataFrame({'GradeClass': [0, 1, 2, 3, 4], 'Probability': prediction_probs})
        st.bar_chart(prob_df.set_index('GradeClass'))

        with st.spinner("Calculating explanation... (this takes a moment)"):
            df_full = load_data().drop(columns=['StudentID', 'GPA', 'GradeClass'], errors='ignore')
            background_sample = scaler.transform(df_full.sample(30, random_state=42))
            explainer = load_explainer(model, background_sample)
            shap_values_single = explainer.shap_values(input_scaled, nsamples=100)

        shap_for_predicted_class = shap_values_single[:, :, predicted_class][0]
        contributions = pd.DataFrame({
            'Feature': FEATURE_COLUMNS,
            'Impact': shap_for_predicted_class
        }).sort_values('Impact', key=abs, ascending=False).head(5)

        st.subheader("Top Contributing Factors")
        for _, row in contributions.iterrows():
            direction = "increases" if row['Impact'] > 0 else "decreases"
            st.write(f"- **{row['Feature']}** {direction} predicted risk (impact: {row['Impact']:.3f})")

elif page == "Analytics":
    st.title("📊 Analytics & Model Insights")

    df = load_data()

    tab1, tab2, tab3 = st.tabs(["Exploratory Analysis", "Model Comparison", "Feature Importance"])

    with tab1:
        st.subheader("Grade Class Distribution")
        st.image("images/gradeclass_distribution.png")

        st.subheader("Absences by Grade Class")
        st.image("images/absences_by_gradeclass.png")

        st.subheader("Study Time by Grade Class")
        st.image("images/studytime_by_gradeclass.png")

        st.subheader("Correlation Heatmap")
        st.image("images/correlation_heatmap.png")

    with tab2:
        st.subheader("Model Performance Comparison")
        comparison_data = pd.DataFrame({
            'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network'],
            'Accuracy': [0.62, 0.58, 0.67, 0.73],
            'Macro F1': [0.47, 0.43, 0.53, 0.56]
        })
        st.dataframe(comparison_data, use_container_width=True)
        st.bar_chart(comparison_data.set_index('Model')[['Accuracy', 'Macro F1']])
        st.caption("Neural Network selected as final model — best accuracy, macro F1, and performance on high-risk classes (3.0, 4.0).")

    with tab3:
        st.subheader("What Drives Risk Predictions?")
        st.image("images/shap_summary_class4.png")
        st.write("""
        Based on SHAP analysis, the top factors driving high-risk (GradeClass 4.0) predictions are:
        1. **Absences** — the dominant factor
        2. **StudyTimeWeekly**
        3. **HighAbsenceFlag** (engineered feature)
        4. **ParentalSupport**
        """)