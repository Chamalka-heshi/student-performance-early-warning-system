import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import shap
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="EduAlert",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom styling ----
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #1e3a5f; }
    .risk-card {
        padding: 20px; border-radius: 12px; margin-bottom: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    </style>
""", unsafe_allow_html=True)

# ---- Cached loaders ----
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

# ---- Role selection state ----
if 'role' not in st.session_state:
    st.session_state.role = None

# ================= LANDING PAGE =================
if st.session_state.role is None:
    st.markdown("<h1 style='text-align: center;'>🎓 EduAlert</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Select your portal to continue</p>", unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🧑‍🎓 Student")
        st.write("View your own academic status and personalized recommendations.")
        if st.button("Enter Student Portal", use_container_width=True):
            st.session_state.role = "student"
            st.rerun()
    with col2:
        st.markdown("### 👨‍🏫 Academic Staff")
        st.write("View at-risk students, class analytics, and prediction explanations.")
        if st.button("Enter Staff Portal", use_container_width=True):
            st.session_state.role = "staff"
            st.rerun()
    with col3:
        st.markdown("### 👨‍💼 Admin")
        st.write("Manage data and monitor system-wide statistics.")
        if st.button("Enter Admin Portal", use_container_width=True):
            st.session_state.role = "admin"
            st.rerun()

    st.info("💡 This is a portfolio demo — role selection simulates access control. A production version would use authenticated logins tied to a real user database.")

else:
    # ================= SIDEBAR NAVIGATION (defines `selected`) =================
    with st.sidebar:
        st.markdown(f"### Logged in as: **{st.session_state.role.title()}**")
        if st.button("⬅ Switch Role"):
            st.session_state.role = None
            st.rerun()
        st.markdown("---")

        if st.session_state.role == "student":
            selected = option_menu(
                "Student Portal", ["My Risk Status", "Recommendations"],
                icons=["speedometer2", "lightbulb"], default_index=0
            )
        elif st.session_state.role == "staff":
            selected = option_menu(
                "Staff Portal", ["At-Risk Students", "Class Analytics", "Predict & Explain"],
                icons=["exclamation-triangle", "bar-chart", "search"], default_index=0
            )
        elif st.session_state.role == "admin":
            selected = option_menu(
                "Admin Portal", ["System Overview", "Data Management"],
                icons=["gear", "database"], default_index=0
            )

    # ================= STUDENT PORTAL =================
    if st.session_state.role == "student":

        if selected == "My Risk Status":
            st.title("🧑‍🎓 My Academic Status")
            st.write("Enter your current information to see your academic risk status.")

            model, scaler = load_model_and_scaler()

            with st.form("student_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.slider("Age", 15, 18, 16)
                    study_time = st.slider("Weekly Study Time (hours)", 0.0, 20.0, 10.0)
                    absences = st.slider("Number of Absences This Term", 0, 29, 10)
                    tutoring = st.selectbox("Attending Tutoring?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                with col2:
                    parental_support = st.slider("Family Support Level (0=Low, 4=High)", 0, 4, 2)
                    extracurricular = st.selectbox("Extracurricular Activities?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                    sports = st.selectbox("Sports?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                    volunteering = st.selectbox("Volunteering?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

                submitted = st.form_submit_button("Check My Status", use_container_width=True)

            if submitted:
                high_absence_flag = 1 if absences > 15 else 0
                input_data = pd.DataFrame([{
                    'Age': age, 'Gender': 0, 'ParentalEducation': 2,
                    'StudyTimeWeekly': study_time, 'Absences': absences, 'Tutoring': tutoring,
                    'ParentalSupport': parental_support, 'Extracurricular': extracurricular,
                    'Sports': sports, 'Music': 0, 'Volunteering': volunteering,
                    'HighAbsenceFlag': high_absence_flag,
                    'Ethnicity_0': 1, 'Ethnicity_1': 0, 'Ethnicity_2': 0, 'Ethnicity_3': 0
                }])[FEATURE_COLUMNS]

                input_scaled = scaler.transform(input_data)
                prediction_probs = model.predict(input_scaled)[0]
                predicted_class = int(np.argmax(prediction_probs))
                st.session_state.last_prediction = predicted_class
                st.session_state.last_absences = absences
                st.session_state.last_study_time = study_time

                risk_labels = {0: "Excellent", 1: "Good", 2: "Moderate Risk", 3: "High Risk", 4: "Highest Risk"}
                risk_colors = {0: "🟢", 1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}

                st.markdown(f"## {risk_colors[predicted_class]} Status: {risk_labels[predicted_class]}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Absences", absences)
                col2.metric("Study Hours/Week", study_time)
                col3.metric("Support Level", parental_support)

                st.info("👉 Go to **Recommendations** in the sidebar for personalized suggestions based on your status.")

        elif selected == "Recommendations":
            st.title("💡 Personalized Recommendations")

            if 'last_prediction' not in st.session_state:
                st.warning("Please check your status first under 'My Risk Status'.")
            else:
                predicted_class = st.session_state.last_prediction
                absences = st.session_state.last_absences
                study_time = st.session_state.last_study_time

                recommendations = []
                if absences > 15:
                    recommendations.append("📅 Your attendance is a significant factor in your risk level. Aim to reduce absences going forward — consistent attendance is the strongest predictor of better outcomes.")
                if study_time < 10:
                    recommendations.append("📚 Consider increasing your weekly study time. While attendance matters most, additional study hours can still help.")
                if predicted_class >= 3:
                    recommendations.append("🎯 Consider scheduling a meeting with your academic advisor to discuss a support plan.")
                    recommendations.append("👥 Tutoring support is available and has shown a positive association with improved outcomes.")
                if not recommendations:
                    recommendations.append("✅ You're on a good track. Keep maintaining consistent attendance and study habits.")

                for rec in recommendations:
                    st.markdown(f"<div class='risk-card' style='background-color:white;'>{rec}</div>", unsafe_allow_html=True)

    # ================= STAFF PORTAL =================
    elif st.session_state.role == "staff":

        if selected == "At-Risk Students":
            st.title("⚠️ At-Risk Students")
            df = load_data()
            at_risk = df[df['GradeClass'] >= 3].sort_values('GradeClass', ascending=False)

            st.metric("Total At-Risk Students", len(at_risk))
            st.dataframe(
                at_risk[['Age', 'Absences', 'StudyTimeWeekly', 'ParentalSupport', 'Tutoring', 'GradeClass']].head(50),
                use_container_width=True
            )
            st.caption("Showing first 50 at-risk students (GradeClass 3 or 4), sorted by risk level.")

        elif selected == "Class Analytics":
            st.title("📊 Class Analytics")
            df = load_data()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Students", len(df))
            col2.metric("Pass Rate", f"{round((df['GradeClass'] < 3).mean() * 100, 1)}%")
            col3.metric("High Risk %", f"{round((df['GradeClass'] == 4).mean() * 100, 1)}%")
            col4.metric("Avg Absences", round(df['Absences'].mean(), 1))

            st.subheader("Grade Distribution")
            st.bar_chart(df['GradeClass'].value_counts().sort_index())

            st.subheader("Absences vs Grade Class")
            st.image("images/absences_by_gradeclass.png")

        elif selected == "Predict & Explain":
            st.title("🔍 Predict Student Risk")
            st.write("Enter a student's information to get a risk prediction with explanation.")
            model, scaler = load_model_and_scaler()

            col1, col2 = st.columns(2)
            with col1:
                age = st.slider("Age", 15, 18, 16, key="staff_age")
                gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female", key="staff_gender")
                ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3], format_func=lambda x: f"Group {x}", key="staff_eth")
                parental_education = st.slider("Parental Education (0-4)", 0, 4, 2, key="staff_pe")
                study_time = st.slider("Weekly Study Time (hours)", 0.0, 20.0, 10.0, key="staff_st")
                absences = st.slider("Number of Absences", 0, 29, 10, key="staff_abs")
            with col2:
                tutoring = st.selectbox("Tutoring", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="staff_tut")
                parental_support = st.slider("Parental Support (0-4)", 0, 4, 2, key="staff_ps")
                extracurricular = st.selectbox("Extracurricular", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="staff_ex")
                sports = st.selectbox("Sports", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="staff_sp")
                music = st.selectbox("Music", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="staff_mu")
                volunteering = st.selectbox("Volunteering", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="staff_vol")

            if st.button("Predict Risk", type="primary", key="staff_predict_btn"):
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

                prob_df = pd.DataFrame({'GradeClass': [0,1,2,3,4], 'Probability': prediction_probs})
                st.bar_chart(prob_df.set_index('GradeClass'))

                with st.spinner("Calculating explanation..."):
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

    # ================= ADMIN PORTAL =================
    elif st.session_state.role == "admin":

        if selected == "System Overview":
            st.title("⚙️ System Overview")
            df = load_data()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(df))
            col2.metric("Features Used", len(FEATURE_COLUMNS))
            col3.metric("Model Type", "Neural Network")

            st.subheader("Model Performance")
            comparison_data = pd.DataFrame({
                'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network'],
                'Accuracy': [0.62, 0.58, 0.67, 0.73],
                'Macro F1': [0.47, 0.43, 0.53, 0.56]
            })
            st.dataframe(comparison_data, use_container_width=True)

        elif selected == "Data Management":
            st.title("🗄️ Data Management")
            st.info("This section would allow uploading new semester data and retraining the model in a production system.")

            df = load_data()
            st.subheader("Current Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)

            st.subheader("Dataset Summary")
            st.write(df.describe())