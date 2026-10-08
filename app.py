import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load trained model and scaler
model = joblib.load("student_performance_model.pkl")
scaler = joblib.load("scaler.pkl")

# Header
st.title("🎓 Student Performance Prediction System")
st.markdown(
    "### Machine Learning based academic performance prediction"
)

st.write(
    "Enter the student's academic and behavioral information "
    "to predict their expected performance."
)

st.divider()

# Input section
st.subheader("📊 Student Information")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider(
        "Study Hours per Day",
        0.5, 10.0, 5.0, 0.5
    )

    attendance = st.slider(
        "Attendance (%)",
        40, 100, 75
    )

    previous_score = st.slider(
        "Previous Exam Score",
        20, 100, 65
    )

    assignment_score = st.slider(
        "Assignment Score",
        20, 100, 70
    )

with col2:
    sleep_hours = st.slider(
        "Sleep Hours",
        4.0, 10.0, 7.0, 0.5
    )

    screen_time = st.slider(
        "Daily Screen Time (hours)",
        1.0, 9.0, 4.0, 0.5
    )

    participation = st.slider(
        "Class Participation",
        1, 10, 5
    )

    assignments_completed = st.slider(
        "Assignments Completed (%)",
        40, 100, 75
    )

st.divider()

# Prediction
if st.button("🔮 Predict Performance", use_container_width=True):

    student = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Score": [previous_score],
        "Assignment_Score": [assignment_score],
        "Sleep_Hours": [sleep_hours],
        "Screen_Time": [screen_time],
        "Participation": [participation],
        "Assignments_Completed": [assignments_completed]
    })

    # Feature engineering
    student["Study_Efficiency"] = (
        student["Study_Hours"] /
        (student["Screen_Time"] + 1)
    )

    student["Academic_Average"] = (
        student["Previous_Score"] +
        student["Assignment_Score"]
    ) / 2

    student["Engagement_Score"] = (
        student["Attendance"] * 0.4 +
        student["Participation"] * 5 +
        student["Assignments_Completed"] * 0.2
    )

    # Scale features
    student_scaled = scaler.transform(student)

    # Prediction
    prediction = model.predict(student_scaled)[0]
    probability = model.predict_proba(student_scaled)[0]

    st.divider()
    st.subheader("📈 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        if prediction == 1:
            st.success("✅ Predicted Performance: PASS")
        else:
            st.error("❌ Predicted Performance: FAIL")

    with result_col2:
        st.metric(
            "Pass Probability",
            f"{probability[1]:.2%}"
        )

    st.progress(float(probability[1]))

    st.info(
        "Prediction generated using the trained machine learning "
        "model based on academic and behavioral features."
    )

# Footer
st.divider()

st.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit"
)
