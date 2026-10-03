import streamlit as st
import joblib
import numpy as np

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page Configuration
st.set_page_config(
    page_title="Score Prediction System",
    layout="centered"
)

st.title("Score Prediction System Based On Hours")
st.write("Predict the expected score based on the number of hours studied per day..")

st.divider()

# Input Section
st.subheader("Student Details")

hours = st.number_input(
    "Enter Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

st.write(f"**Study Hours:** {hours:.1f} hours/day")

# Prediction Button
if st.button("Predict Score", type="primary"):

    # Convert input to NumPy array
    data = np.array([[hours]])

    # # Scale input using trained scaler
    # scaled_data = scaler.transform(data)

    # Make prediction
    prediction = model.predict(data)

    predicted_score = float(prediction[0])

    # Keep score between 0 and 100
    predicted_score = max(0.0, min(100.0, predicted_score))

    # Display Prediction
    st.divider()

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Score: **{predicted_score:.2f} / 100**"
    )

    # Performance Message
    if predicted_score >= 85:
        st.info("Excellent expected performance!")
        st.balloons()

    elif predicted_score >= 70:
        st.info("Good expected performance.")

    elif predicted_score >= 50:
        st.warning(
            "Average expected performance. "
            "More study may help improve the score."
        )

    else:
        st.error(
            "More study is recommended to improve "
            "the expected score."
        )


    st.subheader("Prediction Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Study Hours",
            f"{hours:.1f} hrs/day"
        )

    with col2:
        st.metric(
            "Predicted Score",
            f"{predicted_score:.2f}"
        )

# Command to run: C:\Python314\python.exe -m streamlit run filename.py