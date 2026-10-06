import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Canteen Food Waste Predictor",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Smart Canteen Food Waste Predictor")
st.write("Upload daily sales data and predict tomorrow's food requirement.")

# -----------------------------
# CSV Upload
# -----------------------------
st.sidebar.header("📂 Daily Sales Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload daily sales CSV",
    type=["csv"]
)

# -----------------------------
# Sample Data
# -----------------------------
sample_data = {
    "students": [
        100, 150, 200, 250, 300,
        180, 220, 280, 350, 400,
        120, 170, 230, 270, 320,
        190, 240, 290, 360, 420
    ],

    "previous_orders": [
        80, 130, 180, 220, 270,
        150, 190, 250, 310, 360,
        100, 145, 200, 240, 285,
        160, 210, 260, 320, 380
    ],

    "day": [
        1, 2, 3, 4, 5,
        6, 7, 1, 2, 3,
        4, 5, 6, 7, 1,
        2, 3, 4, 5, 6
    ],

    "meals_prepared": [
        85, 135, 185, 225, 275,
        155, 195, 255, 315, 365,
        105, 150, 205, 245, 290,
        165, 215, 265, 325, 385
    ]
}

df = pd.DataFrame(sample_data)

# -----------------------------
# Read Uploaded CSV
# -----------------------------
if uploaded_file is not None:

    try:
        uploaded_df = pd.read_csv(uploaded_file)

        st.success("✅ Sales data uploaded successfully!")

        st.subheader("📊 Uploaded Daily Sales Data")

        st.dataframe(
            uploaded_df,
            use_container_width=True
        )

        # Check required columns
        required_columns = [
            "students",
            "previous_orders",
            "day",
            "meals_prepared"
        ]

        if all(col in uploaded_df.columns for col in required_columns):

            df = uploaded_df

            st.info(
                f"Using {len(df)} uploaded records for ML training."
            )

        else:
            st.error(
                "CSV must contain these columns: "
                "students, previous_orders, day, meals_prepared"
            )

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# -----------------------------
# Train Model
# -----------------------------
X = df[
    [
        "students",
        "previous_orders",
        "day"
    ]
]

y = df["meals_prepared"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# -----------------------------
# Prediction Inputs
# -----------------------------
st.sidebar.header("🔮 Prediction")

students = st.sidebar.number_input(
    "Expected Students",
    min_value=10,
    max_value=2000,
    value=300
)

previous_orders = st.sidebar.number_input(
    "Previous Day Orders",
    min_value=0,
    max_value=2000,
    value=270
)

day_name = st.sidebar.selectbox(
    "Day",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

day_mapping = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
}

day = day_mapping[day_name]

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("🔮 Predict Food Requirement"):

    input_data = pd.DataFrame({
        "students": [students],
        "previous_orders": [previous_orders],
        "day": [day]
    })

    prediction = model.predict(input_data)[0]

    meals_required = round(prediction)

    # 5% safety buffer
    recommended = round(meals_required * 1.05)

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Expected Students",
            students
        )

    with col2:
        st.metric(
            "Predicted Meals",
            meals_required
        )

    with col3:
        st.metric(
            "Recommended Preparation",
            recommended
        )

    # -----------------------------
    # Waste Analysis
    # -----------------------------
    st.subheader("♻️ Food Waste Analysis")

    estimated_extra = recommended - meals_required

    st.write(
        f"Estimated extra meals: **{estimated_extra}**"
    )

    if estimated_extra <= 10:

        st.success(
            "✅ Food preparation is well optimized."
        )

    elif estimated_extra <= 20:

        st.warning(
            "⚠️ Moderate food wastage may occur."
        )

    else:

        st.error(
            "❌ High potential food wastage."
        )

    # -----------------------------
    # Recommendation
    # -----------------------------
    st.subheader("💡 Canteen Recommendation")

    st.write(
        f"Prepare approximately **{recommended} meals** "
        f"for **{students} students** on {day_name}."
    )

# -----------------------------
# Sales Graph
# -----------------------------
st.subheader("📈 Daily Sales Analysis")

chart_data = df[["previous_orders", "meals_prepared"]]

st.line_chart(chart_data)

# -----------------------------
# Dataset
# -----------------------------
with st.expander("📋 View Training Dataset"):

    st.dataframe(
        df,
        use_container_width=True
    )

    