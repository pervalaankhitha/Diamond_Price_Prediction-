
import streamlit as st
import pandas as pd
import pickle
import os

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# --------------------------------
# Application Title
# --------------------------------

st.title("💎 Diamond Price Prediction")

st.write(
    "Predict diamond prices using the "
    "K-Nearest Neighbors (KNN) Regression algorithm."
)

st.divider()

# --------------------------------
# Load Trained Model
# --------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "diamond_price_model.pkl"
)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found! "
        "Please upload diamond_price_model.pkl "
        "to your project folder."
    )
    st.stop()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# --------------------------------
# User Input Section
# --------------------------------

st.subheader("Enter Diamond Details")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input(
        "Carat",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    cut = st.selectbox(
        "Cut",
        ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    )

    color = st.selectbox(
        "Color",
        ["D", "E", "F", "G", "H", "I", "J"]
    )

    clarity = st.selectbox(
        "Clarity",
        ["I1", "SI2", "SI1", "VS2", "VS1",
         "VVS2", "VVS1", "IF"]
    )

with col2:
    depth = st.number_input(
        "Depth",
        min_value=40.0,
        max_value=80.0,
        value=61.5
    )

    table = st.number_input(
        "Table",
        min_value=40.0,
        max_value=80.0,
        value=57.0
    )

    x = st.number_input(
        "Length (x)",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    y = st.number_input(
        "Width (y)",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )

    z = st.number_input(
        "Height (z)",
        min_value=0.0,
        max_value=10.0,
        value=3.0
    )

st.divider()

# --------------------------------
# Prediction
# --------------------------------

if st.button("💎 Predict Diamond Price", use_container_width=True):

    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    try:
        prediction = model.predict(input_data)

        st.success("Prediction Completed Successfully!")

        st.subheader("Estimated Diamond Price")

        st.metric(
            label="Predicted Price",
            value=f"${prediction[0]:,.2f}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# --------------------------------
# Footer
# --------------------------------

st.divider()

st.caption(
    "Developed using Python, Scikit-learn, "
    "KNN Regression and Streamlit."
)
