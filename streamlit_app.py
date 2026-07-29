"""
Car Price Estimator - Streamlit web app
Machine Learning for Developers (CAI2C08)

Loads the trained Random Forest pipeline and predicts a car's MSRP from its
specifications. The user sets a few key specs; any unspecified detail falls back
to a typical (median/mode) value, and the pipeline handles imputing/encoding.

Run locally:   streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Estimator", page_icon="🚗", layout="centered")


@st.cache_resource
def load_artifacts():
    """Load the model + metadata once and cache them across reruns."""
    model = joblib.load("car_price_model.pkl")
    meta = joblib.load("feature_meta.pkl")
    return model, meta


try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find the model files. Make sure **car_price_model.pkl** and "
        "**feature_meta.pkl** are in the same folder as this app."
    )
    st.stop()

defaults = meta["defaults"]
categories = meta["categories"]


def cat_options(col, fallback):
    """Dropdown options for a categorical column, with a safe fallback."""
    return categories.get(col, fallback)


def safe_int(value, allowed, fallback):
    """Return int(value) if it is in `allowed`, else `fallback` (for select_slider)."""
    v = int(round(float(value)))
    return v if v in allowed else fallback


# ----------------------------- Header -----------------------------
st.title("🚗 Car Price Estimator")
st.caption(
    "Estimate a car's price (MSRP) from its specifications. Powered by a Random "
    "Forest trained on 17,000+ real vehicles — average error ≈ $2,900."
)

st.subheader("Enter the car's specifications")

col1, col2 = st.columns(2)

with col1:
    hp = st.slider("Horsepower (hp)", 60, 810,
                   int(defaults["Engine Horsepower Hp"]), step=5)
    torque = st.slider("Torque (ft-lbs)", 60, 1060,
                       int(defaults["Engine Torque Ft Lbs"]), step=5)
    engine_size = st.slider("Engine size (litres)", 1.0, 8.5,
                            round(float(defaults["Engine Size"]), 1), step=0.1)
    cyl_options = [2, 3, 4, 5, 6, 8, 10, 12, 16]
    cylinders = st.select_slider(
        "Cylinders", options=cyl_options,
        value=safe_int(defaults["Cylinder Count"], cyl_options, 4),
    )

with col2:
    body_type = st.selectbox("Body type", cat_options("Body Type", ["Sedan", "SUV"]))
    drive = st.selectbox("Drive type",
                         cat_options("Engine Drive Type", ["front wheel drive"]))
    fuel = st.selectbox("Fuel type",
                        cat_options("Engine Fuel Type", ["regular unleaded"]))
    seats = st.slider("Seats", 1, 15, int(defaults["Body Seats"]))

st.divider()

# --------------------------- Prediction ---------------------------
if st.button("Estimate Price", type="primary", use_container_width=True):
    # Basic input validation with a user-facing message
    if hp < 40 or torque < 40:
        st.warning("Please enter realistic horsepower and torque values (≥ 40).")
    else:
        # Start from typical values, then override with the user's inputs
        row = dict(defaults)
        row.update({
            "Engine Horsepower Hp": hp,
            "Engine Torque Ft Lbs": torque,
            "Engine Size": engine_size,
            "Cylinder Count": cylinders,
            "Body Type": body_type,
            "Engine Drive Type": drive,
            "Engine Fuel Type": fuel,
            "Body Seats": seats,
        })

        X_new = pd.DataFrame([row])[meta["columns"]]

        try:
            price = float(model.predict(X_new)[0])
            st.metric("Estimated price", f"${price:,.0f}")
            st.caption(
                "Typical values are assumed for any specification not shown above."
            )
        except Exception as e:  # user-facing error, don't crash the app
            st.error(f"Prediction failed: {e}")
else:
    st.info("Adjust the specifications above, then click **Estimate Price**.")

# ----------------------------- Footer -----------------------------
st.divider()
st.caption("Car price prediction demo")
