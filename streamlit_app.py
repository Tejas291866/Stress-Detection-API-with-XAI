# streamlit_app.py

import streamlit as st
import requests
import json # To pretty-print JSON

# --- Configuration (Make sure this matches your API's config) ---
FASTAPI_URL = "http://127.0.0.1:8000/predict_and_explain"
TOTAL_FLATTENED_FEATURES = 34 # 17 timesteps * 2 features_per_step

st.set_page_config(
    page_title="SWELL Stress Detection with XAI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 SWELL Stress Detection & Explanation")
st.markdown("---")

st.header("1. Enter HRV Feature Data")
st.info(f"Please provide {TOTAL_FLATTENED_FEATURES} HRV feature values, separated by commas. "
        "These should be raw, unscaled features from your SWELL dataset.")

# Text area for user input
features_input = st.text_area(
    "HRV Features (comma-separated):",
    height=150,
    placeholder="e.g., 0.959367, 0.912809, 0.215038, ..., 0.0, 0.0 (total 34 values)"
)

# Example input for easy testing
st.subheader("Quick Test Examples:")
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Sample Test Instance (Test.csv row 0)"):
        # This is the first row from your test.csv (features only)
        sample_features = [0.959367, 0.912809, 0.215038, 0.573273, 0.893758, 0.391924, 0.170624, 0.052601, 0.320492, 0.655979, 0.297491, 0.890632, 0.093489, 0.076891, 0.091177, 0.638531, 0.613941, 0.324391, 0.911075, 0.063854, 0.444697, 0.741088, 0.143714, 0.783637, 0.021074, 0.347714, 0.360144, 0.898867, 0.272138, 0.293573, 0.844337, 0.583952, 0.0, 0.0]
        features_input = ', '.join(map(str, sample_features)) # Update the text_area
        st.experimental_rerun() # Rerun to update the text area with the example

# Button to trigger prediction
if st.button("Predict and Explain Stress Level", type="primary"):
    if features_input:
        try:
            # Parse features from input string
            features_list = [float(f.strip()) for f in features_input.split(',') if f.strip()]

            if len(features_list) != TOTAL_FLATTENED_FEATURES:
                st.error(f"Error: Expected {TOTAL_FLATTENED_FEATURES} features, but received {len(features_list)}. "
                         "Please check your input.")
            else:
                st.spinner("Predicting and generating explanations...")
                # Make the POST request to your FastAPI endpoint
                response = requests.post(
                    FASTAPI_URL,
                    json={"features": features_list}
                )
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                result = response.json()

                st.markdown("---")
                st.header("2. Prediction Results")

                st.success(f"**Predicted Stress Class:** {result['predicted_class']}")

                st.subheader("Detailed Explanations (SHAP)")
                for exp in result['explanations']:
                    st.write(
                        f"- **{exp['feature']}**: Was **{exp['value_description']}** (value: {exp['value']:.4f}) "
                        f"and contributed **{exp['contribution_direction']}** to the '{result['predicted_class']}' prediction."
                    )
                
                st.subheader("Raw Explanation Data")
                st.json(result['explanations']) # Display raw JSON for debugging/inspection

        except ValueError:
            st.error("Invalid input. Please ensure all features are valid numbers.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Make sure your FastAPI app is running on "
                     f"`{FASTAPI_URL.replace('/predict_and_explain', '')}`.")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter HRV features or use a sample to get a prediction.")

st.markdown("---")
st.markdown("Powered by FastAPI and Streamlit.")