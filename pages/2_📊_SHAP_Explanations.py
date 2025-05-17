"""
SHAP Explanations Page for MLP SHAP Dashboard.

Technical Implementation Details:
-------------------------------
1. SHAP Integration:
   - Deep SHAP algorithm implementation
   - Background dataset integration
   - Feature attribution computation
   - Multi-class support

2. Visualization Components:
   - Summary Plot: Feature importance distribution
   - Waterfall Plot: Individual prediction explanation
   - Force Plot: Interactive feature impact
   - Report Generation: Multi-plot compilation

3. Performance Optimizations:
   - Batch processing for SHAP computation
   - Memory-efficient plotting
   - Background sample optimization
   - GPU acceleration support

4. Data Management:
   - Session state synchronization
   - Efficient data transfer
   - Memory cleanup
   - Cache optimization

5. Technical Dependencies:
   - SHAP: Feature attribution
   - Matplotlib: Plotting backend
   - Streamlit: UI components
   - NumPy: Numerical operations
"""

import streamlit as st
import streamlit.components.v1 as components
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils import plot_shap_summary, plot_force_plot, plot_waterfall, generate_report

st.set_page_config(page_title="SHAP Explanations", layout="wide")

st.title("📊 SHAP Explanations")

# Check if model exists in session state
if 'model' not in st.session_state:
    st.error("Please configure and train your model in the main page first!")
    st.stop()

# Get model and data from session state
model = st.session_state['model']
X = st.session_state['X']
X_train = st.session_state['X_train']
X_test = st.session_state['X_test']
data = st.session_state['data']

# Calculate SHAP values
with st.spinner("Calculating SHAP values... This may take a moment."):
    explainer = shap.Explainer(model.predict_proba, X_train)
    shap_values_all = explainer(X_test[:50])

# Handle multiclass
if len(shap_values_all.shape) == 3:  # shape = (samples, features, classes)
    class_names = data.target_names.tolist()
    selected_class_index = st.selectbox(
        "Select Class for SHAP Analysis", 
        list(enumerate(class_names)), 
        format_func=lambda x: x[1]
    )[0]
    
    shap_values = shap.Explanation(
        values=shap_values_all.values[:, :, selected_class_index],
        base_values=shap_values_all.base_values[:, selected_class_index],
        data=shap_values_all.data,
        feature_names=X.columns
    )
else:
    shap_values = shap_values_all

# Create tabs for different SHAP visualizations
plot_tab1, plot_tab2, plot_tab3, plot_tab4 = st.tabs([
    "Summary Plot", 
    "Waterfall Plot",
    "Force Plot",
    "Feature Importance"
])

with plot_tab1:
    st.markdown("""
    ### SHAP Summary Plot
    
    This plot shows the overall impact of each feature on model predictions:
    
    **How to Read:**
    - Features are ranked by importance (top to bottom)
    - Red points indicate high feature values
    - Blue points indicate low feature values
    - Position on x-axis shows impact on prediction:
        - Right → Increases prediction
        - Left → Decreases prediction
    """)
    plot_shap_summary(shap_values, X.columns)

with plot_tab2:
    st.markdown("""
    ### SHAP Waterfall Plot
    
    Shows how each feature contributes to a specific prediction:
    
    **How to Read:**
    - Starts from base value (dataset average)
    - Each bar shows one feature's contribution
    - Final value shows the prediction
    - Red = positive impact, Blue = negative impact
    """)
    sample_for_waterfall = st.slider(
        "Select sample for Waterfall plot", 
        0, len(X_test[:50])-1, 0,
        help="Choose different samples to see how feature contributions vary"
    )
    plot_waterfall(shap_values, sample_for_waterfall)

with plot_tab3:
    st.markdown("""
    ### SHAP Force Plot
    
    This interactive visualization shows how each feature contributes to pushing the prediction away from the base value:
    
    **Key Features:**
    - Interactive tooltips on hover
    - Dynamic resizing
    - Real-time feature contribution analysis
    """)
    sample_for_force = st.slider(
        "Select sample for Force plot", 
        0, len(X_test[:50])-1, 0,
        help="Choose different samples to see varying feature impacts"
    )
    plot_force_plot(shap_values, sample_for_force)

with plot_tab4:
    st.markdown("""
    ### Feature Importance Table
    
    Quantitative ranking of feature impacts:
    
    **How to Read:**
    - Higher absolute SHAP values = More important features
    - Shows average magnitude of impact across all predictions
    - Use this to identify key features for your model
    """)
    feature_means = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Mean |SHAP Value|": feature_means
    }).sort_values(by="Mean |SHAP Value|", ascending=False)
    st.dataframe(importance_df)

# Report Generation
st.markdown("---")
st.subheader("📥 Generate Report")
st.markdown("""
Generate a comprehensive SHAP analysis report containing:
- Summary Plot
- Waterfall Plot
- Feature Importance Rankings
""")

if st.button("Generate SHAP Report"):
    with st.spinner("Generating report..."):
        generate_report(shap_values, X.columns)
        with open("report.png", "rb") as f:
            st.download_button(
                "Download SHAP Report",
                f,
                file_name="shap_report.png",
                help="Download a PNG file containing all SHAP visualizations"
            ) 