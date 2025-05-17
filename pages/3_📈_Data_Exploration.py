"""
Data Exploration Page for MLP SHAP Dashboard.

This page provides comprehensive data exploration and visualization tools
to help users understand the characteristics of the selected dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Data Exploration", layout="wide")

# Title and description
st.title("📈 Data Exploration")
st.markdown("""
This page provides detailed insights into the selected dataset's characteristics,
distributions, and relationships between features.
""")

# Get data from session state
if 'data' not in st.session_state or 'X' not in st.session_state or 'y' not in st.session_state:
    st.error("Please configure the model in the main page first!")
    st.stop()

data = st.session_state['data']
X = st.session_state['X']
y = st.session_state['y']

# Create DataFrame with features and target
if isinstance(X, pd.DataFrame):
    df = X.copy()  # For custom datasets that are already DataFrames
    df['target'] = y
    if hasattr(data, 'target_names') and len(data.target_names) == len(np.unique(y)):
        df['target_name'] = pd.Categorical([data.target_names[i] for i in y])
    else:
        df['target_name'] = pd.Categorical([f"Class {i}" for i in y])
else:
    df = pd.DataFrame(X, columns=data.feature_names)
    df['target'] = y
    df['target_name'] = pd.Categorical(data.target_names[y])

# Sidebar controls
st.sidebar.header("Visualization Options")
plot_type = st.sidebar.selectbox(
    "Select Plot Type",
    ["Dataset Overview", "Feature Distributions", "Correlation Analysis", "Feature Relationships"]
)

# 1. Dataset Overview
if plot_type == "Dataset Overview":
    st.header("Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Samples", X.shape[0])
    with col2:
        st.metric("Number of Features", X.shape[1])
    with col3:
        st.metric("Number of Classes", len(np.unique(y)))
    
    # Basic statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    
    # Class distribution
    st.subheader("Class Distribution")
    fig = px.pie(df, names='target_name', title='Distribution of Target Classes')
    st.plotly_chart(fig)
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

# 2. Feature Distributions
elif plot_type == "Feature Distributions":
    st.header("Feature Distributions")
    
    # Feature selection
    selected_features = st.multiselect(
        "Select features to visualize",
        options=data.feature_names,
        default=data.feature_names[:3]
    )
    
    if selected_features:
        # Distribution plots
        for feature in selected_features:
            st.subheader(f"Distribution of {feature}")
            fig = px.histogram(
                df, x=feature, color='target_name',
                marginal="box",
                title=f"Distribution of {feature} by Class"
            )
            st.plotly_chart(fig)
            
            # Basic statistics for the feature
            st.markdown("**Feature Statistics:**")
            stats = df.groupby('target_name')[feature].describe()
            st.dataframe(stats)

# 3. Correlation Analysis
elif plot_type == "Correlation Analysis":
    st.header("Correlation Analysis")
    
    # Correlation matrix
    corr_matrix = df.drop(['target', 'target_name'], axis=1).corr()
    
    # Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig)
    
    # Top correlations
    st.subheader("Top Feature Correlations")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    st.dataframe(corr_df.head(10))

# 4. Feature Relationships
elif plot_type == "Feature Relationships":
    st.header("Feature Relationships")
    
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X-axis feature", data.feature_names)
    with col2:
        y_feature = st.selectbox("Select Y-axis feature", 
                                [f for f in data.feature_names if f != x_feature],
                                index=1 if len(data.feature_names) > 1 else 0)
    
    # Scatter plot
    fig = px.scatter(
        df, x=x_feature, y=y_feature,
        color='target_name',
        title=f"Relationship between {x_feature} and {y_feature}",
        trendline="ols"
    )
    st.plotly_chart(fig)
    
    # Additional insights
    st.subheader("Feature Pair Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Correlation:**")
        correlation = df[x_feature].corr(df[y_feature])
        st.metric("Pearson Correlation", f"{correlation:.3f}")
    with col2:
        st.markdown("**Class Separation:**")
        f_stat = df.groupby('target_name')[[x_feature, y_feature]].mean().std().mean()
        st.metric("Class Separation Score", f"{f_stat:.3f}") 