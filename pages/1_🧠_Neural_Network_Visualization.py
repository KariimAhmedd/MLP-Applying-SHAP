"""
Neural Network Visualization Page for MLP SHAP Dashboard.

Technical Implementation Details:
-------------------------------
1. Visualization Architecture:
   - Interactive graph rendering using graphviz
   - Dynamic node and edge styling
   - Real-time weight visualization
   - Session state integration

2. Performance Considerations:
   - Lazy loading of model weights
   - Efficient graph generation
   - Memory-optimized visualization
   - Browser-friendly SVG output

3. State Management:
   - Model persistence across pages
   - Data synchronization
   - Configuration preservation
   - Error handling for missing state

4. User Interface Components:
   - Interactive node inspection
   - Dynamic layout updates
   - Responsive design elements
   - Informative tooltips
"""

import streamlit as st
from utils import visualize_neural_network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Neural Network Visualization", layout="wide")

st.title("🧠 Neural Network Visualization")

# Check if model exists in session state
if 'model' not in st.session_state:
    st.error("Please configure and train your model in the main page first!")
    st.stop()

# Get model and data from session state
model = st.session_state['model']
X = st.session_state['X']
X_test = st.session_state['X_test']
num_features = st.session_state['num_features']
hidden_layers = st.session_state['hidden_layers']
num_classes = st.session_state['num_classes']

# Add comprehensive architecture explanation
st.markdown("""
## 🎯 Understanding Neural Network Parameters

### 1. Number of Hidden Layers (1-3)
The number of hidden layers affects the network's ability to learn complex patterns:

🔹 **Single Layer (1)**
- Best for: Simple, linear-like relationships
- Faster training and less prone to overfitting
- Limited ability to learn complex patterns
- Example use: Basic classification tasks

🔹 **Multiple Layers (2-3)**
- Better for: Complex, non-linear relationships
- Can learn hierarchical features
- More prone to overfitting if not properly regularized
- Example use: Complex pattern recognition

### 2. Nodes per Layer (5-200)

🔸 **Few Nodes (5-50)**
- Advantages:
  * Faster training
  * Less memory usage
  * Less prone to overfitting
- Disadvantages:
  * May underfit if too few
  * Limited learning capacity

🔸 **Many Nodes (100-200)**
- Advantages:
  * Can learn more complex patterns
  * Better feature representation
- Disadvantages:
  * Slower training
  * More prone to overfitting
  * Higher memory usage

### 3. Activation Function

📊 **ReLU (Rectified Linear Unit)**
- Most commonly used
- Benefits:
  * Reduces vanishing gradient problem
  * Faster training
  * Sparse activation
- Best for: Deep networks, general purpose

📊 **Tanh**
- Benefits:
  * Zero-centered outputs
  * Good for bounded predictions
- Best for: When data is centered around zero

📊 **Logistic (Sigmoid)**
- Benefits:
  * Outputs between 0 and 1
  * Good for probability prediction
- Best for: Binary classification, final layer

### 4. Learning Rate (0.0001-0.1)

📈 **Small Learning Rate (0.0001-0.001)**
- Advantages:
  * More stable training
  * Better final accuracy
- Disadvantages:
  * Slower convergence
  * May get stuck in local minima

📈 **Large Learning Rate (0.01-0.1)**
- Advantages:
  * Faster initial learning
  * Can escape local minima
- Disadvantages:
  * May overshoot optimal weights
  * Training might be unstable

### 5. Max Iterations (100-1000)

⏱️ **Fewer Iterations (100-300)**
- Good for:
  * Quick prototyping
  * Simple problems
  * When data is well-behaved
- Risk: Might not reach optimal solution

⏱️ **More Iterations (500-1000)**
- Good for:
  * Complex problems
  * When accuracy is crucial
  * When learning rate is small
- Risk: Might overfit if not properly regularized

💡 **Best Practices**:
1. Start simple (1 layer, moderate nodes)
2. Monitor training/validation performance
3. Increase complexity only if needed
4. Use early stopping to prevent overfitting
""")

# Create tabs for different visualization modes
viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Network Architecture", "Live Predictions", "Weight Analysis"])

with viz_tab1:
    st.markdown("""
    ### Network Architecture Visualization
    This visualization shows the complete neural network structure with:
    - Input layer (green nodes)
    - Hidden layers (orange nodes)
    - Output layer (red nodes)
    - Weight connections (colored by sign and thickness)
    """)
    
    # Display the neural network visualization
    try:
        graph = visualize_neural_network(
            num_features=num_features,
            hidden_layers=hidden_layers,
            num_classes=num_classes,
            model=model
        )
        st.graphviz_chart(graph, use_container_width=True)
        
        # Add architecture details
        st.markdown(f"""
        ### Architecture Details:
        - Input Features: {num_features}
        - Hidden Layers: {' → '.join(map(str, hidden_layers))}
        - Output Classes: {num_classes}
        - Activation Function: {model.activation}
        - Total Parameters: {sum(w.size for w in model.coefs_)}
        """)
    except Exception as e:
        st.error(f"Error in visualization: {str(e)}")
        st.info("Try adjusting the network architecture or refreshing the page.")

with viz_tab2:
    st.markdown("### Live Prediction Visualization")
    
    # Sample selection
    sample_idx = st.slider("Select sample to visualize", 0, len(X_test)-1, 0)
    sample_input = X_test[sample_idx]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show network with activations
        st.markdown("#### Network Flow")
        activations = []
        current_activation = sample_input.reshape(1, -1)
        
        # Calculate activations through the network
        for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
            current_activation = np.dot(current_activation, coef) + intercept
            
            # Apply activation function
            if model.activation == 'relu':
                current_activation = np.maximum(0, current_activation)
            elif model.activation == 'tanh':
                current_activation = np.tanh(current_activation)
            elif model.activation == 'logistic':
                current_activation = 1/(1 + np.exp(-current_activation))
            
            activations.append(current_activation)
        
        # Visualize network with activations
        graph_with_activations = visualize_neural_network(
            num_features=num_features,
            hidden_layers=hidden_layers,
            num_classes=num_classes,
            model=model,
            sample_input=sample_input,
            activations=activations
        )
        st.graphviz_chart(graph_with_activations, use_container_width=True)
    
    with col2:
        # Show feature values
        st.markdown("#### Input Features")
        feature_df = pd.DataFrame({
            'Feature': X.columns,
            'Value': sample_input
        })
        st.dataframe(feature_df)
        
        # Show prediction
        st.markdown("#### Prediction")
        pred_proba = model.predict_proba(sample_input.reshape(1, -1))[0]
        for i, prob in enumerate(pred_proba):
            st.progress(prob)
            st.write(f"Class {i}: {prob:.4f}")

with viz_tab3:
    st.markdown("### Weight Analysis")
    
    # Get weights from each layer
    weights = model.coefs_
    
    # Create weight distribution plots
    for i, layer_weights in enumerate(weights):
        st.markdown(f"#### Layer {i+1} Weights")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Weight distribution
        ax1.hist(layer_weights.flatten(), bins=50)
        ax1.set_title('Weight Distribution')
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Count')
        
        # Weight heatmap
        im = ax2.imshow(layer_weights, cmap='RdBu', aspect='auto')
        ax2.set_title('Weight Heatmap')
        ax2.set_xlabel('To Node')
        ax2.set_ylabel('From Node')
        plt.colorbar(im, ax=ax2)
        
        st.pyplot(fig)
        plt.close()
        
        # Weight statistics
        st.markdown(f"""
        **Layer Statistics:**
        - Mean: {layer_weights.mean():.4f}
        - Std: {layer_weights.std():.4f}
        - Min: {layer_weights.min():.4f}
        - Max: {layer_weights.max():.4f}
        """)
        
        # Add separator
        st.markdown("---")

st.markdown("""
---
### 🎯 How to Use This Visualization

1. **Start with the Network Architecture View**:
   - Understand the overall structure
   - Note the number of layers and nodes
   - Observe the general connection patterns

2. **Explore Sample Predictions**:
   - Switch to the Sample Prediction Flow tab
   - Try different samples using the slider
   - Watch how input values affect the network
   - Compare activation patterns between samples

3. **Look for Patterns**:
   - Strong connections (thick lines)
   - Consistent activation patterns
   - Feature importance (connection density)

4. **Use the Insights**:
   - Identify important features
   - Understand model behavior
   - Debug potential issues
""") 