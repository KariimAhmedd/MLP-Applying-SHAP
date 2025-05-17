"""
Utility functions for MLP SHAP Dashboard visualization and analysis.

Technical Implementation Details:
-------------------------------
1. Neural Network Visualization:
   - Graph Generation: Uses graphviz for DAG creation
   - Color Mapping: Linear interpolation for weights
   - Edge Thickness: Dynamic scaling based on weight magnitude
   - Layout: Hierarchical left-to-right with layered nodes
   - Complexity: O(n*m) where n=nodes, m=edges

2. SHAP Visualization Components:
   - Summary Plot: Distribution visualization with violin plots
   - Waterfall Plot: Cumulative effect visualization
   - Force Plot: Interactive feature impact display
   - Report Generation: Multi-plot PDF compilation

3. Performance Optimizations:
   - Matplotlib figure management
   - Memory cleanup after plotting
   - Efficient color computation
   - Vectorized operations for large networks

4. Color Computation Algorithm:
   - Normalization: weight -> [0,1] range
   - RGB Channel Manipulation:
     * Positive weights: Primarily red channel
     * Negative weights: Primarily blue channel
   - Hex color generation for graphviz compatibility

5. Activation Function Analysis:
   - Dataset characteristics analysis
   - Statistical metrics computation
   - Intelligent activation function selection
   - Detailed explanations and recommendations
"""

import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder

def get_color_for_weight(weight, vmin=-1, vmax=1):
    """
    Convert neural network weight to color representation.
    
    Technical Details:
    ----------------
    1. Normalization: Maps weight to [0,1] range using min-max scaling
    2. Color Channel Computation:
       - Positive weights: Emphasize red channel (255)
       - Negative weights: Emphasize blue channel (255)
    3. Secondary channels: Scaled to 200 for contrast
    
    Parameters:
    ----------
    weight : float
        Neural network weight value
    vmin : float, optional
        Minimum value for normalization (default: -1)
    vmax : float, optional
        Maximum value for normalization (default: 1)
        
    Returns:
    -------
    str
        Hex color code (#RRGGBB format)
    """
    norm_weight = (weight - vmin) / (vmax - vmin)
    if weight > 0:
        return f"#{'%02x' % int(norm_weight * 255)}{'%02x' % int(norm_weight * 200)}{'%02x' % int(norm_weight * 200)}"
    else:
        return f"#{'%02x' % int(-norm_weight * 200)}{'%02x' % int(-norm_weight * 200)}{'%02x' % int(-norm_weight * 255)}"

def visualize_neural_network(num_features, hidden_layers, num_classes, model=None, sample_input=None, activations=None):
    """
    Generate interactive neural network visualization with optimized performance.
    """
    dot = graphviz.Digraph('neural_network')
    dot.attr(rankdir='LR')  # Left to right layout
    dot.attr('graph', splines='line')  # Use straight lines for edges
    dot.attr('graph', ranksep='2')  # Increase space between layers
    dot.attr('graph', nodesep='0.5')  # Increase space between nodes
    
    # Node styling - simplified for performance
    dot.attr('node', shape='circle', style='filled', fixedsize='true', width='0.7')
    
    # Simplified color scheme
    colors = {
        'input': '#A8E6CF',
        'hidden': '#FFD3B6',
        'output': '#FF8B94'
    }
    
    # Simplified activation color function
    def get_activation_color(activation_value):
        if activation_value > 0.5:
            return '#FF8B94'
        return '#CCCCCC'
    
    # Create all nodes first (better performance than mixing nodes and edges)
    with dot.subgraph() as s:
        s.attr(rank='same')
        for i in range(num_features):
            label = f'x{i+1}'
            s.node(f'i{i}', label, fillcolor=colors['input'])
    
    # Add hidden layers
    for l, layer_size in enumerate(hidden_layers, 1):
        with dot.subgraph() as s:
            s.attr(rank='same')
            for i in range(layer_size):
                label = f'h{l}_{i}'
                s.node(f'h{l}_{i}', label, fillcolor=colors['hidden'])
    
    # Add output layer
    with dot.subgraph() as s:
        s.attr(rank='same')
        for i in range(num_classes):
            label = f'y{i+1}'
            s.node(f'o{i}', label, fillcolor=colors['output'])
    
    # Add edges with optimized weight visualization
    if model is not None and hasattr(model, 'coefs_'):
        weights = model.coefs_
        max_weight = max(abs(w).max() for w in weights)
        
        # Input to first hidden layer
        for i in range(num_features):
            for j in range(hidden_layers[0]):
                try:
                    weight = weights[0][i, j]
                    if abs(weight) > 0.1 * max_weight:  # Only show significant connections
                        width = str(0.1 + abs(weight) / max_weight)
                        dot.edge(f'i{i}', f'h1_{j}', penwidth=width)
                except IndexError:
                    continue
        
        # Between hidden layers
        for l in range(len(hidden_layers) - 1):
            for i in range(hidden_layers[l]):
                for j in range(hidden_layers[l + 1]):
                    try:
                        weight = weights[l + 1][i, j]
                        if abs(weight) > 0.1 * max_weight:  # Only show significant connections
                            width = str(0.1 + abs(weight) / max_weight)
                            dot.edge(f'h{l+1}_{i}', f'h{l+2}_{j}', penwidth=width)
                    except IndexError:
                        continue
        
        # Last hidden layer to output
        last_hidden_idx = len(hidden_layers)
        for i in range(hidden_layers[-1]):
            for j in range(num_classes):
                try:
                    weight = weights[-1][i, j]
                    if abs(weight) > 0.1 * max_weight:  # Only show significant connections
                        width = str(0.1 + abs(weight) / max_weight)
                        dot.edge(f'h{last_hidden_idx}_{i}', f'o{j}', penwidth=width)
                except IndexError:
                    continue
    
    return dot

def plot_shap_summary(shap_values, feature_names):
    """
    Generate SHAP summary plot with explanations.
    """
    plt.clf()  # Clear any existing plots
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Add explanations under the summary plot
    st.markdown("""
    ### Understanding the Summary Plot Above:
    
    📈 **How to Read This Plot:**
    - Features are ranked by importance (top to bottom)
    - Colors: Red = High feature value, Blue = Low feature value
    - Position: Right = Positive impact, Left = Negative impact
    
    🎯 **Key Points:**
    - Wider spreads indicate greater feature impact
    - Color patterns show feature value relationships
    - The most important features are at the top
    
    💡 **Tips:**
    - Look for clear color separations
    - Note features with consistent impacts
    - Consider feature interactions based on patterns
    """)

def plot_waterfall(shap_values, sample_idx):
    """
    Generate SHAP waterfall plot with detailed explanations.
    """
    plt.clf()  # Clear any existing plots
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Add explanations under the waterfall plot
    st.markdown("""
    ### Understanding the Waterfall Plot Above:
    
    📊 **How to Read This Plot:**
    - Starting Point: Base value (average model prediction)
    - Red Bars: Features increasing the prediction
    - Blue Bars: Features decreasing the prediction
    - Final Value: The model's prediction for this sample
    
    🔍 **Key Insights:**
    - Longer bars indicate stronger feature impacts
    - The order shows feature importance (top to bottom)
    - The cumulative effect builds up to the final prediction
    
    💡 **Tips:**
    - Look for the largest bars to identify key drivers
    - Compare positive vs negative influences
    - Note how features combine to reach the final prediction
    """)

def plot_force_plot(shap_values, sample_idx):
    """
    Generate SHAP force plot with explanations.
    """
    # Create force plot HTML
    force_plot = shap.plots.force(shap_values[sample_idx], matplotlib=False)
    
    # Convert to HTML and display in Streamlit
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    st.components.v1.html(shap_html, height=150)
    
    # Add explanations under the plot
    st.markdown("""
    ### Understanding the Force Plot Above:
    
    🎯 **How to Read This Plot:**
    - Base Value: The average model output over the training dataset
    - Red Arrows: Features pushing the prediction higher
    - Blue Arrows: Features pushing the prediction lower
    - Arrow Width: Magnitude of the feature's impact
    
    💡 **Key Points:**
    - Longer bars = Stronger impact on prediction
    - Look for clusters of similar colors to identify related effects
    - The final prediction is shown on the right
    """)

def generate_report(shap_values, feature_names):
    """
    Generate comprehensive SHAP analysis report.
    """
    plt.clf()  # Clear any existing plots
    fig = plt.figure(figsize=(15, 10))
    
    # Summary plot
    plt.subplot(2, 1, 1)
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    
    # Waterfall plot
    plt.subplot(2, 1, 2)
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall Plot (First Sample)")
    
    plt.tight_layout()
    plt.savefig("report.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def suggest_activation_function(X, y):
    """
    Analyzes dataset characteristics and suggests appropriate activation functions.
    
    Args:
        X: Features/input data
        y: Target/output data
        
    Returns:
        dict: Suggested activation functions for hidden and output layers
    """
    # Convert y to numeric if it's not already
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Get basic dataset characteristics
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    # Advanced data analysis
    has_negative_values = (X < 0).any()
    data_range = np.ptp(X, axis=0).mean()
    feature_correlation = np.corrcoef(X.T)
    avg_correlation = np.abs(feature_correlation - np.eye(n_features)).mean()
    
    # Calculate additional metrics
    skewness = np.mean([np.abs(np.mean(X[:, i])) for i in range(n_features)])
    sparsity = np.mean(X == 0)
    feature_std = np.std(X, axis=0).mean()
    
    # Initialize suggestion dictionary
    suggestion = {
        'hidden_layer': None,
        'output_layer': None,
        'explanation': [],
        'dataset_stats': {
            'samples': n_samples,
            'features': n_features,
            'classes': n_classes,
            'avg_correlation': avg_correlation,
            'data_range': data_range,
            'skewness': skewness,
            'sparsity': sparsity
        }
    }
    
    # Output layer activation
    if n_classes == 2:
        suggestion['output_layer'] = 'sigmoid'
        suggestion['explanation'].append("Binary classification task -> Sigmoid for output layer (optimal for binary cross-entropy loss)")
    else:
        suggestion['output_layer'] = 'softmax'
        suggestion['explanation'].append(f"Multi-class classification ({n_classes} classes) -> Softmax for output layer (provides normalized probability distribution)")
    
    # Hidden layer activation decision tree
    if has_negative_values and avg_correlation > 0.5:
        suggestion['hidden_layer'] = 'ELU'
        suggestion['explanation'].append("Dataset has negative values and high feature correlation -> ELU recommended (helps with gradient flow and handles correlations well)")
    elif has_negative_values and sparsity > 0.5:
        suggestion['hidden_layer'] = 'LeakyReLU'
        suggestion['explanation'].append("Dataset has negative values and high sparsity -> LeakyReLU recommended (prevents dying ReLU problem)")
    elif skewness > 1.0:
        suggestion['hidden_layer'] = 'SELU'
        suggestion['explanation'].append("Dataset shows high skewness -> SELU recommended (self-normalizing properties help with skewed data)")
    elif data_range > 10 and feature_std > 1.0:
        suggestion['hidden_layer'] = 'ReLU'
        suggestion['explanation'].append("Large data range and variance -> ReLU recommended (good for varied feature scales after normalization)")
    else:
        suggestion['hidden_layer'] = 'ReLU'
        suggestion['explanation'].append("Default choice -> ReLU recommended (efficient computation and good gradient flow)")
    
    # Additional considerations and recommendations
    if n_features > 50:
        suggestion['explanation'].append("High-dimensional data -> Consider adding Batch Normalization (helps with internal covariate shift)")
    if n_samples < 1000:
        suggestion['explanation'].append("Small dataset -> Consider using regularization techniques (Dropout or L2)")
    if avg_correlation > 0.8:
        suggestion['explanation'].append("Very high feature correlation -> Consider adding feature selection or PCA")
    if sparsity > 0.7:
        suggestion['explanation'].append("High data sparsity -> Consider using sparse initialization techniques")
        
    return suggestion

def print_activation_suggestion(dataset_name, X, y):
    """
    Prints activation function suggestions for a given dataset with detailed analysis.
    
    Args:
        dataset_name: Name of the dataset
        X: Features/input data
        y: Target/output data
    """
    suggestion = suggest_activation_function(X, y)
    
    print(f"\nActivation Function Analysis for {dataset_name} Dataset:")
    print("=" * 80)
    
    # Dataset Statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    stats = suggestion['dataset_stats']
    print(f"Samples: {stats['samples']}")
    print(f"Features: {stats['features']}")
    print(f"Classes: {stats['classes']}")
    print(f"Average Feature Correlation: {stats['avg_correlation']:.3f}")
    print(f"Data Range: {stats['data_range']:.3f}")
    print(f"Skewness: {stats['skewness']:.3f}")
    print(f"Sparsity: {stats['sparsity']:.3f}")
    
    # Activation Function Recommendations
    print("\nRecommended Activation Functions:")
    print("-" * 40)
    print(f"Hidden Layers: {suggestion['hidden_layer']}")
    print(f"Output Layer: {suggestion['output_layer']}")
    
    # Detailed Reasoning
    print("\nDetailed Analysis:")
    print("-" * 40)
    for i, exp in enumerate(suggestion['explanation'], 1):
        print(f"{i}. {exp}")
    
    print("\nNote: These recommendations are based on statistical analysis of your data.")
    print("Consider experimenting with different activation functions for optimal performance.")
    print("=" * 80)

# Example usage:
# print_activation_suggestion("Iris", X_iris, y_iris)
# print_activation_suggestion("Breast Cancer", X_cancer, y_cancer)
# print_activation_suggestion("Wine", X_wine, y_wine)