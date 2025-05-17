import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder

def suggest_activation_function(X, y):
    """
    Analyzes dataset characteristics and suggests appropriate activation functions.
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

# Load and analyze each dataset
datasets = [
    (load_iris(), "Iris"),
    (load_breast_cancer(), "Breast Cancer"),
    (load_wine(), "Wine")
]

for data, name in datasets:
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.data)
    
    # Get activation function suggestions
    print_activation_suggestion(name, X_scaled, data.target) 