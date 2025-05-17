# MLP SHAP Dashboard - Technical Documentation

## Architecture Overview

The application is built on three main components:
1. Neural Network Visualization Engine
2. SHAP (SHapley Additive exPlanations) Implementation
3. Interactive Dashboard Interface

## 1. Neural Network Visualization Engine

### Core Implementation (`utils.py`)
```python
def visualize_neural_network(num_features, hidden_layers, num_classes, model=None, sample_input=None)
```

#### Technical Details:
- **Graph Generation**: Uses `graphviz` to create a directed acyclic graph (DAG)
- **Layout Algorithm**: Left-to-right hierarchical layout with three distinct node types
- **Color Mapping**: Implements linear color interpolation for weights:
  - Positive weights: Red spectrum (#FF0000 to #FFCCCC)
  - Negative weights: Blue spectrum (#0000FF to #CCCCFF)
- **Edge Thickness**: Dynamic scaling based on weight magnitude:
  ```python
  width = 0.1 + 2.0 * abs(weight) / max_weight
  ```

## 2. SHAP Implementation

### A. DeepExplainer
```python
class DeepExplainer(Explainer)
```

#### Algorithm Details:
1. **Background Integration**:
   - Complexity: O(N) where N = number of background samples
   - Variance reduction: σ ∝ 1/√N
   - Optimal sample size: 100-1000 samples

2. **Feature Attribution**:
   - Uses DeepLIFT algorithm with Shapley value approximation
   - Integrates over background distribution
   - Computes E[f(x)] - E[f(x')] for feature importance

### B. Gradient Explainer
```python
class GradientExplainer(Explainer)
```

#### Implementation Features:
1. **Expected Gradients Algorithm**:
   - Combines:
     * Integrated Gradients
     * SHAP values
     * SmoothGrad
   - Linear approximation between background samples
   - Independent feature assumption

2. **Optimization Techniques**:
   - Batch processing for matrix operations
   - GPU acceleration via CUDA kernels
   - Memory-efficient sparse computations

## 3. Visualization Components

### A. Summary Plot
```python
def plot_shap_summary(shap_values, feature_names)
```
- Violin plot implementation for distribution visualization
- Color mapping for positive/negative impacts
- Feature ranking by absolute SHAP values

### B. Waterfall Plot
```python
def plot_waterfall(shap_values, sample_idx)
```
- Cumulative effect visualization
- Base value reference
- Individual feature contributions

### C. Force Plot
```python
def plot_force_plot(shap_values, sample_idx)
```
- Interactive D3.js-based visualization
- Real-time feature inspection
- Dynamic force-directed layout

## Technical Optimizations

### 1. Memory Management
- Batch processing for large datasets
- Sparse matrix operations
- GPU memory optimization
- Memory-mapped file handling for large datasets

### 2. Computational Efficiency
```python
# Example of vectorized operations
samples_input = [torch.zeros((nsamples,) + X[t].shape[1:], device=X[t].device) for t in range(len(X))]
samples_delta = [np.zeros((nsamples,) + self.data[t].shape[1:]) for t in range(len(self.data))]
```

### 3. GPU Acceleration
```cpp
// CUDA kernel for SHAP value computation
gpu_treeshap::GPUTreeShap(X, paths.begin(), paths.end(), trees.num_outputs,
                          phis.begin(), phis.end());
```

## Session State Management

### Implementation:
```python
st.session_state['model']  # Model persistence
st.session_state['X']      # Data persistence
st.session_state['X_train']  # Training data
```

### Key Features:
1. Persistent storage across pages
2. Efficient data sharing
3. Memory-optimized state management

## Error Handling and Validation

### Implementation Strategy:
1. Input validation
2. Exception handling
3. User feedback
4. Graceful degradation

## Performance Considerations

### 1. Time Complexity
- Neural Network Visualization: O(n*m) where n=nodes, m=edges
- SHAP Computation: O(M*N) where M=features, N=background samples
- Visualization Rendering: O(k) where k=displayed elements

### 2. Space Complexity
- Model Storage: O(p) where p=parameters
- SHAP Values: O(s*f) where s=samples, f=features
- Visualization Data: O(v) where v=visible elements

## Future Optimizations

1. **Planned Improvements**:
   - Implement lazy loading for large models
   - Add distributed computation support
   - Enhance GPU utilization

2. **Scalability Considerations**:
   - Database integration for large datasets
   - Caching layer for repeated computations
   - Microservices architecture for horizontal scaling

## Technical Dependencies

### Core Libraries:
- SHAP: v0.47.2
- Streamlit: Latest
- PyTorch/TensorFlow: Latest stable
- Graphviz: Latest stable

### Hardware Requirements:
- CPU: Multi-core processor
- RAM: 8GB minimum
- GPU: Optional, CUDA-compatible for acceleration 