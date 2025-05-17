# MLP SHAP Dashboard 🧠

An interactive dashboard for visualizing and understanding Multi-Layer Perceptron (MLP) neural networks using SHAP (SHapley Additive exPlanations) values.

## Features 🌟

- **Interactive Neural Network Visualization**: See your model's architecture in real-time
- **SHAP Explanations**: Understand model predictions with detailed interpretability
- **Multiple Dataset Support**:
  - Built-in datasets (Breast Cancer, Iris, Wine)
  - Fake News Detection
  - Custom dataset upload
- **GPU Acceleration**: Supports Apple M-series GPUs (MPS) and NVIDIA GPUs (CUDA)
- **Real-time Training Monitoring**: Watch your model learn with progress tracking
- **Advanced Model Architecture**:
  - Configurable hidden layers
  - Batch normalization
  - Dropout regularization
  - Learning rate scheduling

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/KariimAhmedd/MLP-Applying-SHAP.git
cd MLP-Applying-SHAP
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 💡

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Choose your dataset:
   - Select from built-in datasets
   - Upload your own dataset
   - Use the fake news detection feature

4. Configure your model:
   - Adjust number of hidden layers
   - Set nodes per layer
   - Choose activation functions
   - Tune learning parameters

5. Train and analyze:
   - Watch real-time training progress
   - Explore SHAP explanations
   - Visualize network architecture
   - Examine prediction probabilities

## Requirements 📋

- Python 3.8+
- PyTorch
- Streamlit
- scikit-learn
- SHAP
- pandas
- numpy
- matplotlib
- seaborn

## Technical Details 🔧

### Model Architecture

- Multi-Layer Perceptron (MLP) with configurable architecture
- Batch Normalization for stable training
- Dropout for regularization
- AdamW optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting

### Hardware Acceleration

- Automatic device detection
- Support for:
  - Apple M-series GPUs (MPS)
  - NVIDIA GPUs (CUDA)
  - CPU fallback

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- SHAP library for model interpretability
- Streamlit for the interactive web interface
- PyTorch for deep learning capabilities
- scikit-learn for data processing utilities