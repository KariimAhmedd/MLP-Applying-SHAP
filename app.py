"""
Main application module for MLP SHAP Dashboard.

Technical Implementation Details:
-------------------------------
1. Architecture:
   - Multi-page Streamlit application
   - Modular design with separate visualization components
   - Session state management for data persistence

2. Core Components:
   - Dataset Management: Standardized loading and preprocessing
   - Model Configuration: Dynamic MLP architecture setup
   - Training Pipeline: Sklearn-based neural network training
   - State Management: Cross-page data sharing

3. Performance Optimizations:
   - Vectorized operations for data preprocessing
   - Efficient memory management via session state
   - Scalable visualization components

4. Dependencies:
   - streamlit: Web interface
   - scikit-learn: ML model implementation
   - shap: Model interpretability
   - numpy/pandas: Data processing
"""

import streamlit as st
st.set_page_config(page_title="MLP SHAP Dashboard", layout="wide")

import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import visualize_neural_network, suggest_activation_function
from sklearn.utils import Bunch
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Detect dataset patterns and suggest appropriate class names
def suggest_class_names(df, filename=""):
    """
    Analyzes a dataset and suggests appropriate class names based on patterns
    """
    # Initialize with default class names
    class_names = {}
    target = df.iloc[:, -1]
    unique_classes = np.unique(target)
    num_classes = len(unique_classes)
    
    # Analyze target variable name for clues
    target_name = df.columns[-1].lower()
    target_values = [str(v).lower() for v in target.unique()]
    
    # Check if target already has meaningful string values
    if target.dtype == 'object' or target.dtype.name == 'category':
        # Target might already have meaningful names
        for i, cls in enumerate(unique_classes):
            if isinstance(cls, str):
                class_names[i] = cls.title()  # Capitalize first letter
            else:
                class_names[i] = f"Class {i}"
        return class_names, "Using existing categorical values as class names"
    
    # Binary classification scenarios
    if num_classes == 2:
        # Check for common binary classification patterns
        binary_patterns = [
            # Format: (keywords, class_0_name, class_1_name)
            (['yes', 'no', 'y', 'n'], "No", "Yes"),
            (['true', 'false', 't', 'f'], "False", "True"),
            (['subscribe', 'deposit', 'purchase', 'buy'], "No Subscription", "Subscribed"),
            (['fraud', 'scam', 'anomaly'], "Normal", "Fraudulent"),
            (['default', 'loan', 'credit', 'payment'], "No Default", "Default"),
            (['churn', 'exit', 'retention', 'cancel'], "Retained", "Churned"),
            (['disease', 'diabetes', 'cancer', 'tumor', 'sick', 'health'], "Negative", "Positive"),
            (['spam', 'ham', 'email', 'message'], "Not Spam", "Spam"),
            (['gender', 'sex', 'male', 'female', 'm', 'f'], "Female", "Male"),
            (['approve', 'accept', 'reject', 'decline'], "Rejected", "Approved"),
            (['pass', 'fail', 'exam', 'test', 'grade'], "Failed", "Passed")
        ]
        
        # Check if target name or values match any pattern
        for keywords, name_0, name_1 in binary_patterns:
            if any(kw in target_name for kw in keywords) or any(kw in str(v).lower() for kw in keywords for v in target_values):
                return {0: name_0, 1: name_1}, f"Based on '{target_name}' patterns"
                
        # Default binary names
        return {0: "Negative", 1: "Positive"}, "Default binary classification labels"
    
    # Multi-class scenarios
    elif num_classes <= 10:
        # Check for common multi-class patterns
        
        # Check if it's potentially a rating system
        if any(word in target_name for word in ['rating', 'score', 'star', 'grade', 'rank']):
            class_names = {i: f"{i+1} Star" if i < 5 else f"{i+1} Stars" for i in range(num_classes)}
            return class_names, "Rating scale detected"
        
        # For 3-5 classes, check if it's a sentiment scale
        if 3 <= num_classes <= 5 and any(word in target_name for word in ['sentiment', 'satisfaction', 'opinion', 'feel']):
            if num_classes == 3:
                return {0: "Negative", 1: "Neutral", 2: "Positive"}, "Sentiment scale detected"
            elif num_classes == 5:
                return {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}, "Detailed sentiment scale detected"
    
    # Default: generic class names
    for i, cls in enumerate(unique_classes):
        class_names[i] = f"Class {i}"
    
    return class_names, "Default classification labels"

# Check device availability and set device
device = "cpu"
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")

st.sidebar.info(f"Using device: {device}")

# PyTorch MLP Model
class TorchMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.5):
        super(TorchMLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.batch_norms = torch.nn.ModuleList()
        self.loss_ = float('inf')
        self.n_iter_ = 0
        self.converged = False
        
        # Input layer with batch normalization
        self.layers.append(torch.nn.Linear(input_size, hidden_layers[0]))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_layers[i + 1]))
        
        # Output layer
        self.layers.append(torch.nn.Linear(hidden_layers[-1], num_classes))
        
        # Move model to device
        self.to(device)
        
        # Activation functions
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Apply layers with batch norm, activation, and dropout
        for i, (layer, batch_norm) in enumerate(zip(self.layers[:-1], self.batch_norms)):
            x = layer(x)
            x = batch_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        return self.softmax(x)

    def get_layer_weights(self):
        """Get weights for visualization"""
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
        return weights

    def predict(self, X):
        """Predict class labels"""
        # Convert to numpy array if X is a pandas DataFrame/Series
        if hasattr(X, 'values'):
            X = X.values
        
        # Ensure array is of correct type
        X = np.array(X, dtype=np.float32)
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = self(X_tensor)
            _, predictions = torch.max(outputs, 1)
            return predictions.cpu().numpy()

    def predict_proba(self, X):
        """Predict class probabilities"""
        # Convert to numpy array if X is a pandas DataFrame/Series
        if hasattr(X, 'values'):
            X = X.values
        
        # Ensure array is of correct type
        X = np.array(X, dtype=np.float32)
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = self(X_tensor)
            return outputs.cpu().numpy()

def train_torch_model(X_train, y_train, hidden_layers, num_epochs=100, batch_size=32):
    # Convert data to numpy arrays if they are pandas Series/DataFrame
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    # Ensure arrays are of correct type
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    
    # Create data loader for batch training
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model with improved architecture
    model = TorchMLP(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        num_classes=len(np.unique(y_train)),
        dropout_rate=0.5
    )
    
    # Loss and optimizer with learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 10
    no_improve_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_X, batch_y in loader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / num_batches
            model.loss_ = avg_loss
            model.n_iter_ = epoch + 1
            
            # Update progress
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Update learning rate
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                model.converged = True
                st.info(f"Training converged early at epoch {epoch + 1}")
                break
    
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        raise e
    
    finally:
        progress_bar.empty()
        status_text.empty()
    
    return model

st.title("🔍 Interactive MLP + SHAP Visualizer")

# 🔷 Explanation Box
with st.expander("ℹ️ About This App", expanded=True):
    st.markdown("""
    ### 🤖 Multi-Layer Perceptron (MLP) with SHAP Explanations

    This app trains a **Multi-Layer Perceptron (MLP)** model and explains its predictions using **SHAP (SHapley Additive exPlanations)**.

    ### 📚 Available Pages:
    1. **Neural Network Visualization**: Interactive visualization of the network architecture
    2. **SHAP Explanations**: Detailed model interpretability using SHAP values
    
    ### 🎯 How to Use:
    1. Configure your model in the sidebar
    2. Navigate between pages using the sidebar
    3. Interact with visualizations to understand your model
    
    👨‍🏫 This app helps you understand neural networks through interactive visualizations.
    """)

# Initialize session state for model and data
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
    st.session_state.scaler = StandardScaler()

# Load and preprocess fake news dataset
@st.cache_data
def load_fake_news():
    try:
        # Load the dataset
        df = pd.read_csv('fake_news_dataset.csv')
        
        # Basic preprocessing
        text_features = ['title', 'text']
        target = 'label'
        
        # Combine text features
        df['combined_text'] = df[text_features].fillna('').agg(' '.join, axis=1)
        
        # Convert text to TF-IDF features with better parameters
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            strip_accents='unicode',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        
        X = vectorizer.fit_transform(df['combined_text']).toarray()
        y_raw = df[target].values
        
        # Initialize label encoder for target variable
        if 'label_encoder' not in st.session_state:
            st.session_state.label_encoder = LabelEncoder()
        y = st.session_state.label_encoder.fit_transform(y_raw)
        
        # Create feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Create a Bunch object to match scikit-learn format
        bunch_obj = Bunch(
            data=X,
            target=y,
            feature_names=feature_names,
            target_names=np.array(['Real', 'Fake']),
            DESCR="Fake News Dataset",
            vectorizer=vectorizer
        )
        
        return bunch_obj
    except Exception as e:
        st.error(f"Error loading fake news dataset: {str(e)}")
        return None

# 1. Dataset Selection
dataset_source = st.sidebar.radio("Dataset Source", ["Built-in", "Fake News", "Custom Upload"])

if dataset_source == "Built-in":
    dataset_name = st.sidebar.selectbox("Select Dataset", ["Breast Cancer", "Iris", "Wine"])
    
    def load_dataset(name):
        if name == "Breast Cancer":
            return load_breast_cancer(as_frame=True)
        elif name == "Iris":
            return load_iris(as_frame=True)
        elif name == "Wine":
            return load_wine(as_frame=True)
    
    data = load_dataset(dataset_name)
    X = data.data
    y = data.target
    
    # Initialize label encoder if needed
    if not np.issubdtype(y.dtype, np.number):
        if 'label_encoder' not in st.session_state:
            st.session_state.label_encoder = LabelEncoder()
        y = st.session_state.label_encoder.fit_transform(y)
    
    # Now convert to int32 after encoding
    y = y.astype(np.int32)
    
    # Store dimensions
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    
    # Show success message with dataset information
    st.sidebar.success(f"""
    Model initialized successfully:
    - Features: {num_features}
    - Samples: {X.shape[0]}
    - Classes: {', '.join(data.target_names)}
    """)
    
    # 3. Preprocessing & Training
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled.astype(np.float32)
    
    # Store the fitted scaler in session state for later use
    st.session_state['scaler'] = scaler
    
    # Ensure arrays are numpy arrays
    X_scaled = np.array(X_scaled, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Store necessary objects in session state
    st.session_state['data'] = data
    st.session_state['X'] = X
    st.session_state['y'] = y
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    st.session_state['num_features'] = num_features
    st.session_state['hidden_layers'] = []
    st.session_state['num_classes'] = num_classes

    # Store class names/meanings for interpretability
    if hasattr(data, 'target_names') and len(data.target_names) > 0:
        st.session_state['class_names'] = {i: name for i, name in enumerate(data.target_names)}
    elif dataset_name == "Breast Cancer":
        st.session_state['class_names'] = {0: "Malignant", 1: "Benign"}
    elif dataset_name == "Iris":
        st.session_state['class_names'] = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    elif dataset_name == "Wine":
        st.session_state['class_names'] = {0: "Wine Type 1", 1: "Wine Type 2", 2: "Wine Type 3"}

    # Allow users to provide class names for better interpretability
    st.sidebar.write("📝 **Define Class Names for Better Interpretability**")
    class_names = {}
    for cls in np.unique(y):
        name = st.sidebar.text_input(f"Name for Class {cls}:", value=f"Class {cls}", key=f"class_{cls}")
        class_names[int(cls)] = name
    
    # Store class names in session state
    st.session_state['class_names'] = class_names

elif dataset_source == "Fake News":
    # Load and initialize model if not already done
    if not st.session_state.model_initialized:
        with st.spinner("Loading and preparing the fake news model..."):
            data = load_fake_news()
            if data is not None:
                # Store necessary components in session state
                st.session_state.X = data.data
                st.session_state.y = data.target
                st.session_state.vectorizer = data.vectorizer
                st.session_state.feature_names = data.feature_names
                
                # Prepare the model
                X_scaled = st.session_state.scaler.fit_transform(data.data)
                X_scaled = X_scaled.astype(np.float32)
                y = data.target.astype(np.int32)
                
                # Ensure arrays are numpy arrays
                X_scaled = np.array(X_scaled, dtype=np.float32)
                y = np.array(y, dtype=np.int32)
                
                # Split and train
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                
                # Define architecture
                input_size = X_scaled.shape[1]
                hidden_layers = [256, 128, 64]  # Adjusted architecture for text classification
                num_classes = len(np.unique(y))
                
                # Train model
                model = train_torch_model(X_train, y_train, hidden_layers)
                
                # Store everything in session state
                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.num_features = input_size
                st.session_state.hidden_layers = hidden_layers
                st.session_state.num_classes = num_classes
                
                # Mark as initialized
                st.session_state.model_initialized = True
                
                # Show success message
                st.sidebar.success(f"""
                Model initialized successfully:
                - Features: {input_size}
                - Samples: {len(y)}
                - Classes: Real, Fake
                """)
            else:
                st.error("Failed to load fake news dataset")
                st.stop()
    
    # Initialize input state variables
    if 'article_title' not in st.session_state:
        st.session_state.article_title = ""
    if 'article_content' not in st.session_state:
        st.session_state.article_content = ""
    if 'article_date' not in st.session_state:
        st.session_state.article_date = ""
    if 'article_source' not in st.session_state:
        st.session_state.article_source = ""
    if 'article_author' not in st.session_state:
        st.session_state.article_author = ""
    if 'article_category' not in st.session_state:
        st.session_state.article_category = ""
    if 'predict_clicked' not in st.session_state:
        st.session_state.predict_clicked = False

    # Function to handle button click
    def on_predict_click():
        st.session_state.predict_clicked = True

    # Text input for fake news prediction
    st.write("📝 Enter the article details to classify:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.article_title = st.text_input("Title of the article:", key="title_input")
        st.session_state.article_date = st.date_input("Article Date:", key="date_input")
        st.session_state.article_source = st.text_input("Source (e.g., website, newspaper):", key="source_input")
    
    with col2:
        st.session_state.article_author = st.text_input("Author:", key="author_input")
        st.session_state.article_category = st.selectbox(
            "Category:",
            ["Politics", "Technology", "Health", "Business", "Entertainment", "Sports", "Other"],
            key="category_input"
        )
    
    st.session_state.article_content = st.text_area("Content of the article:", height=200, key="content_input")
    
    # Create predict button
    st.button("Predict", on_click=on_predict_click, use_container_width=True)
    
    # Only show prediction if button was clicked
    if st.session_state.predict_clicked:
        try:
            # Use components from session state
            vectorizer = st.session_state.vectorizer
            model = st.session_state.model
            
            # Combine all features for prediction
            metadata = f"""
            Title: {st.session_state.article_title}
            Author: {st.session_state.article_author}
            Source: {st.session_state.article_source}
            Category: {st.session_state.article_category}
            Date: {st.session_state.article_date}
            """
            
            combined_text = f"{metadata}\n\nContent: {st.session_state.article_content}"
            
            # Transform text
            input_features = vectorizer.transform([combined_text]).toarray()
            
            # Use the scaler from session state, or handle the case when it's not available
            if 'scaler' in st.session_state:
                input_scaled = st.session_state['scaler'].transform(input_features)
            else:
                st.error("Model scaler not found. Please retrain the model first.")
                st.stop()
            
            # Make prediction using PyTorch model
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Convert prediction back to original label if label encoder exists
            display_prediction = prediction
            if 'label_encoder' in st.session_state:
                display_prediction = st.session_state['label_encoder'].inverse_transform([prediction])[0]
            
            # Get class meaning if available
            class_meaning = ""
            if 'class_names' in st.session_state and prediction in st.session_state['class_names']:
                class_meaning = st.session_state['class_names'][prediction]
            
            # Create a more detailed analysis card
            st.write("---")
            st.subheader("📊 Analysis Results")
            
            # Display metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Article Details**")
                st.write(f"Source: {st.session_state.article_source}")
                st.write(f"Date: {st.session_state.article_date}")
            with col2:
                st.markdown("**Author Info**")
                st.write(f"Author: {st.session_state.article_author}")
                st.write(f"Category: {st.session_state.article_category}")
            with col3:
                st.markdown("**Prediction Results**")
                confidence = max(prediction_proba)
                if prediction == 1:
                    st.error("⚠️ **LIKELY FAKE**")
                else:
                    st.success("✅ **LIKELY REAL**")
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Show detailed probability distribution
            st.write("---")
            st.subheader("Probability Distribution")
            fig = plt.figure(figsize=(10, 4))
            plt.bar(['Real', 'Fake'], prediction_proba, 
                   color=['#99ff99', '#ff9999'])
            plt.title("Classification Confidence")
            plt.ylabel("Probability")
            plt.ylim(0, 1)
            for i, prob in enumerate(prediction_proba):
                plt.text(i, prob, f'{prob:.1%}', ha='center', va='bottom')
            st.pyplot(fig)
            plt.close()
            
            # Show credibility indicators
            st.write("---")
            st.subheader("🔍 Credibility Indicators")
            
            # Calculate credibility score based on metadata
            credibility_score = 0
            indicators = []
            
            # Source check
            if st.session_state.article_source:
                credibility_score += 1
                indicators.append(("✅ Source provided", "green"))
            else:
                indicators.append(("❌ No source provided", "red"))
            
            # Author check
            if st.session_state.article_author:
                credibility_score += 1
                indicators.append(("✅ Author identified", "green"))
            else:
                indicators.append(("❌ Anonymous content", "red"))
            
            # Date check
            if st.session_state.article_date:
                credibility_score += 1
                indicators.append(("✅ Date provided", "green"))
            else:
                indicators.append(("❌ No date provided", "red"))
            
            # Content length check
            if len(st.session_state.article_content.split()) > 100:
                credibility_score += 1
                indicators.append(("✅ Detailed content", "green"))
            else:
                indicators.append(("❌ Limited content", "red"))
            
            # Display credibility indicators
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Credibility Score", f"{credibility_score}/4")
            
            with col2:
                for indicator, color in indicators:
                    st.markdown(f"<span style='color: {color}'>{indicator}</span>", unsafe_allow_html=True)

            # Add class meaning explanation
            st.info(f"**Class Interpretation**: The prediction '{display_prediction}' represents '{class_meaning}'")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Debug: Exception details:", type(e).__name__, str(e))
        finally:
            st.session_state.predict_clicked = False
    else:
        st.stop()

else:
    st.sidebar.markdown("""
    ### Upload Custom Dataset
    Please ensure your dataset follows these guidelines:
    - CSV or Excel format (.csv, .xls, .xlsx)
    - Last column should be the target variable
    
    The app will automatically:
    ✓ Handle missing values
    ✓ Remove outliers
    ✓ Convert categorical features to numeric
    ✓ Normalize features
    
    **Categorical Data Support:**
    This app handles categorical features by converting them to numeric values using label encoding.
    """)
    
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Read the dataset based on file type
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:  # Excel files
                df = pd.read_excel(uploaded_file)
            
            # Store original shape for reporting
            original_shape = df.shape
            
            # Process the dataset
            with st.spinner("Processing uploaded dataset..."):
                # Display a progress bar
                progress_bar = st.progress(0)
                
                # 1. Basic info and extraction (20%)
                X = df.iloc[:, :-1]  # All columns except the last one
                y = df.iloc[:, -1]   # Last column as target
                progress_bar.progress(20)
                
                # 2. Handling missing values (40%)
                missing_before = X.isna().sum().sum()
                
                # Handle numeric and categorical columns separately for missing values
                numeric_cols = X.select_dtypes(include=['number']).columns
                if not numeric_cols.empty:
                    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                
                categorical_cols = X.select_dtypes(exclude=['number']).columns
                if not categorical_cols.empty:
                    for col in categorical_cols:
                        X[col] = X[col].fillna(X[col].mode().iloc[0])
                
                progress_bar.progress(40)
                
                # 3. Removing outliers - only for numeric columns (60%)
                if not numeric_cols.empty:
                    Q1 = X[numeric_cols].quantile(0.25)
                    Q3 = X[numeric_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Create a mask for non-outlier rows
                    outlier_mask = ~((X[numeric_cols] < (Q1 - 3 * IQR)) | (X[numeric_cols] > (Q3 + 3 * IQR))).any(axis=1)
                    X = X[outlier_mask]
                    y = y[outlier_mask]
                
                progress_bar.progress(60)
                
                # 4. Type conversion - process each categorical column individually (80%)
                # Convert non-numeric columns using LabelEncoder
                non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
                
                if len(non_numeric_cols) > 0:
                    for col in non_numeric_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                
                # Handle target variable
                if not pd.api.types.is_numeric_dtype(y):
                    try:
                        # Try to convert directly if it contains numeric strings
                        y = pd.to_numeric(y, errors='raise')
                    except:
                        # Otherwise use label encoding
                        unique_labels = y.unique()
                        label_map = {label: idx for idx, label in enumerate(unique_labels)}
                        y = y.map(label_map)
                
                progress_bar.progress(90)
                
                # 5. Finalize processing (100%)
                df = pd.concat([X, y], axis=1)
                progress_bar.progress(100)
                
                # Show summary in expandable section
                with st.expander("🔍 Data Processing Summary", expanded=False):
                    st.write(f"""
                    ✨ Processing Complete:
                    - Rows: {df.shape[0]} (originally {original_shape[0]})
                    - Columns: {df.shape[1]}
                    - Missing values handled: {missing_before if missing_before > 0 else 'None found'}
                    - Data types: All converted to numeric
                    """)
                
                # Quick preview of cleaned data
                with st.expander("Preview Cleaned Data", expanded=False):
                    st.dataframe(df.head())
            
            # Create compatible data object
            data = Bunch(
                data=X,
                target=y,
                feature_names=X.columns.tolist(),
                target_names=np.unique(y),
                DESCR=f"Processed dataset with {X.shape[1]} features",
                filename=uploaded_file.name
            )
            
            # Fit and store scaler in session state
            scaler = StandardScaler()
            scaler.fit(X)
            st.session_state['scaler'] = scaler
            
            # Show success message with dataset information
            st.sidebar.success(f"""
            Model initialized successfully:
            - Features: {X.shape[1]}
            - Samples: {X.shape[0]}
            - Classes: {', '.join([str(c) for c in np.unique(y)])}
            """)

            # Get suggested class names based on dataset patterns
            suggested_class_names, suggestion_reason = suggest_class_names(df, uploaded_file.name)
            
            # Allow users to provide class names with suggestions
            st.sidebar.write("📝 **Define Class Names for Better Interpretability**")
            st.sidebar.info(f"{suggestion_reason}")
            
            class_names = {}
            for cls in np.unique(y):
                cls_int = int(cls) if isinstance(cls, (int, np.integer)) else cls
                suggested_name = suggested_class_names.get(cls_int, f"Class {cls_int}")
                name = st.sidebar.text_input(
                    f"Name for Class {cls}:",
                    value=suggested_name,
                    key=f"class_{cls}",
                    help=f"Suggested name based on dataset analysis: {suggested_name}"
                )
                class_names[cls_int] = name
            
            # Store class names in session state
            st.session_state['class_names'] = class_names

            # Dataset Visualization Section
            st.write("---")
            st.header("📊 Dataset Analysis")
            
            tab1, tab2, tab3 = st.tabs(["📈 Distribution", "🔍 Statistics", "🎯 Target Analysis"])
            
            with tab1:
                st.subheader("Feature Distributions")
                
                # Select features to visualize
                num_cols = st.multiselect(
                    "Select features to visualize:",
                    options=X.columns.tolist(),
                    default=X.columns[:min(5, len(X.columns))].tolist()
                )
                
                if num_cols:
                    # Create distribution plots
                    fig = plt.figure(figsize=(12, 4 * ((len(num_cols) + 1) // 2)))
                    for idx, col in enumerate(num_cols, 1):
                        plt.subplot((len(num_cols) + 1) // 2, 2, idx)
                        plt.hist(X[col], bins=30, edgecolor='black')
                        plt.title(f'Distribution of {col}')
                        plt.xlabel('Value')
                        plt.ylabel('Frequency')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            with tab2:
                st.subheader("Dataset Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information**")
                    st.write(f"- Total Samples: {X.shape[0]}")
                    st.write(f"- Total Features: {X.shape[1]}")
                    st.write(f"- Target Classes: {len(np.unique(y))}")
                
                with col2:
                    st.write("**Feature Summary**")
                    stats_df = X.describe()
                    st.dataframe(stats_df)
                
                # Correlation matrix
                st.write("**Feature Correlation Matrix**")
                corr_matrix = X.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
                plt.colorbar()
                plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
                plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
                plt.title('Feature Correlation Matrix')
                st.pyplot(fig)
                plt.close()
            
            with tab3:
                st.subheader("Target Variable Analysis")
                
                # Target distribution
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Count plot
                unique_counts = pd.Series(y).value_counts()
                ax1.bar(range(len(unique_counts)), unique_counts)
                ax1.set_title('Target Class Distribution')
                ax1.set_xlabel('Class')
                ax1.set_ylabel('Count')
                
                # Pie chart
                ax2.pie(unique_counts, labels=[f'Class {i}' for i in range(len(unique_counts))],
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title('Target Class Proportions')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Feature importance for target
                st.write("**Feature Importance Analysis**")
                from sklearn.feature_selection import mutual_info_classif
                
                # Calculate feature importance scores
                importance_scores = mutual_info_classif(X, y)
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig, ax = plt.subplots(figsize=(10, 4))
                plt.bar(importance_df['Feature'], importance_df['Importance'])
                plt.xticks(rotation=45, ha='right')
                plt.title('Feature Importance for Target Prediction')
                plt.xlabel('Features')
                plt.ylabel('Importance Score')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            st.write("---")
            
        except Exception as e:
            st.sidebar.error(f"""
            Error processing dataset: {str(e)}
            
            Please ensure your dataset:
            1. Is in CSV or Excel format
            2. Has the target variable in the last column
            3. Categorical features are properly formatted (the app will automatically encode them)
            4. There are no corrupt or incompatible data entries
            """)
            st.stop()
    else:
        # Show example dataset format
        st.sidebar.markdown("### Example Dataset Format")
        example_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        st.sidebar.dataframe(example_data)
        st.stop()

# 2. Architecture and Hyperparameters
st.sidebar.header("Neural Network Architecture")
num_features = X.shape[1]
num_classes = len(np.unique(y))

# Multiple hidden layers configuration with feedback
num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 3, 1)
st.sidebar.markdown("""
🔸 **Current: {0} layer{1}**
- 1 Layer: Simple patterns, linear relationships
- 2 Layers: Moderate complexity, non-linear patterns
- 3 Layers: Complex patterns, deep feature learning
""".format(num_hidden_layers, 's' if num_hidden_layers > 1 else ''))

hidden_layers = []
for i in range(num_hidden_layers):
    nodes = st.sidebar.slider(f"Nodes in Hidden Layer {i+1}", 5, 200, 50, step=5)
    hidden_layers.append(nodes)
    
    # Add explanation under each node slider
    st.sidebar.markdown("""
    🔹 **Layer {0}: {1} nodes**
    - 5-20: Fast but limited learning
    - 20-100: Good balance of speed/capacity
    - 100-200: Complex patterns but slower
    Current choice: {2}
    """.format(
        i+1, 
        nodes, 
        'Limited capacity' if nodes < 20 else 'Balanced' if nodes < 100 else 'High capacity'
    ))

st.sidebar.header("Training Parameters")
activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh", "logistic"])
activation_info = {
    "relu": "- Best for deep networks\n- Fast training\n- No vanishing gradient",
    "tanh": "- Good for normalized data\n- Symmetric around zero\n- Smooth gradients",
    "logistic": "- Best for binary outputs\n- Range [0,1]\n- Probability interpretation"
}
st.sidebar.markdown("""
📊 **{0}**
{1}
""".format(activation.upper(), activation_info[activation]))

learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001, step=0.0005)
st.sidebar.markdown("""
🎯 **Current: {0:.4f}**
- 0.0001-0.001: Very stable, slow learning
- 0.001-0.01: Good balance speed/stability
- 0.01-0.1: Fast but potentially unstable
Your choice is: {1}
""".format(
    learning_rate,
    'Very stable' if learning_rate < 0.001 else 'Balanced' if learning_rate < 0.01 else 'Aggressive'
))

max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 300, step=100)
st.sidebar.markdown("""
⏱️ **Current: {0} iterations**
- 100-300: Quick but might not converge
- 300-700: Good for most cases
- 700-1000: Deep learning, risk of overfitting
Your setting: {1}
""".format(
    max_iter,
    'Quick training' if max_iter < 300 else 'Standard training' if max_iter < 700 else 'Deep training'
))

# Check if MPS (Metal Performance Shaders) is available
device = (
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
st.sidebar.info(f"Using device: {device}")

# PyTorch MLP Model
class TorchMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, dropout_rate=0.5):
        super(TorchMLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.batch_norms = torch.nn.ModuleList()
        self.loss_ = float('inf')
        self.n_iter_ = 0
        self.converged = False
        
        # Input layer with batch normalization
        self.layers.append(torch.nn.Linear(input_size, hidden_layers[0]))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_layers[i + 1]))
        
        # Output layer
        self.layers.append(torch.nn.Linear(hidden_layers[-1], num_classes))
        
        # Move model to device
        self.to(device)
        
        # Activation functions
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Apply layers with batch norm, activation, and dropout
        for i, (layer, batch_norm) in enumerate(zip(self.layers[:-1], self.batch_norms)):
            x = layer(x)
            x = batch_norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.layers[-1](x)
        return self.softmax(x)

    def get_layer_weights(self):
        """Get weights for visualization"""
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.data.cpu().numpy())
        return weights

    def predict(self, X):
        """Predict class labels"""
        # Convert to numpy array if X is a pandas DataFrame/Series
        if hasattr(X, 'values'):
            X = X.values
        
        # Ensure array is of correct type
        X = np.array(X, dtype=np.float32)
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = self(X_tensor)
            _, predictions = torch.max(outputs, 1)
            return predictions.cpu().numpy()

    def predict_proba(self, X):
        """Predict class probabilities"""
        # Convert to numpy array if X is a pandas DataFrame/Series
        if hasattr(X, 'values'):
            X = X.values
        
        # Ensure array is of correct type
        X = np.array(X, dtype=np.float32)
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            outputs = self(X_tensor)
            return outputs.cpu().numpy()

# Modified training function with loss tracking
def train_torch_model(X_train, y_train, hidden_layers, num_epochs=100, batch_size=32):
    # Convert data to numpy arrays if they are pandas Series/DataFrame
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    # Ensure arrays are of correct type
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    
    # Create data loader for batch training
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model with improved architecture
    model = TorchMLP(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        num_classes=len(np.unique(y_train)),
        dropout_rate=0.5
    )
    
    # Loss and optimizer with learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 10
    no_improve_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_X, batch_y in loader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / num_batches
            model.loss_ = avg_loss
            model.n_iter_ = epoch + 1
            
            # Update progress
            progress = (epoch + 1) / num_epochs
            progress_bar.progress(progress)
            status_text.text(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Update learning rate
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                model.converged = True
                st.info(f"Training converged early at epoch {epoch + 1}")
                break
    
    except Exception as e:
        st.error(f"Error during training: {str(e)}")
        raise e
    
    finally:
        progress_bar.empty()
        status_text.empty()
    
    return model

# Modified prediction function
def predict_torch(model, X):
    # Convert to numpy array if X is a pandas DataFrame/Series
    if hasattr(X, 'values'):
        X = X.values
    
    # Ensure array is of correct type
    X = np.array(X, dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        probabilities = outputs.cpu().numpy()
        predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities

# Modified network visualization function
def visualize_torch_network(model, num_features, hidden_layers, num_classes):
    import graphviz
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Neural Network')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Add input nodes
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer')
        for i in range(min(num_features, 10)):  # Limit to 10 nodes for visibility
            c.node(f'i{i}', f'Input {i}')
        if num_features > 10:
            c.node('i...', '...')
    
    # Add hidden layers
    for l, layer_size in enumerate(hidden_layers):
        with dot.subgraph(name=f'cluster_h{l}') as c:
            c.attr(label=f'Hidden Layer {l+1}')
            for i in range(min(layer_size, 10)):  # Limit to 10 nodes for visibility
                c.node(f'h{l}_{i}', f'H{l+1}_{i}')
            if layer_size > 10:
                c.node(f'h{l}_...', '...')
    
    # Add output nodes
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output Layer')
        for i in range(num_classes):
            c.node(f'o{i}', f'Output {i}')
    
    # Add edges with weights
    weights = model.get_layer_weights()
    max_weight = max(abs(w).max() for w in weights)
    
    # Add some representative edges
    def add_representative_edges(from_nodes, to_nodes, weights, layer_idx):
        from_count = len(from_nodes)
        to_count = len(to_nodes)
        
        # Add edges for visible nodes
        for i in range(min(from_count, 3)):  # Show only first 3 connections
            for j in range(min(to_count, 3)):
                if i < weights.shape[1] and j < weights.shape[0]:
                    weight = weights[j, i]
                    width = abs(weight) / max_weight * 3
                    color = 'red' if weight < 0 else 'green'
                    dot.edge(from_nodes[i], to_nodes[j], 
                            color=color,
                            penwidth=str(width))
    
    # Connect input to first hidden layer
    input_nodes = [f'i{i}' for i in range(min(num_features, 10))]
    if num_features > 10:
        input_nodes.append('i...')
    
    for l in range(len(hidden_layers)):
        if l == 0:
            from_nodes = input_nodes
        else:
            from_nodes = [f'h{l-1}_{i}' for i in range(min(hidden_layers[l-1], 10))]
            if hidden_layers[l-1] > 10:
                from_nodes.append(f'h{l-1}_...')
        
        to_nodes = [f'h{l}_{i}' for i in range(min(hidden_layers[l], 10))]
        if hidden_layers[l] > 10:
            to_nodes.append(f'h{l}_...')
        
        add_representative_edges(from_nodes, to_nodes, weights[l], l)
    
    # Connect last hidden layer to output
    last_hidden = [f'h{len(hidden_layers)-1}_{i}' for i in range(min(hidden_layers[-1], 10))]
    if hidden_layers[-1] > 10:
        last_hidden.append(f'h{len(hidden_layers)-1}_...')
    output_nodes = [f'o{i}' for i in range(num_classes)]
    add_representative_edges(last_hidden, output_nodes, weights[-1], -1)
    
    return dot

# Preprocessing for model training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.astype(np.float32)

# Store the fitted scaler in session state for later use
st.session_state['scaler'] = scaler

# Ensure X_scaled and y are proper numpy arrays before splitting
if hasattr(X_scaled, 'values'):
    X_scaled = X_scaled.values
if hasattr(y, 'values'):
    y = y.values
X_scaled = np.array(X_scaled, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train PyTorch model
with st.spinner("Training model on GPU..."):
    model = train_torch_model(X_train, y_train, hidden_layers)
    st.session_state['model'] = model

# Store necessary objects in session state
st.session_state['data'] = data
st.session_state['X'] = X
st.session_state['y'] = y
st.session_state['X_train'] = X_train
st.session_state['X_test'] = X_test
st.session_state['y_train'] = y_train
st.session_state['y_test'] = y_test
st.session_state['num_features'] = num_features
st.session_state['hidden_layers'] = hidden_layers
st.session_state['num_classes'] = num_classes

# Store class names/meanings for interpretability
if hasattr(data, 'target_names') and len(data.target_names) > 0:
    st.session_state['class_names'] = {i: name for i, name in enumerate(data.target_names)}
elif dataset_source == "Built-in":
    if dataset_name == "Breast Cancer":
        st.session_state['class_names'] = {0: "Malignant", 1: "Benign"}
    elif dataset_name == "Iris":
        st.session_state['class_names'] = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    elif dataset_name == "Wine":
        st.session_state['class_names'] = {0: "Wine Type 1", 1: "Wine Type 2", 2: "Wine Type 3"}
elif dataset_source == "Fake News":
    st.session_state['class_names'] = {0: "Real News", 1: "Fake News"}
else:
    # For custom datasets, use class indices with generic labels
    st.session_state['class_names'] = {i: f"Class {i}" for i in range(num_classes)}

# Display basic model performance
st.header("📊 Model Performance")
col1, col2 = st.columns(2)
with col1:
    train_pred, _ = predict_torch(model, X_train)
    train_acc = np.mean(train_pred == y_train)
    st.success(f"✅ Training Accuracy: {train_acc:.2%}")
with col2:
    test_pred, _ = predict_torch(model, X_test)
    test_acc = np.mean(test_pred == y_test)
    st.success(f"✅ Testing Accuracy: {test_acc:.2%}")

# Main content area with better spacing and visibility
st.write("---")  # Add a visual separator

# Neural Network Visualization Section
st.header("🧠 Neural Network Architecture")
st.markdown("""
This section shows the structure of your neural network. The visualization includes:
- Input layer with feature nodes
- Hidden layers with configurable neurons
- Output layer with class nodes
- Connection weights shown by color and thickness
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Network Visualization")
    try:
        graph = visualize_torch_network(model, num_features, hidden_layers, num_classes)
        st.graphviz_chart(graph, use_container_width=True)
    except Exception as e:
        st.error(f"Error in network visualization: {str(e)}")
        st.info("Try adjusting the network architecture or refreshing the page.")

with col2:
    st.subheader("Network Statistics")
    st.markdown(f"""
    **Model Architecture:**
    - Input Features: {num_features}
    - Hidden Layers: {len(hidden_layers)}
    - Neurons per layer: {hidden_layers}
    - Output Classes: {num_classes}
    
    **Training Details:**
    - Device: {device}
    - Final Loss: {model.loss_:.4f}
    - Optimizer: Adam
    - Learning Rate: 0.001
    """)

# Model Performance Section with enhanced visibility
st.write("---")
st.header("📈 Model Performance Analysis")

# Create three columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Training Accuracy",
        value=f"{train_acc:.2%}",
        delta=f"{train_acc - 0.5:.2%} vs baseline"
    )

with col2:
    st.metric(
        label="Testing Accuracy",
        value=f"{test_acc:.2%}",
        delta=f"{test_acc - train_acc:.2%} vs training"
    )

with col3:
    # Calculate convergence status
    convergence_status = (
        "converged early" if model.converged else
        "reached max epochs" if model.n_iter_ >= max_iter else
        "training incomplete"
    )
    
    st.metric(
        label="Final Loss",
        value=f"{model.loss_:.4f}",
        delta=convergence_status
    )

# Performance Analysis
if train_acc > test_acc + 0.1:
    st.warning("""
    ⚠️ **Potential Overfitting Detected**
    - Training accuracy is significantly higher than testing accuracy
    - Consider:
        1. Reducing network complexity
        2. Adding dropout layers
        3. Using fewer epochs
    """)
elif test_acc < 0.7:
    st.warning("""
    ⚠️ **Model May Be Underfitting**
    - Accuracy is lower than expected
    - Consider:
        1. Increasing network complexity
        2. Training for more epochs
        3. Adjusting learning rate
    """)
else:
    st.success("""
    ✅ **Model Performance Looks Good**
    - Training and testing accuracies are balanced
    - No significant overfitting or underfitting detected
    """)

# Footer with additional information
st.write("---")
st.markdown("""
### 📚 Additional Information
- The neural network visualization shows the complete structure of your model
- Performance metrics help identify if the model is overfitting or underfitting
- Experiment with different architectures and parameters to improve performance
""")

# Add a download button for the model architecture
if st.button("📥 Download Model Architecture"):
    try:
        dot_data = graph.pipe(format='dot').decode('utf-8')
        st.download_button(
            label="Download DOT file",
            data=dot_data,
            file_name="neural_network_architecture.dot",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"Error generating download: {str(e)}")

# Add inference section
st.write("---")
st.header("🎯 Model Predictions")

# Simplified interface with two clear options
prediction_mode = st.radio(
    "Choose Prediction Mode:",
    ["✨ Quick Predict (Use Sliders)", "📊 Batch Predict (Test Set)"],
    help="Quick Predict: Make individual predictions using sliders\nBatch Predict: Test multiple samples at once"
)

if prediction_mode == "✨ Quick Predict (Use Sliders)":
    st.subheader("Make a Prediction")
    
    if dataset_source == "Fake News":
        # Initialize session state for inputs if not exists
        if 'article_title' not in st.session_state:
            st.session_state.article_title = ""
        if 'article_content' not in st.session_state:
            st.session_state.article_content = ""
        if 'article_date' not in st.session_state:
            st.session_state.article_date = ""
        if 'article_source' not in st.session_state:
            st.session_state.article_source = ""
        if 'article_author' not in st.session_state:
            st.session_state.article_author = ""
        if 'article_category' not in st.session_state:
            st.session_state.article_category = ""
        if 'predict_clicked' not in st.session_state:
            st.session_state.predict_clicked = False

        # Function to handle button click
        def on_predict_click():
            st.session_state.predict_clicked = True

        # Text input for fake news prediction
        st.write("📝 Enter the article details to classify:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.article_title = st.text_input("Title of the article:", key="title_input")
            st.session_state.article_date = st.date_input("Article Date:", key="date_input")
            st.session_state.article_source = st.text_input("Source (e.g., website, newspaper):", key="source_input")
        
        with col2:
            st.session_state.article_author = st.text_input("Author:", key="author_input")
            st.session_state.article_category = st.selectbox(
                "Category:",
                ["Politics", "Technology", "Health", "Business", "Entertainment", "Sports", "Other"],
                key="category_input"
            )
        
        st.session_state.article_content = st.text_area("Content of the article:", height=200, key="content_input")
        
        # Create predict button
        st.button("Predict", on_click=on_predict_click, use_container_width=True)
        
        # Only show prediction if button was clicked
        if st.session_state.predict_clicked:
            try:
                # Combine all features for prediction
                metadata = f"""
                Title: {st.session_state.article_title}
                Author: {st.session_state.article_author}
                Source: {st.session_state.article_source}
                Category: {st.session_state.article_category}
                Date: {st.session_state.article_date}
                """
                
                combined_text = f"{metadata}\n\nContent: {st.session_state.article_content}"
                
                # Transform text
                vectorizer = st.session_state['vectorizer']
                input_features = vectorizer.transform([combined_text]).toarray()
                
                # Use the scaler from session state, or handle the case when it's not available
                if 'scaler' in st.session_state:
                    input_scaled = st.session_state['scaler'].transform(input_features)
                else:
                    st.error("Model scaler not found. Please retrain the model first.")
                    st.stop()
                
                # Make prediction using PyTorch model
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                
                # Convert prediction back to original label if label encoder exists
                display_prediction = prediction
                if 'label_encoder' in st.session_state:
                    display_prediction = st.session_state['label_encoder'].inverse_transform([prediction])[0]
                
                # Get class meaning if available
                class_meaning = ""
                if 'class_names' in st.session_state and prediction in st.session_state['class_names']:
                    class_meaning = st.session_state['class_names'][prediction]
                
                # Create a more detailed analysis card
                st.write("---")
                st.subheader("📊 Analysis Results")
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Article Details**")
                    st.write(f"Source: {st.session_state.article_source}")
                    st.write(f"Date: {st.session_state.article_date}")
                with col2:
                    st.markdown("**Author Info**")
                    st.write(f"Author: {st.session_state.article_author}")
                    st.write(f"Category: {st.session_state.article_category}")
                with col3:
                    st.markdown("**Prediction Results**")
                    confidence = max(prediction_proba)
                    if prediction == 1:
                        st.error("⚠️ **LIKELY FAKE**")
                    else:
                        st.success("✅ **LIKELY REAL**")
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Show detailed probability distribution
                st.write("---")
                st.subheader("Probability Distribution")
                fig = plt.figure(figsize=(10, 4))
                plt.bar(['Real', 'Fake'], prediction_proba, 
                       color=['#99ff99', '#ff9999'])
                plt.title("Classification Confidence")
                plt.ylabel("Probability")
                plt.ylim(0, 1)
                for i, prob in enumerate(prediction_proba):
                    plt.text(i, prob, f'{prob:.1%}', ha='center', va='bottom')
                st.pyplot(fig)
                plt.close()
                
                # Show credibility indicators
                st.write("---")
                st.subheader("🔍 Credibility Indicators")
                
                # Calculate credibility score based on metadata
                credibility_score = 0
                indicators = []
                
                # Source check
                if st.session_state.article_source:
                    credibility_score += 1
                    indicators.append(("✅ Source provided", "green"))
                else:
                    indicators.append(("❌ No source provided", "red"))
                
                # Author check
                if st.session_state.article_author:
                    credibility_score += 1
                    indicators.append(("✅ Author identified", "green"))
                else:
                    indicators.append(("❌ Anonymous content", "red"))
                
                # Date check
                if st.session_state.article_date:
                    credibility_score += 1
                    indicators.append(("✅ Date provided", "green"))
                else:
                    indicators.append(("❌ No date provided", "red"))
                
                # Content length check
                if len(st.session_state.article_content.split()) > 100:
                    credibility_score += 1
                    indicators.append(("✅ Detailed content", "green"))
                else:
                    indicators.append(("❌ Limited content", "red"))
                
                # Display credibility indicators
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Credibility Score", f"{credibility_score}/4")
                
                with col2:
                    for indicator, color in indicators:
                        st.markdown(f"<span style='color: {color}'>{indicator}</span>", unsafe_allow_html=True)

                # Add class meaning explanation
                st.info(f"**Class Interpretation**: The prediction '{display_prediction}' represents '{class_meaning}'")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.write("Debug: Exception details:", type(e).__name__, str(e))
            finally:
                st.session_state.predict_clicked = False
    
    else:
        # Create two columns for better organization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create sliders for feature inputs
            st.write("📊 Adjust Feature Values:")
            input_values = {}
            
            # Group features in tabs if there are many
            if len(X.columns) > 6:
                num_tabs = (len(X.columns) + 5) // 6  # 6 features per tab
                tabs = st.tabs([f"Features {i+1}-{min(i+6, len(X.columns))}" for i in range(0, len(X.columns), 6)])
                
                for tab_idx, tab in enumerate(tabs):
                    with tab:
                        start_idx = tab_idx * 6
                        end_idx = min(start_idx + 6, len(X.columns))
                        
                        for feature in X.columns[start_idx:end_idx]:
                            min_val = float(X[feature].min())
                            max_val = float(X[feature].max())
                            mean_val = float(X[feature].mean())
                            
                            input_values[feature] = st.slider(
                                f"{feature}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                format="%.2f",
                                help=f"Average: {mean_val:.2f}"
                            )
            else:
                # If few features, show all sliders directly
                for feature in X.columns:
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    
                    input_values[feature] = st.slider(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        format="%.2f",
                        help=f"Average: {mean_val:.2f}"
                    )
        
        with col2:
            st.write("🎯 Prediction")
            predict_button = st.button("Predict", use_container_width=True)
            
            if predict_button:  # Only predict when button is pressed
                with st.spinner("Calculating prediction..."):
                    # Create input array
                    input_array = np.array([input_values[f] for f in X.columns]).reshape(1, -1)
                    
                    # Use the scaler from session state, or handle the case when it's not available
                    if 'scaler' in st.session_state:
                        input_scaled = st.session_state['scaler'].transform(input_array)
                    else:
                        st.error("Model scaler not found. Please retrain the model first.")
                        st.stop()
                    
                    # Make prediction
                    prediction = model.predict(input_scaled)[0]
                    prediction_proba = model.predict_proba(input_scaled)[0]
                    
                    # Convert prediction back to original label if label encoder exists
                    display_prediction = prediction
                    if 'label_encoder' in st.session_state:
                        display_prediction = st.session_state['label_encoder'].inverse_transform([prediction])[0]
                    
                    # Get class meaning if available
                    class_meaning = ""
                    if 'class_names' in st.session_state and prediction in st.session_state['class_names']:
                        class_meaning = st.session_state['class_names'][prediction]
                    
                    # Show prediction with confidence
                    max_prob = max(prediction_proba)
                    st.metric(
                        "Predicted Class",
                        f"{display_prediction} ({class_meaning})" if class_meaning else f"{display_prediction}",
                        f"Confidence: {max_prob:.1%}"
                    )
                    
                    # Add class meaning explanation
                    st.info(f"**Class Interpretation**: The prediction '{display_prediction}' represents '{class_meaning}'")
                    
                    # Probability distribution with class meanings
                    fig = plt.figure(figsize=(8, 3))
                    
                    # Get x-labels with class meanings
                    x_labels = []
                    for i in range(len(prediction_proba)):
                        if 'class_names' in st.session_state and i in st.session_state['class_names']:
                            x_labels.append(f"{i}\n({st.session_state['class_names'][i]})")
                        else:
                            x_labels.append(f"{i}")
                            
                    plt.bar(range(len(prediction_proba)), prediction_proba, 
                           color=['#ff9999' if i != prediction else '#99ff99' for i in range(len(prediction_proba))])
                    plt.title("Prediction Confidence")
                    plt.xlabel("Class")
                    plt.ylabel("Probability")
                    plt.xticks(range(len(prediction_proba)), x_labels)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Calculate feature importance
                    with st.spinner("Calculating feature importance..."):
                        try:
                            # Limit the number of background samples for better performance
                            background_samples = min(50, len(X_train))
                            # Use fewer samples for high-dimensional data
                            if X_train.shape[1] > 100:
                                n_shap_samples = 50
                            else:
                                n_shap_samples = 100
                                
                            # Create explainer with smaller sample size
                            background_data = shap.sample(X_train, background_samples)
                            explainer = shap.KernelExplainer(model.predict_proba, background_data)
                            
                            # Get SHAP values with reduced computation for large feature sets
                            shap_values = explainer.shap_values(input_scaled, nsamples=n_shap_samples)
                            
                            # Process SHAP values based on their structure
                            if isinstance(shap_values, list):
                                all_shap = np.array(shap_values)
                                feature_importance = np.mean(np.abs(all_shap), axis=0)[0]
                            else:
                                feature_importance = np.abs(shap_values[0])
                            
                            if len(feature_importance) == len(X.columns):
                                feature_importance_df = pd.DataFrame({
                                    'Feature': X.columns,
                                    'Impact': feature_importance
                                }).sort_values('Impact', ascending=False)
                                
                                st.write("🔍 Top Influential Features:")
                                top_features = feature_importance_df.head(3)
                                total_impact = feature_importance_df['Impact'].sum()
                                
                                for _, row in top_features.iterrows():
                                    impact_percentage = (row['Impact'] / total_impact) * 100
                                    st.info(f"**{row['Feature']}**: {impact_percentage:.1f}% relative impact")
                                
                                top_5 = feature_importance_df.head(5)
                                fig = plt.figure(figsize=(8, 3))
                                plt.bar(range(len(top_5)), 
                                       top_5['Impact'],
                                       color='skyblue')
                                plt.xticks(range(len(top_5)), 
                                         top_5['Feature'], 
                                         rotation=45, ha='right')
                                plt.title("Top 5 Feature Importance")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.warning("Feature importance calculation produced mismatched dimensions.")
                                st.info(f"Your model has made a prediction with {max(prediction_proba):.1%} confidence.")
                        except Exception as e:
                            st.warning(f"Unable to calculate detailed feature importance: {type(e).__name__}")
                            st.info(f"This can happen with complex datasets. Your model has made a prediction with {max(prediction_proba):.1%} confidence.")

else:  # Batch Predict mode
    st.subheader("Batch Predictions")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_samples = st.slider("Number of samples", 5, 50, 10, 
                              help="Select how many random samples to test")
        
        if st.button("Generate Predictions", use_container_width=True):
            with st.spinner("Processing batch predictions..."):
                # Get random samples
                sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
                X_samples = X_test[sample_indices] if isinstance(X_test, np.ndarray) else X_test.iloc[sample_indices].values
                y_true = y_test[sample_indices] if isinstance(y_test, np.ndarray) else y_test.iloc[sample_indices].values
                
                # Make predictions
                y_pred = model.predict(X_samples)
                y_pred_proba = model.predict_proba(X_samples)
                
                # Calculate metrics
                accuracy = (y_true == y_pred).mean()
                st.metric("Batch Accuracy", f"{accuracy:.1%}")
                
                # Create and style results DataFrame
                results_df = pd.DataFrame({
                    'True Class': y_true,
                    'Predicted': y_pred,
                    'Confidence': [f"{max(proba):.1%}" for proba in y_pred_proba],
                    'Correct': y_true == y_pred
                })
                
                # Add class meaning columns
                if 'class_names' in st.session_state:
                    results_df['True Class Meaning'] = results_df['True Class'].apply(
                        lambda x: st.session_state['class_names'].get(x, f"Class {x}")
                    )
                    results_df['Predicted Meaning'] = results_df['Predicted'].apply(
                        lambda x: st.session_state['class_names'].get(x, f"Class {x}")
                    )
                    
                    # Reorder columns
                    results_df = results_df[['True Class', 'True Class Meaning', 'Predicted', 'Predicted Meaning', 'Confidence', 'Correct']]
                
                # Style the DataFrame
                def color_correct(val):
                    return 'background-color: #99ff99' if val else 'background-color: #ff9999'
                
                styled_df = results_df.style.apply(lambda x: ['background-color: #f0f2f6' if x.name % 2 == 0 else '' for _ in x], axis=1)\
                                           .apply(lambda x: [color_correct(v) if i == 3 else '' for i, v in enumerate(x)], axis=1)
                
                st.write("### Results")
                st.dataframe(styled_df, use_container_width=True)
    
    with col2:
        # Show confusion matrix
        if 'y_pred' in locals():
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            
            # Create labels with class meanings if available
            if 'class_names' in st.session_state:
                labels = [f"{i}\n({st.session_state['class_names'].get(i, f'Class {i}')})" 
                          for i in range(len(np.unique(np.concatenate([y_true, y_pred]))))]
            else:
                labels = range(len(np.unique(np.concatenate([y_true, y_pred]))))
                
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            st.pyplot(fig)
            plt.close()

st.write("---")
st.markdown("""
### 💡 Tips:
- **Quick Predict**: Use sliders to test specific scenarios
- **Batch Predict**: Test model performance on multiple samples
- Higher confidence means the model is more certain about its prediction
""")