import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def create_model(input_dim, num_classes, hidden_activation='relu', output_activation='softmax', 
                hidden_layers=[64, 32], dropout_rate=0.2, use_batch_norm=False):
    """
    Create a neural network with specified activation functions and architecture.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        hidden_activation: Activation function for hidden layers
        output_activation: Activation function for output layer
        hidden_layers: List of neurons in each hidden layer
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    """
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation=hidden_activation))
    if use_batch_norm:
        model.add(BatchNormalization())
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Additional hidden layers
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation=hidden_activation))
        if use_batch_norm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(num_classes, activation=output_activation))
    
    return model

def train_and_evaluate(name, X, y, hidden_activation='relu', output_activation='softmax', 
                      hidden_layers=[64, 32], dropout_rate=0.2, use_batch_norm=False):
    """
    Train and evaluate a neural network with the specified configuration.
    """
    # Convert to one-hot encoding if needed
    if output_activation == 'softmax':
        y = tf.keras.utils.to_categorical(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and compile the model
    model = create_model(
        input_dim=X.shape[1],
        num_classes=y.shape[1] if output_activation == 'softmax' else 1,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )
    
    # Compile with appropriate loss function
    loss = 'categorical_crossentropy' if output_activation == 'softmax' else 'binary_crossentropy'
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    print(f"\nTraining model for {name} dataset:")
    print("=" * 50)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy for {name}: {accuracy:.4f}")
    return model, history

# Load and preprocess datasets
datasets = [
    (load_iris(), "Iris", {
        'hidden_activation': 'relu',
        'output_activation': 'softmax',
        'hidden_layers': [16, 8],
        'dropout_rate': 0.2,
        'use_batch_norm': False
    }),
    (load_breast_cancer(), "Breast Cancer", {
        'hidden_activation': 'elu',
        'output_activation': 'sigmoid',
        'hidden_layers': [32, 16],
        'dropout_rate': 0.3,
        'use_batch_norm': True
    }),
    (load_wine(), "Wine", {
        'hidden_activation': 'selu',
        'output_activation': 'softmax',
        'hidden_layers': [24, 12],
        'dropout_rate': 0.25,
        'use_batch_norm': True
    })
]

# Train models for each dataset
for data, name, params in datasets:
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data.data)
    
    # Train and evaluate the model
    model, history = train_and_evaluate(
        name=name,
        X=X_scaled,
        y=data.target,
        **params
    )
    
    print(f"\nModel configuration for {name}:")
    print("-" * 40)
    for param, value in params.items():
        print(f"{param}: {value}")
    print("-" * 40) 