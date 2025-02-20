import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic dataset
X = np.random.randn(1000, 10).astype(np.float32)
y = (np.sum(X[:, :3], axis=1) > 0).astype(np.float32)

# Prepare TensorFlow dataset
X_tf = X
y_tf = y.reshape(-1, 1)

# TensorFlow Model
def create_tf_model():
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,),
                    kernel_initializer=initializer),
        layers.Dense(32, activation='relu',
                    kernel_initializer=initializer),
        layers.Dense(1, activation='sigmoid',
                    kernel_initializer=initializer)
    ])
    return model

# Training function
def train_tensorflow():
    # Create and compile model
    model = create_tf_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        X_tf, y_tf,
        epochs=10,
        batch_size=32,
        verbose=1,
        shuffle=True
    )

    return model

if __name__ == "__main__":
    print("Training TensorFlow model...")
    tensorflow_model = train_tensorflow()