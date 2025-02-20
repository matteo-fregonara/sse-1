# shared_dataset.py
import numpy as np


def get_dataset():
    # Set seed for reproducibility
    np.random.seed(42)

    # Generate synthetic dataset
    X = np.random.randn(1000, 10).astype(np.float32)
    y = (np.sum(X[:, :3], axis=1) > 0).astype(np.float32)

    return X, y


# train_pytorch.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from shared_dataset import get_dataset

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cpu')

# Set seed for reproducibility
torch.manual_seed(42)

# Load the shared dataset
X, y = get_dataset()


# PyTorch Dataset
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# PyTorch Model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Using same initialization as TensorFlow
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        # Initialize weights with Glorot/Xavier uniform
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Training function
def train_pytorch():
    # Create data loaders
    dataset = SimpleDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model and optimizer
    model = SimpleNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),  # TensorFlow defaults
        eps=1e-7  # TensorFlow default
    )

    # Training loop
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            # Compute accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print metrics in TensorFlow style
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        print(f'Epoch {epoch + 1}/10 - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f}')

    return model


if __name__ == "__main__":
    print("Training PyTorch model on CPU...")
    pytorch_model = train_pytorch()

# train_tensorflow.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from shared_dataset import get_dataset

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the shared dataset
X, y = get_dataset()
y_tf = y.reshape(-1, 1)  # Reshape for TensorFlow


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

    # Verify we're using CPU
    print(f"TensorFlow is using: {tf.config.get_visible_devices()}")

    # Train model
    history = model.fit(
        X, y_tf,
        epochs=10,
        batch_size=32,
        verbose=1,
        shuffle=True
    )

    return model


if __name__ == "__main__":
    print("Training TensorFlow model on CPU...")
    tensorflow_model = train_tensorflow()