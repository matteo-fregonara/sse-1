import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic dataset
X = np.random.randn(1000, 10).astype(np.float32)
y = (np.sum(X[:, :3], axis=1) > 0).astype(np.float32)

# PyTorch Dataset
class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

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
    model = SimpleNet()
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
    print("Training PyTorch model...")
    pytorch_model = train_pytorch()