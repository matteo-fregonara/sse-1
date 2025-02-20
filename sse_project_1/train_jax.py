import os

# Force CPU usage for JAX
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from shared_dataset import get_dataset
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
key = jax.random.PRNGKey(42)

# Load the shared dataset
X, y = get_dataset()
y_jax = y.reshape(-1, 1)  # Reshape for JAX

# Print which device JAX is using
print(f"JAX is using: {jax.devices()}")


# Flax Model Definition
class SimpleNet(nn.Module):
    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Dense(features=64, kernel_init=nn.initializers.glorot_uniform())(x)
        x = nn.relu(x)
        x = nn.Dense(features=32, kernel_init=nn.initializers.glorot_uniform())(x)
        x = nn.relu(x)
        x = nn.Dense(features=1, kernel_init=nn.initializers.glorot_uniform())(x)
        x = nn.sigmoid(x)
        return x


# Binary cross entropy loss
def binary_cross_entropy(logits, labels):
    return -jnp.mean(labels * jnp.log(logits + 1e-10) +
                     (1 - labels) * jnp.log(1 - logits + 1e-10))


# Accuracy calculation
def compute_accuracy(logits, labels):
    preds = (logits > 0.5).astype(jnp.float32)
    return jnp.mean(preds == labels)


# Training step - limited by CPU JIT
@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_x)
        loss = binary_cross_entropy(logits, batch_y)
        accuracy = compute_accuracy(logits, batch_y)
        return loss, accuracy

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, accuracy), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss, accuracy


# Training function
def train_jax():
    model = SimpleNet()

    # Initialize model
    init_rng = jax.random.PRNGKey(42)
    params = model.init(init_rng, jnp.ones([1, 10]))['params']

    # Create optimizer (equivalent to Adam in TF and PyTorch)
    optimizer = optax.adam(
        learning_rate=0.001,
        b1=0.9,
        b2=0.999,
        eps=1e-7
    )

    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

    # Training loop
    batch_size = 32
    num_batches = len(X) // batch_size
    X_jax = jnp.array(X)
    y_jax = jnp.array(y_jax)

    for epoch in range(10):
        # Shuffle data
        shuffle_idx = jax.random.permutation(key, len(X))
        key = jax.random.fold_in(key, epoch)
        shuffled_x = X_jax[shuffle_idx]
        shuffled_y = y_jax[shuffle_idx]

        # Train over batches
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = shuffled_x[start_idx:end_idx]
            batch_y = shuffled_y[start_idx:end_idx]

            state, loss, accuracy = train_step(state, batch_x, batch_y)
            epoch_loss += loss / num_batches
            epoch_accuracy += accuracy / num_batches

        print(f'Epoch {epoch + 1}/10 - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}')

    return state


if __name__ == "__main__":
    print("Training JAX/Flax model on CPU...")
    jax_model = train_jax()