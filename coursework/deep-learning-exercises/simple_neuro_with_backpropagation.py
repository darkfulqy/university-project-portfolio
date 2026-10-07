import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    weights = torch.tensor(initial_weights, dtype=torch.float32, requires_grad=True)
    bias = torch.tensor(initial_bias, dtype=torch.float32, requires_grad=True)
    batch_size = features.size(0)
    mse_values = []
    for epoch in range(epochs):
        z = torch.matmul(features,weights) + bias
        predictions = torch.sigmoid(z)
        mse=torch.nn.functional.mse_loss(predictions,labels)
        mse_values.append(round(mse.item(),4))
        dz=2*(predictions-labels)*predictions*(1-predictions)/batch_size
        gradient=torch.matmul(features.T,dz)
        weights=weights-learning_rate*gradient
        bias=bias-learning_rate*dz.sum()
    return [round(w,4) for w in weights.tolist()], round(bias.item(),4), [round(m,4) for m in mse_values]
    pass

print(train_neuron(features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2))