import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    
    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term
    
    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    # Your code here using PyTorch built-ins:
    # - torch.matmul() for linear combination
    # - torch.sigmoid() for activation
    # - torch.nn.functional.mse_loss() for MSE
    features_tensor=torch.tensor(features,dtype=torch.float32)
    weights_tensor=torch.tensor(weights,dtype=torch.float32)
    bias_tensor=torch.tensor(bias,dtype=torch.float32)
    linear_combination=torch.matmul(features_tensor,weights_tensor)+bias_tensor
    predicted_probabilities=torch.sigmoid(linear_combination)
    labels_tensor=torch.tensor(labels,dtype=torch.float32)
    mse=torch.nn.functional.mse_loss(predicted_probabilities,labels_tensor)
    return predicted_probabilities.tolist(), round(mse.item(),4)



print(single_neuron_model([[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1))