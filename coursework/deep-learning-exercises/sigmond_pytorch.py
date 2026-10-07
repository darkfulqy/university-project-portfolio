import torch
def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    z = torch.tensor(z, dtype=torch.float32)
    result = 1/(1+torch.exp(-z))
    return round(result.item(), 4)
    


z=torch.tensor(0.5,dtype=torch.float32)
print(sigmoid(z))

