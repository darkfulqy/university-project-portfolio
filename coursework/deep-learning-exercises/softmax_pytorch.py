import torch
def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    scores_tensor=torch.tensor(scores,dtype=torch.float32)
    softmax_tensor=torch.softmax(scores_tensor,dim=0)
    return softmax_tensor.tolist()

score=[1,2, 3]
print(softmax(score))