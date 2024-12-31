import torch
from torch import Tensor

def rmse_coeff(input: Tensor, target: Tensor):
    # Average of Rmse coefficient for all batches, or for a single mask
    assert input.size() == target.size()

    mse = (target - input)**2 
    rmse = torch.sqrt(mse.mean())

    return rmse


def accuracy_coeff(input: Tensor, target: Tensor):
    # Average of Rmse coefficient for all batches, or for a single mask
    assert input.size()[0] == target.size()[0]

    target_label = torch.argmax(input, dim=1)
    
    accuracy = torch.sum(target_label == target) / input.size()[0] 

    return accuracy

