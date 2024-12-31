import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.rmse_score import rmse_coeff, accuracy_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    rmse_score = 0
    accuracy_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
    # iterate over the validation set
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask, label = batch['image'], batch['mask'], batch['label']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask = mask.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred, label_pred = net(image)

            # compute rmse score
            rmse_score += rmse_coeff(mask_pred, mask)
            accuracy_score += accuracy_coeff(label_pred, label)

    net.train()
    return rmse_score / max(num_val_batches, 1), accuracy_score / max(num_val_batches, 1)
