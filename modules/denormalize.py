import numpy as np
import torch
import math
#------------------Denormalization---------------------------------------------
def denormalize_bin(tensor):
    tr = torch.clamp(tensor, -1., 1.)  # Clamp the values between -1 and 1
    tr = tr.add(1).div(2)  # Shift to [0, 1]
    return tr

def denormalize_tr(tensor):
    tr = torch.clamp(tensor, -1., 1.)  # Clamp the values between -1 and 1
    tr = tr.add(1).div(2).mul(255)  # Shift to [0, 1] and scale to [0, 255]
    tr = tr.byte()  # Convert the tensor to uint8
    return tr

def denormalize_ar(tensor):
    tr = torch.clamp(tensor, -1., 1.)  # Clamp the values between -1 and 1
    tr = tr.add(1).div(2).mul(255)  # Shift to [0, 1] and scale to [0, 255]
    tr = tr.byte()  # Convert the tensor to uint8
    arr = tr.permute(0, 2, 3, 1).cpu().detach().numpy()  # Convert to (N, H, W, C) and numpy array
    return arr