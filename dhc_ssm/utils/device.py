"""
Centralized device management utility
"""
import torch

def ensure_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    if tensor.device.type != device:
        return tensor.to(device)
    return tensor

class DeviceManager:
    def __init__(self, device: str = 'cpu'):
        self.device = device
    
    def to_device(self, *tensors):
        return [ensure_device(t, self.device) for t in tensors]
