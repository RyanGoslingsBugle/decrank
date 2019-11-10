import torch
import torch.utils.data as utils
import torch.nn as nn


class TorchModels:
    def __init__(self):
        self.models = {
        }

    @staticmethod
    def create_loader(x):
        tensor_x = torch.stack([torch.Tensor(i) for i in x])
        dataset = utils.TensorDataset(tensor_x)
        return utils.DataLoader(dataset)
