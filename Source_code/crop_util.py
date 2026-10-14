import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gc

def find_centroid(tensor):
    non_zero_coords = torch.nonzero(tensor)
    centroid = non_zero_coords.float().mean(dim=0).long()
    return centroid

def cut_dense_region_centered(tensorA, tensorB, size):
    batch_size = tensorA.shape[0]
    cut_tensorA_list = []
    cut_tensorB_list = []
    padding_params_list = []

    for batch in range(batch_size):
        centroid = find_centroid(tensorA[batch, 0, :, :, :])
        center_i, center_j, center_k = centroid[0].item(), centroid[1].item(), centroid[2].item()
        
        start_i = max(0, center_i - size // 2)
        start_j = max(0, center_j - size // 2)
        start_k = max(0, center_k - size // 2)
        end_i = min(tensorA.shape[2], start_i + size)
        end_j = min(tensorA.shape[3], start_j + size)
        end_k = min(tensorA.shape[4], start_k + size)
        
        if end_i - start_i < size:
            start_i = end_i - size
        if end_j - start_j < size:
            start_j = end_j - size
        if end_k - start_k < size:
            start_k = end_k - size

        cut_tensorA = tensorA[batch:batch+1, :, start_i:end_i, start_j:end_j, start_k:end_k]
        cut_tensorB = tensorB[batch:batch+1, :, start_i:end_i, start_j:end_j, start_k:end_k]

        padding_params = (
            max(0, tensorA.shape[4] - end_k), start_k,
            max(0, tensorA.shape[3] - end_j), start_j,
            max(0, tensorA.shape[2] - end_i), start_i,
        )

        cut_tensorA_list.append(cut_tensorA)
        cut_tensorB_list.append(cut_tensorB)
        padding_params_list.append(padding_params)

    cut_tensorA = torch.cat(cut_tensorA_list, dim=0)
    cut_tensorB = torch.cat(cut_tensorB_list, dim=0)

    return cut_tensorA, cut_tensorB, padding_params_list

def restore_tensor(cut_tensor, padding_params_list):
    restored_tensor_list = []
    for batch in range(cut_tensor.shape[0]):
        restored_channels = []
        for channel in range(cut_tensor.shape[1]):
            padding_params = padding_params_list[batch]
            # F.pad uses (left, right, top, bottom, front, back).
            restored_channel = F.pad(cut_tensor[batch, channel], (padding_params[1], padding_params[0], padding_params[3], padding_params[2], padding_params[5], padding_params[4]))
            restored_channels.append(restored_channel.unsqueeze(0))
        restored_tensor = torch.cat(restored_channels, dim=0).unsqueeze(0)
        restored_tensor_list.append(restored_tensor)

    restored_tensor = torch.cat(restored_tensor_list, dim=0)
    return restored_tensor