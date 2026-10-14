from .UNet3D import UNet3D
from .fineNet import FineNet
from .fuse2d import Fuse2d
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gc

class CascadeUNet(nn.Module):
    def __init__(self, num_out_classes=2, input_channels=1, init_coarse_channels=8, init_fine_channels=8, testing=False):
        super(CascadeUNet, self).__init__()
        self.coarse_unet = UNet3D(num_out_classes=input_channels, input_channels=input_channels, init_feat_channels=init_coarse_channels)
        self.fine_unet = FineNet()
        
    def forward(self, inputs):
        wt_mask = self.coarse_unet(inputs)
        wt_mask = torch.where(wt_mask > 0.5, torch.tensor(1.0), torch.tensor(1e-3))
        wt_mask = wt_mask.sum(dim=1, keepdim=True)
        cropped_image, cropped_wt, padding_list = self.cut_dense_region_centered(wt_mask, inputs, 128)
        combined_input = torch.cat((cropped_image, cropped_wt), dim=1)
        fine_output = self.fine_unet(combined_input)
        restore_output = self.restore_tensor(fine_output, padding_list)
        
        return wt_mask, restore_output
    
    def find_centroid(self, tensor):
        non_zero_coords = torch.nonzero(tensor)
        centroid = non_zero_coords.float().mean(dim=0).long()
        return centroid

    def cut_dense_region_centered(self, tensorA, tensorB, size):
        batch_size = tensorA.shape[0]
        cut_tensorA_list = []
        cut_tensorB_list = []
        padding_params_list = []

        for batch in range(batch_size):
            centroid = self.find_centroid(tensorA[batch, 0, :, :, :])
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

    def restore_tensor(self, cut_tensor, padding_params_list):
        restored_tensor_list = []
        for batch in range(cut_tensor.shape[0]):
            restored_channels = []
            for channel in range(cut_tensor.shape[1]):
                padding_params = padding_params_list[batch]
                # 注意填充参数的顺序 (left, right, top, bottom, front, back)
                restored_channel = F.pad(cut_tensor[batch, channel], (padding_params[1], padding_params[0], padding_params[3], padding_params[2], padding_params[5], padding_params[4]))
                restored_channels.append(restored_channel.unsqueeze(0))
            restored_tensor = torch.cat(restored_channels, dim=0).unsqueeze(0)
            restored_tensor_list.append(restored_tensor)

        restored_tensor = torch.cat(restored_tensor_list, dim=0)
        return restored_tensor
    
    def show(self, data):
        data = data.cpu().numpy()
        batch_index = 0
        batch_data = data[batch_index]
        depth_index = 40
        modality_slices = [batch_data[i, :, :, depth_index] for i in range(batch_data.shape[0])]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, slice in enumerate(modality_slices):
            axes[i].imshow(slice, cmap="gray")
            axes[i].set_title(f'Modality {i+1}')
            axes[i].axis('off')
        plt.suptitle("Middle Slices for Each Modality")
        plt.show()