from .coarseNet_3d import CoarseNet3D
from .fineNet_new import FineNet_New
import torch.nn as nn
import torch
import torch.nn.functional as F

class Var_4(nn.Module):
    """Ablation variant: 3D CoarseNet + FineNet_New with centroid crop."""
    def __init__(self, 
                 num_out_classes=3, 
                 input_channels=4, 
                 coarse_base_channels=8,
                 downsample_factor=4,
                 testing=False):
        super(Var_4, self).__init__()
        
        self.downsample_factor = downsample_factor
        
        self.coarse_unet = CoarseNet3D(in_channels=input_channels, base_channels=coarse_base_channels)

        self.fine_unet = FineNet_New(
            num_classes=num_out_classes, 
            use_deep_supervision=False
        )

    def forward(self, inputs):
        B, C, H, W, D = inputs.shape
        
        downsampled = F.interpolate(inputs, 
                                     scale_factor=1/self.downsample_factor, 
                                     mode='trilinear', 
                                     align_corners=False)

        d_h, d_w, d_d = downsampled.shape[2:]
        pad_h = (8 - d_h % 8) % 8
        pad_w = (8 - d_w % 8) % 8
        pad_d = (8 - d_d % 8) % 8
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            downsampled = F.pad(downsampled, (0, pad_d, 0, pad_w, 0, pad_h))
        
        wt_mask_low = self.coarse_unet(downsampled)
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            wt_mask_low = wt_mask_low[:, :, :d_h, :d_w, :d_d]
        
        wt_mask = F.interpolate(wt_mask_low, 
                                size=(H, W, D), 
                                mode='trilinear', 
                                align_corners=False)
        

        cropped_wt, cropped_image, padding_list = self.cut_dense_region_centered(
            wt_mask, inputs, size=128
        )

        combined_input = torch.cat((cropped_image, cropped_wt), dim=1)

        fine_output = self.fine_unet(combined_input)

        restore_output = self.restore_tensor(fine_output, padding_list)
        
        return wt_mask, restore_output

    def find_centroid(self, tensor):
        non_zero_coords = torch.nonzero(tensor)
        if non_zero_coords.shape[0] == 0:
            return torch.tensor([tensor.shape[0]//2, tensor.shape[1]//2, tensor.shape[2]//2])
        centroid = non_zero_coords.float().mean(dim=0).long()
        return centroid

    def cut_dense_region_centered(self, tensorA, tensorB, size):
        batch_size = tensorA.shape[0]
        cut_tensorA_list = []
        cut_tensorB_list = []
        padding_params_list = []

        for batch in range(batch_size):
            mask = (tensorA[batch, 0, :, :, :] > 0.5).float()
            centroid = self.find_centroid(mask)
            center_i, center_j, center_k = centroid[0].item(), centroid[1].item(), centroid[2].item()

            start_i = max(0, center_i - size // 2)
            start_j = max(0, center_j - size // 2)
            start_k = max(0, center_k - size // 2)
            end_i = min(tensorA.shape[2], start_i + size)
            end_j = min(tensorA.shape[3], start_j + size)
            end_k = min(tensorA.shape[4], start_k + size)

            if end_i - start_i < size:
                start_i = max(0, end_i - size)
                end_i = start_i + size # Ensure exact size
            if end_j - start_j < size:
                start_j = max(0, end_j - size)
                end_j = start_j + size
            if end_k - start_k < size:
                start_k = max(0, end_k - size)
                end_k = start_k + size

            cut_tensorA = tensorA[batch:batch + 1, :, start_i:end_i, start_j:end_j, start_k:end_k]
            cut_tensorB = tensorB[batch:batch + 1, :, start_i:end_i, start_j:end_j, start_k:end_k]

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
                restored_channel = F.pad(cut_tensor[batch, channel],
                                         (padding_params[1], padding_params[0], padding_params[3], padding_params[2],
                                          padding_params[5], padding_params[4]))
                restored_channels.append(restored_channel.unsqueeze(0))
            restored_tensor = torch.cat(restored_channels, dim=0).unsqueeze(0)
            restored_tensor_list.append(restored_tensor)

        restored_tensor = torch.cat(restored_tensor_list, dim=0)
        return restored_tensor