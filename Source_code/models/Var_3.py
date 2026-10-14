from .coarseNet_3d import CoarseNet3D
from .fineNet_new import FineNet_New
import torch.nn as nn
import torch
import torch.nn.functional as F

class Var_3(nn.Module):
    """Ablation variant: 3D CoarseNet + FineNet_New with bbox crop."""
    def __init__(self, 
                 num_out_classes=3, 
                 input_channels=4, 
                 coarse_base_channels=8,
                 downsample_factor=4,
                 testing=False):
        super(Var_3, self).__init__()
        
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
        
        wt_mask_binary = (wt_mask > 0.5).float()

        cropped_image, cropped_wt, padding_list = self.cut_dense_region_bbox(
            wt_mask_binary, wt_mask, inputs, crop_size=128
        )

        combined_input = torch.cat((cropped_image, cropped_wt), dim=1)

        fine_output = self.fine_unet(combined_input)

        restore_output = self.restore_tensor(fine_output, padding_list)
        
        return wt_mask, restore_output

    def cut_dense_region_bbox(self, mask_binary, mask_prob, image, crop_size):
        """
        Args:
            mask_binary: Binary mask for bbox calculation (B, 1, H, W, D)
            mask_prob: Probability mask for cropping (B, 1, H, W, D)
            image: Input image (B, C, H, W, D)
            crop_size: Size of crop region
        """
        B, _, H, W, D = mask_binary.shape
        cut_mask_list = []
        cut_image_list = []
        padding_params_list = []

        for batch in range(B):
            nonzero = torch.nonzero(mask_binary[batch, 0] > 0.5)
            
            if nonzero.shape[0] == 0:
                center_i, center_j, center_k = H // 2, W // 2, D // 2
            else:
                min_indices = nonzero.min(dim=0)[0]  # [h, w, d]
                max_indices = nonzero.max(dim=0)[0]

                center_i = (min_indices[0] + max_indices[0]) // 2
                center_j = (min_indices[1] + max_indices[1]) // 2
                center_k = (min_indices[2] + max_indices[2]) // 2

            start_i = max(0, center_i - crop_size // 2)
            start_j = max(0, center_j - crop_size // 2)
            start_k = max(0, center_k - crop_size // 2)
            end_i = min(H, start_i + crop_size)
            end_j = min(W, start_j + crop_size)
            end_k = min(D, start_k + crop_size)

            if end_i - start_i < crop_size:
                start_i = max(0, end_i - crop_size)
                end_i = min(H, start_i + crop_size)
            if end_j - start_j < crop_size:
                start_j = max(0, end_j - crop_size)
                end_j = min(W, start_j + crop_size)
            if end_k - start_k < crop_size:
                start_k = max(0, end_k - crop_size)
                end_k = min(D, start_k + crop_size)

            cut_mask = mask_prob[batch:batch+1, :, start_i:end_i, start_j:end_j, start_k:end_k]
            cut_image = image[batch:batch+1, :, start_i:end_i, start_j:end_j, start_k:end_k]

            padding_params = (
                max(0, D - end_k), start_k,
                max(0, W - end_j), start_j,
                max(0, H - end_i), start_i,
            )
            
            cut_mask_list.append(cut_mask)
            cut_image_list.append(cut_image)
            padding_params_list.append(padding_params)
        
        cut_mask = torch.cat(cut_mask_list, dim=0)
        cut_image = torch.cat(cut_image_list, dim=0)
        
        return cut_image, cut_mask, padding_params_list

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
