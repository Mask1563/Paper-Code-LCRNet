"""Cascade model with 3D coarse stage and ROI crop."""
from .coarseNet_3d import CoarseNet3D, CoarseNet3D_Tiny
from .fineNet_new import FineNet_New
import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class CascadeBoth_New(nn.Module):
    """Two-stage cascade with 3D coarse stage and bbox-based ROI crop."""
    def __init__(self, 
                 num_out_classes=3, 
                 input_channels=4, 
                 coarse_base_channels=8,
                 use_tiny_coarse=False,
                 use_deep_supervision=True,
                 downsample_factor=4,
                 testing=False):
        super(CascadeBoth_New, self).__init__()
        
        self.downsample_factor = downsample_factor
        self.use_deep_supervision = use_deep_supervision and not testing

        if use_tiny_coarse:
            self.coarse_unet = CoarseNet3D_Tiny(in_channels=input_channels, base_channels=coarse_base_channels//2)
        else:
            self.coarse_unet = CoarseNet3D(in_channels=input_channels, base_channels=coarse_base_channels)

        self.fine_unet = FineNet_New(num_classes=num_out_classes, use_deep_supervision=use_deep_supervision)
        
    def forward(self, inputs):
        """
        Args:
            inputs: (B, C, H, W, D) - Original resolution input (e.g., 4, 160, 160, 128)
        Returns:
            wt_mask: (B, 1, H, W, D) - Coarse whole tumor mask at original resolution
            restore_output: (B, num_classes, H, W, D) - Fine segmentation at original resolution
        """
        B, C, H, W, D = inputs.shape
        
        # Stage 1: coarse 3D localization
        downsampled = F.interpolate(inputs,
                                     scale_factor=1/self.downsample_factor,
                                     mode='trilinear',
                                     align_corners=False)

        # CoarseNet3D has 3 stride-2 layers → input spatial dims must be divisible by 8.
        # Full-image inference can produce arbitrary sizes, so pad here and crop back.
        _, _, dH, dW, dD = downsampled.shape
        pad_h = (8 - dH % 8) % 8
        pad_w = (8 - dW % 8) % 8
        pad_d = (8 - dD % 8) % 8
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            downsampled = F.pad(downsampled, (0, pad_d, 0, pad_w, 0, pad_h), mode='replicate')

        wt_mask_low = self.coarse_unet(downsampled)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            wt_mask_low = wt_mask_low[:, :, :dH, :dW, :dD]

        wt_mask = F.interpolate(wt_mask_low,
                                size=(H, W, D),
                                mode='trilinear',
                                align_corners=False)

        wt_mask_binary = (wt_mask > 0.5).float()

        # Stage 2: ROI crop + fine segmentation
        cropped_image, cropped_wt, padding_list = self.cut_dense_region_bbox(
            wt_mask_binary, inputs, crop_size=128
        )
        
        # FineNet_New has 5 stride-2 layers (pe0..pe4) → input must be divisible by 32.
        _, _, cH, cW, cD = cropped_image.shape
        pad_h2 = (32 - cH % 32) % 32
        pad_w2 = (32 - cW % 32) % 32
        pad_d2 = (32 - cD % 32) % 32
        fine_pad = pad_h2 > 0 or pad_w2 > 0 or pad_d2 > 0
        if fine_pad:
            cropped_image = F.pad(cropped_image, (0, pad_d2, 0, pad_w2, 0, pad_h2), mode='replicate')
            cropped_wt = F.pad(cropped_wt, (0, pad_d2, 0, pad_w2, 0, pad_h2), mode='replicate')

        combined_input = torch.cat((cropped_image, cropped_wt), dim=1)

        fine_output = self.fine_unet(combined_input)

        if fine_pad:
            if isinstance(fine_output, list):
                fine_output = [o[:, :, :cH, :cW, :cD] for o in fine_output]
            else:
                fine_output = fine_output[:, :, :cH, :cW, :cD]

        if isinstance(fine_output, list):
            restore_output = [self.restore_tensor(out, padding_list) for out in fine_output]
        else:
            restore_output = self.restore_tensor(fine_output, padding_list)
        
        return wt_mask, restore_output
    
    def cut_dense_region_bbox(self, mask, image, crop_size):
        """Crop around the bbox center of the binary mask."""
        B, _, H, W, D = mask.shape
        cut_mask_list = []
        cut_image_list = []
        padding_params_list = []

        for batch in range(B):
            nonzero = torch.nonzero(mask[batch, 0] > 0.5)
            
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
                end_i = min(H, start_i + crop_size)  # Re-adjust end after fixing start
            if end_j - start_j < crop_size:
                start_j = max(0, end_j - crop_size)
                end_j = min(W, start_j + crop_size)
            if end_k - start_k < crop_size:
                start_k = max(0, end_k - crop_size)
                end_k = min(D, start_k + crop_size)

            cut_mask = mask[batch:batch+1, :, start_i:end_i, start_j:end_j, start_k:end_k]
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
        """Restore cropped tensor using padding parameters."""
        restored_tensor_list = []
        
        for batch in range(cut_tensor.shape[0]):
            restored_channels = []
            padding_params = padding_params_list[batch]
            
            for channel in range(cut_tensor.shape[1]):
                restored_channel = F.pad(
                    cut_tensor[batch, channel], 
                    (padding_params[1], padding_params[0],  # D dimension
                     padding_params[3], padding_params[2],  # W dimension
                     padding_params[5], padding_params[4])  # H dimension
                )
                restored_channels.append(restored_channel.unsqueeze(0))
            
            restored_tensor = torch.cat(restored_channels, dim=0).unsqueeze(0)
            restored_tensor_list.append(restored_tensor)
        
        restored_tensor = torch.cat(restored_tensor_list, dim=0)
        return restored_tensor
    
    def show(self, data):
        data = data.cpu().numpy()
        batch_index = 0
        batch_data = data[batch_index]
        depth_index = batch_data.shape[3] // 2

        modality_slices = [batch_data[i, :, :, depth_index] for i in range(batch_data.shape[0])]

        fig, axes = plt.subplots(1, len(modality_slices), figsize=(5*len(modality_slices), 5))
        if len(modality_slices) == 1:
            axes = [axes]
        
        for i, slice in enumerate(modality_slices):
            axes[i].imshow(slice, cmap="gray")
            axes[i].set_title(f"Channel {i}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('debug_cascade.png')
        plt.close()


if __name__ == "__main__":
    print("Testing CascadeBoth_New...")
    model = CascadeBoth_New(
        num_out_classes=3, 
        input_channels=4,
        coarse_base_channels=8,
        use_tiny_coarse=False,
        use_deep_supervision=True,
        downsample_factor=4
    )
    
    x = torch.randn(2, 4, 160, 160, 128)
    print(f"Input shape: {x.shape}")

    model.train()
    wt_mask, fine_output = model(x)
    
    print(f"Coarse mask shape: {wt_mask.shape}")
    if isinstance(fine_output, list):
        print(f"Fine output (deep supervision): {[o.shape for o in fine_output]}")
    else:
        print(f"Fine output shape: {fine_output.shape}")
    
    coarse_params = sum(p.numel() for p in model.coarse_unet.parameters()) / 1e6
    fine_params = sum(p.numel() for p in model.fine_unet.parameters()) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"\nParameters:")
    print(f"  Coarse UNet: {coarse_params:.2f}M")
    print(f"  Fine UNet: {fine_params:.2f}M")
    print(f"  Total: {total_params:.2f}M")

    model.eval()
    with torch.no_grad():
        wt_mask, fine_output = model(x)
        print(f"\nInference mode - Fine output shape: {fine_output.shape}")
