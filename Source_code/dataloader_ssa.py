import torch
from torch.utils.data import random_split
from monai.transforms import Compose, ScaleIntensity, RandCropByPosNegLabel, RandRotate, RandFlip, ToTensor
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, CacheDataset, decollate_batch, Dataset
from monai.apps import DecathlonDataset
from monai.handlers.utils import from_engine
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from models.cascadeNet import CascadeUNet
import os
from glob import glob
from tqdm import tqdm
from torchsummary import summary
from datetime import datetime


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """Map BraTS labels to [TC, WT, ET] channels."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


def dataload_ssa(tr_batchsize=1, val_split=0.125, train_transform=None, val_transform=None):
    if train_transform is None:
        train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                RandSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 128], random_size=False),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ]
        )

    if val_transform is None:
        val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                EnsureTyped(keys=["image", "label"]),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                ),
                RandSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 128], random_size=False),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    test_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            lambda x: x,
        ]
    )

    if os.name == 'nt':
        data_root = 'D:/wjh/archive'  
    else:
        data_root = '/home/pyy/archive'

    testimgs = sorted(glob(f'{data_root}/BraTS-SSA-2023/images/*.gz'))
    testlabels = sorted(glob(f'{data_root}/BraTS-SSA-2023/labels/*.gz'))

    test_data_dict = [{'image': image, 'label': label} for image, label in zip(testimgs, testlabels)]
    test_ds = Dataset(data=test_data_dict, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=12, pin_memory=False)

    return test_loader