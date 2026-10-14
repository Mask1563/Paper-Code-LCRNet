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


def dataload(tr_batchsize=1, val_split=0.125, train_transform=None, val_transform=None):
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
            RandSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 128], random_size=False),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            lambda x: x,
        ]
    )

    if os.path.exists('/home/pyy/ori_data'):
        data_root = '/home/pyy/ori_data'
    else:
        data_root = 'D:/wjh/ori_data'  
    trainimgs = sorted(glob(f'{data_root}/train/images/*.gz'))
    trainlabels = sorted(glob(f'{data_root}/train/labels/*.gz'))

    testimgs = sorted(glob(f'{data_root}/test/images/*.gz'))
    testlabels = sorted(glob(f'{data_root}/test/labels/*.gz'))

    train_data_dict = [{'image': image, 'label': label} for image, label in zip(trainimgs, trainlabels)]
    test_data_dict = [{'image': image, 'label': label} for image, label in zip(testimgs, testlabels)]

    original_ds = Dataset(data=train_data_dict)
    train_size = int((1 - val_split) * len(train_data_dict))
    val_size = len(train_data_dict) - train_size
    
    torch.manual_seed(42)
    train_indices, val_indices = torch.utils.data.random_split(range(len(train_data_dict)), [train_size, val_size])

    train_subset = torch.utils.data.Subset(original_ds, train_indices.indices)
    val_subset = torch.utils.data.Subset(original_ds, val_indices.indices)

    train_ds = Dataset(data=train_subset, transform=train_transform)
    val_ds = Dataset(data=val_subset, transform=val_transform)
    test_ds = Dataset(data=test_data_dict, transform=test_transform)
    train_loader = DataLoader(train_ds, batch_size=tr_batchsize, shuffle=True, num_workers=12, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=12, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=12, pin_memory=False)

    return train_loader, val_loader,test_loader