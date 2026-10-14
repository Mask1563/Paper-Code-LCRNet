import argparse
import os
import sys
import time

import torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference

# Dataloader (shared with other demos)
from dataloader_for_paper_test import dataload

# Model imports
sys.modules['transformers'] = None  # avoid unwanted imports

# UXNet
UXNET_PATH = os.path.join(os.path.dirname(__file__), 'UXNet')
if UXNET_PATH not in sys.path:
    sys.path.append(UXNET_PATH)
from UXNet.network_backbone import UXNET

# GLIMS
GLIMS_PATH = os.path.join(os.path.dirname(__file__), 'Modules_GLMs')
if GLIMS_PATH not in sys.path:
    sys.path.append(GLIMS_PATH)
from Modules_GLMs.GLIMS import GLIMS

# SwinUNETR
from monai.networks.nets import SwinUNETR

# Cascade (old/new)
from models.cascade_both import CascadeBoth
from models.cascade_both_new import CascadeBoth_New
from models.cascade_both_new_centroid import CascadeBoth_New_Centroid

# nnU-Net
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def get_device(gpu):
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model(model_name: str, device: torch.device, weight_path: str = None):
    model_name = model_name.lower()

    if model_name == 'uxnet':
        model = UXNET(
            in_chans=4,
            out_chans=3,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            hidden_size=768,
            spatial_dims=3,
        ).to(device)
        if weight_path and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        return model

    if model_name == 'swinunetr':
        model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=48,
            use_checkpoint=False,
            spatial_dims=3,
        ).to(device)
        if weight_path and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        return model

    if model_name == 'glims':
        model = GLIMS(
            img_size=(96, 96, 96),
            in_channels=4,
            out_channels=3,
            feature_size=24,
            use_checkpoint=False,
        ).to(device)
        if weight_path and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        return model

    if model_name == 'cascade_old':
        model = CascadeBoth(
            num_out_classes=3,
            input_channels=4,
            init_coarse_channels=4,
            init_fine_channels=16,
            testing=True,
        ).to(device)
        if weight_path and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        return model

    if model_name in ('cascade_new', 'cascade_new_centroid'):
        use_centroid = (model_name == 'cascade_new_centroid')
        model = (CascadeBoth_New_Centroid if use_centroid else CascadeBoth_New)(
            num_out_classes=3,
            input_channels=4,
            use_deep_supervision=True,
            testing=True,
        ).to(device)
        if weight_path and os.path.exists(weight_path):
            checkpoint = torch.load(weight_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        return model

    if model_name == 'nnunet':
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=False,
            use_mirroring=False,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=weight_path,
            use_folds=(0,),
            checkpoint_name='checkpoint_best.pth',
        )
        model = predictor.network.to(device)
        model.eval()
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def infer(model_name: str, model, inputs: torch.Tensor):
    model_name = model_name.lower()
    if model_name == 'uxnet':
        with torch.cuda.amp.autocast():
            return torch.sigmoid(sliding_window_inference(
                inputs=inputs,
                roi_size=(160, 160, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                mode='constant',
            ))

    if model_name == 'swinunetr':
        with torch.cuda.amp.autocast():
            return torch.sigmoid(sliding_window_inference(
                inputs=inputs,
                roi_size=(160, 160, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                mode='constant',
            ))

    if model_name == 'glims':
        with torch.cuda.amp.autocast():
            out = sliding_window_inference(
                inputs=inputs,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                mode='gaussian',
            )
            if isinstance(out, list):
                out = out[0]
            return torch.sigmoid(out)

    if model_name in ('cascade_old', 'cascade_new', 'cascade_new_centroid'):
        with torch.cuda.amp.autocast():
            coarse, fine = sliding_window_inference(
                inputs=inputs,
                roi_size=(160, 160, 128),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
                mode='gaussian',
            )
            return fine

    if model_name == 'nnunet':
        # expects (B, C, D, H, W)
        inputs_perm = inputs.permute(0, 1, 4, 2, 3).contiguous()
        logits = sliding_window_inference(
            inputs=inputs_perm,
            roi_size=(128, 160, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
            mode='constant',
        )
        return torch.softmax(logits, dim=1)

    raise ValueError(f"Unsupported model: {model_name}")


def measure_fps(model_name: str, model, test_loader, warmup: int, max_samples: int, device: torch.device):
    times = []
    total = 0

    # warmup
    for i, batch in enumerate(test_loader):
        if i >= warmup:
            break
        inputs = batch['image'].to(device)
        _ = infer(model_name, model, inputs)

    torch.cuda.synchronize(device) if device.type == 'cuda' else None

    total_iters = max_samples if max_samples is not None else len(test_loader)

    for i, batch in enumerate(tqdm(test_loader, total=total_iters, desc='Benchmark', leave=False)):
        if max_samples is not None and total >= max_samples:
            break
        inputs = batch['image'].to(device)
        start = time.time()
        _ = infer(model_name, model, inputs)
        torch.cuda.synchronize(device) if device.type == 'cuda' else None
        end = time.time()
        times.append(end - start)
        total += inputs.shape[0]

    avg_time = sum(times) / len(times) if times else 0.0
    fps = 1.0 / avg_time if avg_time > 0 else 0.0
    return avg_time, fps


def main():
    parser = argparse.ArgumentParser(description='Quick FPS benchmark for multiple models (no metrics).')
    parser.add_argument('--model', type=str, required=True,
                        choices=['uxnet', 'swinunetr', 'glims', 'nnunet', 'cascade_old', 'cascade_new', 'cascade_new_centroid'],
                        help='Which model to benchmark')
    parser.add_argument('--weights', type=str, default=None, help='Weight path or nnUNet results folder (for nnunet)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--max_samples', type=int, default=10, help='Number of test samples to benchmark')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup iterations before timing')
    args = parser.parse_args()

    device = get_device(args.gpu)

    # default weight paths
    if args.weights is None:
        if args.model == 'uxnet':
            args.weights = '/home/pyy/3D_MRI/two_stage/weights/best_composite_UXNET12131853.pth'
        elif args.model == 'swinunetr':
            args.weights = '/home/pyy/3D_MRI/two_stage/weights/best_composite_swinunetr_12172032.pth'
        elif args.model == 'glims':
            args.weights = '/home/pyy/3D_MRI/two_stage/weights/best_composite_glms_12231832.pth'
        elif args.model == 'nnunet':
            args.weights = '/home/pyy/nnUNet_results/Dataset001_BraTS2021/nnUNetTrainer__nnUNetPlans__3d_fullres'
        elif args.model == 'cascade_old':
            args.weights = '/home/pyy/3D_MRI/two_stage/weights/old_best_composite12092218.pth'
        elif args.model == 'cascade_new':
            args.weights = '/home/pyy/3D_MRI/two_stage/weights/best_composite11281329.pth'
        elif args.model == 'cascade_new_centroid':
            args.weights = '/home/pyy/3D_MRI/two_stage/weights/best_composite11281329.pth'

    print('=' * 80)
    print(f'FPS Benchmark')
    print(f'  Model   : {args.model}')
    print(f'  Weights : {args.weights}')
    print(f'  GPU     : {device}')
    print(f'  Max samples: {args.max_samples}, Warmup: {args.warmup}')
    print('=' * 80)

    model = build_model(args.model, device, args.weights)
    model.eval()

    _, _, test_loader = dataload(test_batchsize=1)

    avg_time, fps = measure_fps(args.model, model, test_loader, args.warmup, args.max_samples, device)

    print('\n===== FPS Result =====')
    print(f'Average inference time per sample: {avg_time:.4f} s')
    print(f'FPS (samples/sec): {fps:.2f}')


if __name__ == '__main__':
    main()
