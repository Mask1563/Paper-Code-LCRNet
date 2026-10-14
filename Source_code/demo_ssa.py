import os
import sys

# Hack: disable transformers import to avoid numpy/pytree issues.
sys.modules['transformers'] = None

import torch
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
from glob import glob
from tqdm import tqdm
from torchsummary import summary
from dataloader import dataload
from dataloader_ssa import dataload_ssa
import matplotlib.pyplot as plt
import os
import numpy as np
from models.cascade_both import CascadeBoth
from models.cascade_both_new import CascadeBoth_New  # 新增：导入新架构
from cal_composite import cal_composite_score
from cal_composite_for_paper import cal_composite_score_for_paper
import random
import time
from thop import profile, clever_format
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Brats SSA generalization test.

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


VAL_AMP = True


def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(160, 160, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型选择配置
USE_NEW_ARCHITECTURE = True  # 切换测试: True=新架构, False=旧架构

if USE_NEW_ARCHITECTURE:
    if os.name == 'nt':
        MODEL_WEIGHT_PATH = "D:/wjh/3D_MRI/two_stage/weights/best_composite_cascade_new(1.15)01151845.pth"
    else:
        MODEL_WEIGHT_PATH = "/home/pyy/3D_MRI/two_stage/weights/best_composite_cascade_new(1.15)01151845.pth"
else:
    if os.name == 'nt':
        MODEL_WEIGHT_PATH = "D:/wjh/3D_MRI/two_stage/weights/best11121726.pth"
    else:
        MODEL_WEIGHT_PATH = "/home/pyy/3D_MRI/two_stage/weights/best_composite_var3_120new01202151.pth"

print(f"{'=' * 80}")
print(f"Testing Configuration:")
print(
    f"  Architecture: {'CascadeBoth_New (3D Coarse + ConvNeXt)' if USE_NEW_ARCHITECTURE else 'CascadeBoth (2D Fuse + FineNet)'}")
print(f"  Model weights: {MODEL_WEIGHT_PATH}")
print(f"{'=' * 80}\n")

test_loader = dataload_ssa()

if USE_NEW_ARCHITECTURE:
    model = CascadeBoth_New(
        num_out_classes=3,
        input_channels=4,
        use_deep_supervision=True,
        testing=True
    )
else:
    model = CascadeBoth(
        num_out_classes=3,
        input_channels=4,
        init_coarse_channels=4,
        init_fine_channels=16,
        testing=True
    )

model.to(device)

checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'best_composite_score' in checkpoint:
        print(f"  Best Composite Score: {checkpoint['best_composite_score']:.4f}")
else:
    model.load_state_dict(checkpoint)
    print(f"✓ Loaded model weights")

def calculate_model_complexity(model, input_size=(1, 4, 160, 160, 128)):
    print("\nCalculating model complexity...")
    
    total_params = sum(p.numel() for p in model.parameters())
    
    stage1_params = 0
    stage2_params = 0
    
    if hasattr(model, 'coarse_unet'):
        stage1_params = sum(p.numel() for p in model.coarse_unet.parameters())
    
    if hasattr(model, 'fine_unet'):
        stage2_params = sum(p.numel() for p in model.fine_unet.parameters())
        
    print(f"Stage 1 (Coarse) Params: {stage1_params / 1e6:.2f} M")
    print(f"Stage 2 (Fine) Params: {stage2_params / 1e6:.2f} M")
    print(f"Total Params: {total_params / 1e6:.2f} M")
    
    flops = "N/A"
    try:
        dummy_input = torch.randn(input_size).to(device)
        model.eval()
        macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
        flops, params = clever_format([macs, total_params], "%.3f")
        print(f"FLOPs (approx): {flops}")
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")
        
    return {
        "stage1_params_M": stage1_params / 1e6,
        "stage2_params_M": stage2_params / 1e6,
        "total_params_M": total_params / 1e6,
        "flops": flops
    }

model.eval()

complexity_metrics = calculate_model_complexity(model)

post_pred = AsDiscrete(threshold=0.5)
post_label = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# 用于存储综合评分
composite_scores = []
accuracy_scores = []
reliability_scores = []
metrics_list = []
all_sample_results = []

# 记录推理时间
inference_times = []

sample_count = 0
with torch.no_grad():
    with tqdm(test_loader, desc="Testing", unit="batch") as tqdm_loader:
        for batch_idx, test_data in enumerate(tqdm_loader):
            
            # 记录开始时间
            start_time = time.time()

            inputs, labels = test_data["image"].to(device), test_data["label"].to(device)

            # 获取体素间距信息
            if isinstance(test_data, dict) and "image_meta_dict" in test_data:
                meta_dict = test_data["image_meta_dict"]
                if isinstance(meta_dict, dict) and "pixdim" in meta_dict:
                    pixdim = meta_dict["pixdim"]
                    if hasattr(pixdim, 'cpu'):
                        spacing = tuple(pixdim.cpu().numpy()[:3])
                    else:
                        spacing = tuple(pixdim[:3])
                else:
                    spacing = (1.0, 1.0, 1.0)
            else:
                spacing = (1.0, 1.0, 1.0)

            # 模型推理
            coarse_outputs, fine_outputs = inference(inputs)
            
            # 记录结束时间并计算推理耗时
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)

            # 处理深度监督输出（如果模型返回list，取最终输出）
            if isinstance(fine_outputs, list):
                fine_outputs = fine_outputs[0]  # 取最终层输出

            # 应用二值化后处理（与训练脚本保持一致）
            fine_outputs_binary = post_pred(fine_outputs)
            labels_binary = post_label(labels)

            # 计算Dice指标
            dice_metric(y_pred=fine_outputs_binary, y=labels_binary)
            dice_metric_batch(y_pred=fine_outputs_binary, y=labels_binary)

            # 计算综合评分
            # 直接使用fine_outputs和labels（无需转换，dataloader已处理）
            composite_score, accuracy_score, reliability_score, metrics = cal_composite_score_for_paper(
                fine_outputs,  # 已经是(B, 3, H, W, D) [TC, WT, ET]
                coarse_outputs,  # 已经是(B, 1, H, W, D)
                labels,  # 已经是(B, 3, H, W, D) [TC, WT, ET]
                spacing=spacing
            )

            # 存储结果
            composite_scores.append(composite_score.mean().item())
            accuracy_scores.append(accuracy_score.mean().item())
            reliability_scores.append(reliability_score.mean().item())
            metrics_list.append(metrics)

            # 存储每个样本的详细信息用于后续随机展示
            for i in range(inputs.shape[0]):
                sample_result = {
                    'sample_id': sample_count + i,
                    'composite_score': composite_score[
                        i].item() if composite_score.dim() > 0 else composite_score.item(),
                    'accuracy_score': accuracy_score[i].item() if accuracy_score.dim() > 0 else accuracy_score.item(),
                    'reliability_score': reliability_score[
                        i].item() if reliability_score.dim() > 0 else reliability_score.item(),
                    'metrics': {k: v[i] if isinstance(v, torch.Tensor) and v.numel() > 1 else v
                                for k, v in metrics.items()}
                }
                all_sample_results.append(sample_result)

            sample_count += inputs.shape[0]

# 随机选择三个样本展示结果
print("\n===== 随机三个样本的综合评分结果 =====")
random_samples = random.sample(all_sample_results, min(3, len(all_sample_results)))
for i, sample in enumerate(random_samples):
    print(f"\n样本 {sample['sample_id']}:")
    print(f"综合评分: {sample['composite_score']:.4f}")
    print(f"精度评分: {sample['accuracy_score']:.4f}")
    print(f"可靠性评分: {sample['reliability_score']:.4f}")
    for key, value in sample['metrics'].items():
        print(f"{key}: {value:.4f}")

# 计算并打印平均指标
avg_composite_score = np.mean(composite_scores)
avg_accuracy_score = np.mean(accuracy_scores)
avg_reliability_score = np.mean(reliability_scores)

# 计算速度指标
avg_inference_time = np.mean(inference_times)
fps = 1.0 / avg_inference_time
throughput = 1.0 / avg_inference_time  # 对于batch_size=1，吞吐量等于FPS

print("\n===== 效率与复杂度指标 =====")
print(f"Stage 1 (Coarse) Params: {complexity_metrics['stage1_params_M']:.2f} M")
print(f"Stage 2 (Fine) Params: {complexity_metrics['stage2_params_M']:.2f} M")
print(f"Total Params: {complexity_metrics['total_params_M']:.2f} M")
print(f"FLOPs: {complexity_metrics['flops']}")
print(f"Average Inference Time: {avg_inference_time:.4f} s/sample")
print(f"Inference Speed (FPS): {fps:.2f} samples/s")
print(f"Throughput: {throughput:.2f} samples/s")

# 从metrics_list中提取Dice系数和HD95系数（逐样本）
mean_dice_values = np.array([m["Mean Dice"] for m in metrics_list])
tc_dice_values = np.array([m["TC Dice"] for m in metrics_list])
wt_dice_values = np.array([m["WT Dice"] for m in metrics_list])
et_dice_values = np.array([m["ET Dice"] for m in metrics_list])

# HD95系数（三个区域和总体平均）
hd95_tc_values = np.array([m["HD95_TC"] for m in metrics_list])
hd95_wt_values = np.array([m["HD95_WT"] for m in metrics_list])
hd95_et_values = np.array([m["HD95_ET"] for m in metrics_list])
hd95_avg_values = np.array([m["HD95_avg"] for m in metrics_list])

# 综合评分
composite_scores_arr = np.array(composite_scores)
accuracy_scores_arr = np.array(accuracy_scores)
reliability_scores_arr = np.array(reliability_scores)

print("\n===== 测试结果汇总 =====")
print(f"综合评分: {composite_scores_arr.mean():.4f} ± {composite_scores_arr.std():.4f}")
print(f"精度评分: {accuracy_scores_arr.mean():.4f} ± {accuracy_scores_arr.std():.4f}")
print(f"可靠性评分: {reliability_scores_arr.mean():.4f} ± {reliability_scores_arr.std():.4f}")
print(f"\n--- Dice系数 (mean ± std) ---")
print(f"Mean Dice:  {mean_dice_values.mean():.4f} ± {mean_dice_values.std():.4f}")
print(f"TC Dice:    {tc_dice_values.mean():.4f} ± {tc_dice_values.std():.4f}")
print(f"WT Dice:    {wt_dice_values.mean():.4f} ± {wt_dice_values.std():.4f}")
print(f"ET Dice:    {et_dice_values.mean():.4f} ± {et_dice_values.std():.4f}")
print(f"\n--- HD95距离 / mm (中位数, Q1-Q3) ---")
print(f"TC HD95:       {np.median(hd95_tc_values):.2f} ({np.percentile(hd95_tc_values, 25):.2f}-{np.percentile(hd95_tc_values, 75):.2f})")
print(f"WT HD95:       {np.median(hd95_wt_values):.2f} ({np.percentile(hd95_wt_values, 25):.2f}-{np.percentile(hd95_wt_values, 75):.2f})")
print(f"ET HD95:       {np.median(hd95_et_values):.2f} ({np.percentile(hd95_et_values, 25):.2f}-{np.percentile(hd95_et_values, 75):.2f})")
print(f"Mean HD95:     {np.median(hd95_avg_values):.2f} ({np.percentile(hd95_avg_values, 25):.2f}-{np.percentile(hd95_avg_values, 75):.2f})")

# 重置指标
dice_metric.reset()
dice_metric_batch.reset()