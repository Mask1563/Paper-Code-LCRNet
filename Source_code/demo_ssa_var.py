import os
import sys
import time
import random
import torch
import numpy as np
from tqdm import tqdm
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from thop import profile, clever_format

# Hack: disable transformers import to avoid side-effects.
sys.modules["transformers"] = None

from dataloader_ssa import dataload_ssa
from cal_composite_for_paper import cal_composite_score_for_paper
from models.Var_2 import Var_2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
VAL_AMP = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_WEIGHT_PATH = "/home/pyy/3D_MRI/two_stage/weights/best_composite_var2_01181241.pth"  # TODO: set Var_2 SSA weight

print("=" * 80)
print("Testing Configuration (Var_2 on SSA)")
print(f"  Model weights: {MODEL_WEIGHT_PATH}")
print("=" * 80 + "\n")

test_loader = dataload_ssa()

model = Var_2(
    num_out_classes=3,
    input_channels=4,
    coarse_base_channels=8,
    downsample_factor=4,
    testing=True,
)
model.to(device)

checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if "best_composite_score" in checkpoint:
        print(f"  Best Composite Score: {checkpoint['best_composite_score']:.4f}")
else:
    model.load_state_dict(checkpoint)
    print("✓ Loaded model weights")

def calculate_model_complexity(model, input_size=(1, 4, 160, 160, 128)):
    print("\nCalculating model complexity...")
    total_params = sum(p.numel() for p in model.parameters())
    stage1_params = sum(p.numel() for p in model.coarse_unet.parameters()) if hasattr(model, "coarse_unet") else 0
    stage2_params = sum(p.numel() for p in model.fine_unet.parameters()) if hasattr(model, "fine_unet") else 0

    print(f"Stage 1 (Coarse) Params: {stage1_params / 1e6:.2f} M")
    print(f"Stage 2 (Fine) Params: {stage2_params / 1e6:.2f} M")
    print(f"Total Params: {total_params / 1e6:.2f} M")

    flops = "N/A"
    try:
        dummy_input = torch.randn(input_size).to(device)
        model.eval()
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        flops, _ = clever_format([macs, total_params], "%.3f")
        print(f"FLOPs (approx): {flops}")
    except Exception as e:  # noqa: BLE001
        print(f"Could not calculate FLOPs: {e}")

    return {
        "stage1_params_M": stage1_params / 1e6,
        "stage2_params_M": stage2_params / 1e6,
        "total_params_M": total_params / 1e6,
        "flops": flops,
    }

model.eval()
complexity_metrics = calculate_model_complexity(model)

def inference(inputs):
    def _compute(x):
        return sliding_window_inference(
            inputs=x,
            roi_size=(160, 160, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            return _compute(inputs)
    return _compute(inputs)

post_pred = AsDiscrete(threshold=0.5)
post_label = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

composite_scores = []
accuracy_scores = []
reliability_scores = []
metrics_list = []
all_sample_results = []
inference_times = []

sample_count = 0
with torch.no_grad():
    with tqdm(test_loader, desc="Testing", unit="batch") as tqdm_loader:
        for batch_idx, test_data in enumerate(tqdm_loader):
            start_time = time.time()

            inputs, labels = test_data["image"].to(device), test_data["label"].to(device)

            # spacing
            if isinstance(test_data, dict) and "image_meta_dict" in test_data:
                meta_dict = test_data["image_meta_dict"]
                if isinstance(meta_dict, dict) and "pixdim" in meta_dict:
                    pixdim = meta_dict["pixdim"]
                    if hasattr(pixdim, "cpu"):
                        spacing = tuple(pixdim.cpu().numpy()[:3])
                    else:
                        spacing = tuple(pixdim[:3])
                else:
                    spacing = (1.0, 1.0, 1.0)
            else:
                spacing = (1.0, 1.0, 1.0)

            coarse_outputs, fine_outputs = inference(inputs)

            end_time = time.time()
            inference_times.append(end_time - start_time)

            if isinstance(fine_outputs, list):
                fine_outputs = fine_outputs[0]

            fine_outputs_binary = post_pred(fine_outputs)
            labels_binary = post_label(labels)

            dice_metric(y_pred=fine_outputs_binary, y=labels_binary)
            dice_metric_batch(y_pred=fine_outputs_binary, y=labels_binary)

            composite_score, accuracy_score, reliability_score, metrics = cal_composite_score_for_paper(
                fine_outputs,
                coarse_outputs,
                labels,
                spacing=spacing,
            )

            composite_scores.append(composite_score.mean().item())
            accuracy_scores.append(accuracy_score.mean().item())
            reliability_scores.append(reliability_score.mean().item())
            metrics_list.append(metrics)

            for i in range(inputs.shape[0]):
                sample_result = {
                    "sample_id": sample_count + i,
                    "composite_score": composite_score[i].item() if composite_score.dim() > 0 else composite_score.item(),
                    "accuracy_score": accuracy_score[i].item() if accuracy_score.dim() > 0 else accuracy_score.item(),
                    "reliability_score": reliability_score[i].item() if reliability_score.dim() > 0 else reliability_score.item(),
                    "metrics": {k: (v[i] if isinstance(v, torch.Tensor) and v.numel() > 1 else v) for k, v in metrics.items()},
                }
                all_sample_results.append(sample_result)

            sample_count += inputs.shape[0]

print("\n===== 随机三个样本的综合评分结果 =====")
random_samples = random.sample(all_sample_results, min(3, len(all_sample_results)))
for sample in random_samples:
    print(f"\n样本 {sample['sample_id']}:")
    print(f"综合评分: {sample['composite_score']:.4f}")
    print(f"精度评分: {sample['accuracy_score']:.4f}")
    print(f"可靠性评分: {sample['reliability_score']:.4f}")
    for key, value in sample["metrics"].items():
        print(f"{key}: {value:.4f}")

avg_composite_score = np.mean(composite_scores)
avg_accuracy_score = np.mean(accuracy_scores)
avg_reliability_score = np.mean(reliability_scores)

avg_inference_time = np.mean(inference_times)
fps = 1.0 / avg_inference_time
throughput = 1.0 / avg_inference_time  # batch=1

print("\n===== 效率与复杂度指标 =====")
print(f"Stage 1 (Coarse) Params: {complexity_metrics['stage1_params_M']:.2f} M")
print(f"Stage 2 (Fine) Params: {complexity_metrics['stage2_params_M']:.2f} M")
print(f"Total Params: {complexity_metrics['total_params_M']:.2f} M")
print(f"FLOPs: {complexity_metrics['flops']}")
print(f"Average Inference Time: {avg_inference_time:.4f} s/sample")
print(f"Inference Speed (FPS): {fps:.2f} samples/s")
print(f"Throughput: {throughput:.2f} samples/s")

print("\n===== 测试结果汇总 =====")

mean_dice_values = [m["Mean Dice"] for m in metrics_list]
tc_dice_values = [m["TC Dice"] for m in metrics_list]
wt_dice_values = [m["WT Dice"] for m in metrics_list]
et_dice_values = [m["ET Dice"] for m in metrics_list]

avg_mean_dice = np.mean(mean_dice_values)
avg_tc_dice = np.mean(tc_dice_values)
avg_wt_dice = np.mean(wt_dice_values)
avg_et_dice = np.mean(et_dice_values)

hd95_tc_values = [m["HD95_TC"] for m in metrics_list]
hd95_wt_values = [m["HD95_WT"] for m in metrics_list]
hd95_et_values = [m["HD95_ET"] for m in metrics_list]
hd95_avg_values = [m["HD95_avg"] for m in metrics_list]

avg_hd95_tc = np.mean(hd95_tc_values)
avg_hd95_wt = np.mean(hd95_wt_values)
avg_hd95_et = np.mean(hd95_et_values)
avg_hd95_overall = np.mean(hd95_avg_values)

print(f"平均综合评分: {avg_composite_score:.4f}")
print(f"平均精度评分: {avg_accuracy_score:.4f}")
print(f"平均可靠性评分: {avg_reliability_score:.4f}")
print("\n--- Dice系数 ---")
print(f"平均Dice系数: {avg_mean_dice:.4f}")
print(f"TC Dice系数: {avg_tc_dice:.4f}")
print(f"WT Dice系数: {avg_wt_dice:.4f}")
print(f"ET Dice系数: {avg_et_dice:.4f}")
print("\n--- HD95距离 (mm) ---")
print(f"TC HD95: {avg_hd95_tc:.2f}")
print(f"WT HD95: {avg_hd95_wt:.2f}")
print(f"ET HD95: {avg_hd95_et:.2f}")
print(f"总体平均HD95: {avg_hd95_overall:.2f}")

# Reset metrics
dice_metric.reset()
dice_metric_batch.reset()
