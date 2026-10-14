import sys
import os
import argparse
import torch
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from thop import profile, clever_format

# Hack: disable transformers import to avoid side-effects.
sys.modules['transformers'] = None

from dataloader_for_paper_test import dataload
from cal_composite_for_paper import cal_composite_score_for_paper

from models.Var_1 import Var_1
from models.Var_2 import Var_2
from models.Var_3 import Var_3
from models.Var_4 import Var_4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
VAL_AMP = True

# Ablation test. Example: python demo_Var.py --variant Var_1
parser = argparse.ArgumentParser(description="Evaluate Ablation Study Variants (Var_1, Var_2, Var_3, Var_4)")
parser.add_argument("--variant", type=str, required=True, choices=["Var_1", "Var_2", "Var_3", "Var_4"], help="Which variant to test")
# parser.add_argument("--weights", type=str, required=True, help="Path to the model weights (.pth file)")
parser.add_argument("--max_samples", type=int, default=None, help="Only evaluate first N samples for quick checks")
parser.add_argument("--skip_flops", action="store_true", help="Skip FLOPs counting to speed up startup")
args = parser.parse_args()

WEIGHTS_MAP = {
    "Var_1": "/home/pyy/3D_MRI/two_stage/weights/best_composite_var1_01181241.pth",  # 已替换
    "Var_2": "/home/pyy/3D_MRI/two_stage/weights/best_composite_var2_01181241.pth",  # 已替换
    "Var_3": "/home/pyy/3D_MRI/two_stage/weights/best_composite_var3_120new01202151.pth",  # 修改为实际路径
    "Var_4": "/home/pyy/3D_MRI/two_stage/weights/best_composite_var4_120new01202151.pth" # 请修改为训练好的Var_4权重路径
}

weights_path = WEIGHTS_MAP[args.variant]

print(f"{'=' * 80}")
print(f"Testing Configuration: Ablation Study")
print(f"  Variant: {args.variant}")
print(f"  Weights: {weights_path}")
print(f"{'=' * 80}\n")

print(f"Initializing {args.variant}...")

if args.variant == "Var_1":
    model = Var_1(
        num_out_classes=3, 
        input_channels=4, 
        coarse_base_channels=8,
        downsample_factor=4,
        testing=True
    )
elif args.variant == "Var_2":
    model = Var_2(
        num_out_classes=3, 
        input_channels=4, 
        coarse_base_channels=8,
        downsample_factor=4,
        testing=True
    )
elif args.variant == "Var_3":
    model = Var_3(
        num_out_classes=3, 
        input_channels=4, 
        coarse_base_channels=8,
        downsample_factor=4,
        testing=True
    )
elif args.variant == "Var_4":
    model = Var_4(
        num_out_classes=3, 
        input_channels=4, 
        coarse_base_channels=8,
        downsample_factor=4,
        testing=True
    )

model.to(device)

if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model weights directly")
else:
    print(f"❌ Error: Weights file not found at {weights_path}")
    sys.exit(1)

model.eval()

TEST_BATCH_SIZE = 1
_, _, test_loader = dataload(test_batchsize=TEST_BATCH_SIZE)

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
    
    if args.skip_flops:
        flops = "skipped"
        print("FLOPs: skipped")
        return {
            "stage1_params_M": stage1_params / 1e6,
            "stage2_params_M": stage2_params / 1e6,
            "total_params_M": total_params / 1e6,
            "flops": flops
        }

    flops = "N/A"
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(input_size).to(device)
        model.eval()
        flops_count = FlopCountAnalysis(model, dummy).total()
        flops = f"{flops_count / 1e9:.3f} GFLOPs"
        print(f"FLOPs (fvcore): {flops}")
    except Exception as e:
        print(f"fvcore flop count failed: {e}")
        try:
            dummy_input = torch.randn(input_size).to(device)
            model.eval()
            macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
            flops, _ = clever_format([macs, total_params], "%.3f")
            print(f"FLOPs (thop): {flops}")
        except Exception as e_thop:
            print(f"thop flop count failed: {e_thop}")
            flops = "N/A (fvcore & thop failed)"
    
    return {
        "stage1_params_M": stage1_params / 1e6,
        "stage2_params_M": stage2_params / 1e6,
        "total_params_M": total_params / 1e6,
        "flops": flops
    }

complexity_metrics = calculate_model_complexity(model)

def inference(input, model):
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
            return _compute(input)
    return _compute(input)

post_pred = AsDiscrete(threshold=0.5)
post_label = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# 存储结果
composite_scores = []
accuracy_scores = []
reliability_scores = []
metrics_list = []
all_sample_results = []
inference_times = []

sample_count = 0

print("\nStarting Inference Loop...")
with torch.no_grad():
    with tqdm(test_loader, desc="Testing", unit="batch") as tqdm_loader:
        for batch_idx, test_data in enumerate(tqdm_loader):
            if args.max_samples is not None and sample_count >= args.max_samples:
                break
            
            start_time = time.time()

            inputs, labels = test_data["image"].to(device), test_data["label"].to(device)

            # 获取 Spacing
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

            # ========== 推理 ==========
            # Var_X 模型返回 (wt_mask, fine_output)
            # wt_mask: (B, 1, H, W, D)
            # fine_output: (B, 3, H, W, D)
            coarse_outputs, fine_outputs = inference(inputs, model)
            
            # 处理深度监督输出（如果模型返回list，取最终输出）
            if isinstance(fine_outputs, list):
                fine_outputs = fine_outputs[0]

            end_time = time.time()
            inference_times.append(end_time - start_time)

            # ========== 评估指标计算 ==========
            
            # 二值化
            fine_outputs_binary = post_pred(fine_outputs)
            labels_binary = post_label(labels)

            # Dice Metric
            dice_metric(y_pred=fine_outputs_binary, y=labels_binary)
            dice_metric_batch(y_pred=fine_outputs_binary, y=labels_binary)

            # 综合评分
            composite_score, accuracy_score, reliability_score, metrics = cal_composite_score_for_paper(
                fine_outputs,
                coarse_outputs,
                labels,
                spacing=spacing
            )

            # 存储结果
            composite_scores.append(composite_score.mean().item())
            accuracy_scores.append(accuracy_score.mean().item())
            reliability_scores.append(reliability_score.mean().item())
            metrics_list.append(metrics)

            # 存储样本详情
            for i in range(inputs.shape[0]):
                sample_result = {
                    'sample_id': sample_count + i,
                    'composite_score': composite_score[i].item() if composite_score.dim() > 0 else composite_score.item(),
                    'accuracy_score': accuracy_score[i].item() if accuracy_score.dim() > 0 else accuracy_score.item(),
                    'reliability_score': reliability_score[i].item() if reliability_score.dim() > 0 else reliability_score.item(),
                    'metrics': {k: v[i] if isinstance(v, torch.Tensor) and v.numel() > 1 else v
                                for k, v in metrics.items()}
                }
                all_sample_results.append(sample_result)

            sample_count += inputs.shape[0]

# ========== 结果展示 (与 demo_ori.py 一致) ==========

print("\n===== 随机三个样本的综合评分结果 =====")
random_samples = random.sample(all_sample_results, min(3, len(all_sample_results)))
for i, sample in enumerate(random_samples):
    print(f"\n样本 {sample['sample_id']}:")
    print(f"综合评分: {sample['composite_score']:.4f}")
    print(f"精度评分: {sample['accuracy_score']:.4f}")
    print(f"可靠性评分: {sample['reliability_score']:.4f}")
    for key, value in sample['metrics'].items():
        print(f"{key}: {value:.4f}")

# 平均指标
avg_composite_score = np.mean(composite_scores)
avg_accuracy_score = np.mean(accuracy_scores)
avg_reliability_score = np.mean(reliability_scores)

# 速度指标
avg_inference_time = np.mean(inference_times)
fps = 1.0 / avg_inference_time
throughput = 1.0 / avg_inference_time

print("\n===== 效率与复杂度指标 =====")
print(f"Stage 1 (Coarse) Params: {complexity_metrics['stage1_params_M']:.2f} M")
print(f"Stage 2 (Fine) Params: {complexity_metrics['stage2_params_M']:.2f} M")
print(f"Total Params: {complexity_metrics['total_params_M']:.2f} M")
print(f"FLOPs: {complexity_metrics['flops']}")
print(f"Average Inference Time: {avg_inference_time:.4f} s/sample")
print(f"Inference Speed (FPS): {fps:.2f} samples/s")
print(f"Throughput: {throughput:.2f} samples/s")

# Dice 和 HD95 汇总
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

print(f"\n===== 测试结果汇总 ({args.variant}) =====")
print(f"平均综合评分: {avg_composite_score:.4f}")
print(f"平均精度评分: {avg_accuracy_score:.4f}")
print(f"平均可靠性评分: {avg_reliability_score:.4f}")
print(f"\n--- Dice系数 ---")
print(f"平均Dice系数: {avg_mean_dice:.4f}")
print(f"TC Dice系数: {avg_tc_dice:.4f}")
print(f"WT Dice系数: {avg_wt_dice:.4f}")
print(f"ET Dice系数: {avg_et_dice:.4f}")
print(f"\n--- HD95距离 (mm) ---")
print(f"TC HD95: {avg_hd95_tc:.2f}")
print(f"WT HD95: {avg_hd95_wt:.2f}")
print(f"ET HD95: {avg_hd95_et:.2f}")
print(f"总体平均HD95: {avg_hd95_overall:.2f}")

# 重置指标
dice_metric.reset()
dice_metric_batch.reset()
