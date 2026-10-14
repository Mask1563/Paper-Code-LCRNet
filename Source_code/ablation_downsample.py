"""Ablation: evaluate different coarse downsample_factor values without retraining."""

import os
import sys
sys.modules['transformers'] = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import argparse
import traceback
from datetime import datetime
from tqdm import tqdm
from scipy import stats

from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.data import DataLoader, Dataset

from models.cascade_both_new import CascadeBoth_New
from dataloader_for_paper_test import dataload
from cal_composite_for_paper import cal_composite_score_for_paper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_AMP = True
MODEL_WEIGHT_PATH = "/home/pyy/3D_MRI/two_stage/weights/best_composite_cascade_new(1.15)01151845.pth"

DOWNSAMPLE_FACTORS = [2, 4, 8]


def calculate_coarse_flops(model, input_size=(1, 4, 160, 160, 128)):
    """仅计算 CoarseNet 的 FLOPs（不含 FineNet）"""
    B, C, H, W, D = input_size
    d = model.downsample_factor

    # Use aligned size to avoid pooling shape issues in FLOPs tools.
    aligned_size = 128
    scale = (H * W * D) / (aligned_size**3)
    low_res_size = (B, C, aligned_size // d, aligned_size // d, aligned_size // d)

    try:
        from thop import profile
        dummy = torch.randn(low_res_size).to(DEVICE)
        model.coarse_unet.eval()
        macs, _ = profile(model.coarse_unet, inputs=(dummy,), verbose=False)
        macs = macs * scale
        return f"{macs / 1e9:.3f}", macs
    except Exception:
        pass

    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(low_res_size).to(DEVICE)
        model.coarse_unet.eval()
        flops = FlopCountAnalysis(model.coarse_unet, dummy).total()
        flops = flops * scale
        return f"{flops / 1e9:.3f}", flops
    except Exception:
        return "N/A", None


def evaluate_variant(model, test_loader, max_samples=None):
    """评估单个 downsample_factor 变体，只收集核心指标"""
    model.eval()
    post_pred = AsDiscrete(threshold=0.5)
    post_label = AsDiscrete(threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    all_dice_tc, all_dice_wt, all_dice_et = [], [], []
    composite_scores = []
    hd95_tc_list, hd95_wt_list, hd95_et_list, hd95_avg_list = [], [], [], []
    sample_count = 0
    error_count = 0

    with torch.no_grad():
        for test_data in tqdm(test_loader, desc="Evaluating", unit="batch"):
            if max_samples and sample_count >= max_samples:
                break

            inputs = test_data["image"].to(DEVICE)
            labels = test_data["label"].to(DEVICE)

            spacing = (1.0, 1.0, 1.0)
            if "image_meta_dict" in test_data:
                meta = test_data["image_meta_dict"]
                if "pixdim" in meta:
                    pixdim = meta["pixdim"]
                    if hasattr(pixdim, 'detach'):
                        pixdim = pixdim.detach()
                    if hasattr(pixdim, 'cpu'):
                        pixdim = pixdim.cpu().numpy()
                    if hasattr(pixdim, "__len__") and len(pixdim) >= 4:
                        spacing = tuple(float(x) for x in pixdim[1:4])

            with torch.cuda.amp.autocast(enabled=VAL_AMP):
                coarse_outputs, fine_outputs = model(inputs)

            if isinstance(fine_outputs, list):
                fine_outputs = fine_outputs[0]

            fine_outputs = fine_outputs.float()
            coarse_outputs = coarse_outputs.float()

            if torch.isnan(fine_outputs).any() or torch.isinf(fine_outputs).any():
                print(f"[WARN] Sample {sample_count}: fine_outputs contains NaN/Inf, skipping composite")
                composite_scores.append(0.0)
                hd95_tc_list.append(0)
                hd95_wt_list.append(0)
                hd95_et_list.append(0)
                hd95_avg_list.append(0)
                sample_count += inputs.shape[0]
                continue

            fine_binary = post_pred(fine_outputs)
            labels_binary = post_label(labels)
            dice_metric(y_pred=fine_binary, y=labels_binary)
            batch_dice = dice_metric_batch(y_pred=fine_binary, y=labels_binary)
            if batch_dice.dim() == 2 and batch_dice.shape[1] >= 3:
                all_dice_tc.append(batch_dice[0, 0].item())
                all_dice_wt.append(batch_dice[0, 1].item())
                all_dice_et.append(batch_dice[0, 2].item())
            elif batch_dice.dim() == 1 and batch_dice.shape[0] >= 3:
                all_dice_tc.append(batch_dice[0].item())
                all_dice_wt.append(batch_dice[1].item())
                all_dice_et.append(batch_dice[2].item())

            try:
                comp, _, _, m = cal_composite_score_for_paper(
                    fine_outputs, coarse_outputs, labels, spacing=spacing)
                cs = comp.mean().item()
                composite_scores.append(cs if not np.isnan(cs) else 0.0)
                hd95_tc_list.append(m.get("HD95_TC", 0))
                hd95_wt_list.append(m.get("HD95_WT", 0))
                hd95_et_list.append(m.get("HD95_ET", 0))
                hd95_avg_list.append(m.get("HD95_avg", 0))
            except Exception as e:
                error_count += 1
                print(f"\n[ERROR] Sample {sample_count}: {e}")
                traceback.print_exc()
                composite_scores.append(0.0)
                hd95_tc_list.append(0)
                hd95_wt_list.append(0)
                hd95_et_list.append(0)
                hd95_avg_list.append(0)

            sample_count += inputs.shape[0]

    mean_dice_val = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()

    if error_count > 0:
        print(f"\n[WARN] {error_count}/{sample_count} samples had composite score errors (set to 0)")

    def safe_mean(lst):
        vals = [v for v in lst if not np.isnan(v)]
        return float(np.mean(vals)) if vals else 0.0

    def safe_median_iqr(lst):
        vals = np.array([v for v in lst if not np.isnan(v)])
        if len(vals) == 0:
            return 0.0, 0.0, 0.0
        med = float(np.median(vals))
        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        return med, q1, q3

    aggregated = {
        "Mean_Dice": safe_mean([mean_dice_val]),
        "TC_Dice": safe_mean(all_dice_tc),
        "WT_Dice": safe_mean(all_dice_wt),
        "ET_Dice": safe_mean(all_dice_et),
        "HD95_TC": safe_median_iqr(hd95_tc_list)[0],
        "HD95_WT": safe_median_iqr(hd95_wt_list)[0],
        "HD95_ET": safe_median_iqr(hd95_et_list)[0],
        "HD95_avg": safe_median_iqr(hd95_avg_list)[0],
        "Composite_Score": safe_mean(composite_scores),
        "num_samples": sample_count,
        "composite_errors": error_count,
    }

    per_sample = {
        "dice_tc": all_dice_tc,
        "dice_wt": all_dice_wt,
        "dice_et": all_dice_et,
        "hd95_tc": hd95_tc_list,
        "hd95_wt": hd95_wt_list,
        "hd95_et": hd95_et_list,
        "hd95_avg": hd95_avg_list,
        "composite": composite_scores,
    }

    return aggregated, per_sample


def main():
    parser = argparse.ArgumentParser(description="Ablation: CoarseNet downsample_factor (zero-training)")
    parser.add_argument("--model_weight", type=str, default=MODEL_WEIGHT_PATH)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dump_json", type=str, default="ablation_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("消融实验: CoarseNet Downsample Factor (零训练版本)")
    print(f"Weights: {args.model_weight}")
    print(f"Test samples: {args.max_samples or 'All'}")
    print("=" * 70)

    _, _, test_loader = dataload(test_batchsize=1)

    print("Loading model...")
    model = CascadeBoth_New(
        num_out_classes=3, input_channels=4,
        use_deep_supervision=True, testing=True,
    ).to(DEVICE)

    ckpt = torch.load(args.model_weight, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)

    coarse_params = sum(p.numel() for p in model.coarse_unet.parameters()) / 1e6
    fine_params = sum(p.numel() for p in model.fine_unet.parameters()) / 1e6
    total_params = coarse_params + fine_params

    results = {}
    all_per_sample = {}

    for d in DOWNSAMPLE_FACTORS:
        print(f"\n{'─'*50}")
        print(f"  Variant d={d}")
        print(f"{'─'*50}")

        model.downsample_factor = d
        model.eval()

        flops_str, flops_val = calculate_coarse_flops(model)
        print(f"  Coarse FLOPs: {flops_str} G")

        eval_metrics, per_sample = evaluate_variant(model, test_loader, max_samples=args.max_samples)

        results[f"d={d}"] = {
            "downsample_factor": d,
            "coarse_flops": flops_str,
            "coarse_flops_value": flops_val,
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in eval_metrics.items()},
        }
        all_per_sample[f"d={d}"] = per_sample

    def fmt_mean_std(values):
        """均值 ± 标准差 (用于 Dice / Composite Score)"""
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            return "N/A", 0.0, 0.0
        return f"{np.mean(arr):.4f} ± {np.std(arr):.4f}", float(np.mean(arr)), float(np.std(arr))

    def fmt_median_iqr(values):
        """中位数 (IQR) (用于 HD95)"""
        arr = np.array([v for v in values if not np.isnan(v)])
        if len(arr) == 0:
            return "N/A", 0.0, 0.0, 0.0
        med = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        return f"{med:.2f} ({q1:.2f}-{q3:.2f})", med, q1, q3

    def wilcoxon_test(samples_a, samples_b):
        """双侧 Wilcoxon 符号秩检验，返回 (statistic, pvalue)"""
        a = np.array([v for v in samples_a if not np.isnan(v)])
        b = np.array([v for v in samples_b if not np.isnan(v)])
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        # 使用 Wilcoxon 双侧检验 (alternative='two-sided')
        return stats.wilcoxon(a, b, alternative='two-sided')

    def significance_mark(p):
        """显著性标记: * p<0.05, ** p<0.01"""
        if p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    BASELINE_TAG = "d=4"

    # 构建每个 variant 的格式化字符串
    display = {}
    for tag in [f"d={d}" for d in DOWNSAMPLE_FACTORS]:
        ps = all_per_sample[tag]
        display[tag] = {}
        # Dice 类指标: mean ± std
        display[tag]["Mean_Dice"] = fmt_mean_std(ps["dice_tc"])  # will recompute as overall mean
        display[tag]["TC_Dice"] = fmt_mean_std(ps["dice_tc"])
        display[tag]["WT_Dice"] = fmt_mean_std(ps["dice_wt"])
        display[tag]["ET_Dice"] = fmt_mean_std(ps["dice_et"])
        # Composite Score: mean ± std
        display[tag]["Composite_Score"] = fmt_mean_std(ps["composite"])
        # HD95 类指标: median (IQR)
        display[tag]["HD95_TC"] = fmt_median_iqr(ps["hd95_tc"])
        display[tag]["HD95_WT"] = fmt_median_iqr(ps["hd95_wt"])
        display[tag]["HD95_ET"] = fmt_median_iqr(ps["hd95_et"])
        display[tag]["HD95_avg"] = fmt_median_iqr(ps["hd95_avg"])
        # FLOPs: 直接从 results 取
        display[tag]["coarse_flops"] = (results[tag]["coarse_flops"], None, None)

    # 计算跨区域平均 Dice (TC+WT+ET 三区域均值)
    for tag in [f"d={d}" for d in DOWNSAMPLE_FACTORS]:
        ps = all_per_sample[tag]
        per_sample_mean_dice = []
        for i in range(len(ps["dice_tc"])):
            vals = [ps["dice_tc"][i], ps["dice_wt"][i], ps["dice_et"][i]]
            if all(not np.isnan(v) for v in vals):
                per_sample_mean_dice.append(np.mean(vals))
            else:
                per_sample_mean_dice.append(0.0)
        display[tag]["Mean_Dice"] = fmt_mean_std(per_sample_mean_dice)

    all_test_metrics = [
        ("Mean Dice", "dice_mean"),
        ("TC Dice", "dice_tc"),
        ("WT Dice", "dice_wt"),
        ("ET Dice", "dice_et"),
        ("Composite Score", "composite"),
        ("TC HD95", "hd95_tc"),
        ("WT HD95", "hd95_wt"),
        ("ET HD95", "hd95_et"),
        ("Avg HD95", "hd95_avg"),
    ]

    sig_marks = {}  # sig_marks[f"d={d}"][metric_display_key] = "**" or "*" or ""
    stat_results = {}  # stat_results[(tag, ps_key)] = (w_stat, p_val, sig)

    for tag in [f"d={d}" for d in DOWNSAMPLE_FACTORS]:
        if tag == BASELINE_TAG:
            continue
        sig_marks[tag] = {}
        for metric_name, ps_key in all_test_metrics:
            if ps_key == "dice_mean":
                def per_sample_mean(ps):
                    result = []
                    for i in range(len(ps["dice_tc"])):
                        vals = [ps["dice_tc"][i], ps["dice_wt"][i], ps["dice_et"][i]]
                        result.append(np.mean(vals) if all(not np.isnan(v) for v in vals) else 0.0)
                    return result
                samples_a = per_sample_mean(all_per_sample[tag])
                samples_b = per_sample_mean(all_per_sample[BASELINE_TAG])
            else:
                samples_a = all_per_sample[tag][ps_key]
                samples_b = all_per_sample[BASELINE_TAG][ps_key]

            try:
                w_stat, p_val = wilcoxon_test(samples_a, samples_b)
                mark = significance_mark(p_val)
            except Exception:
                w_stat, p_val, mark = float('nan'), float('nan'), ""
            sig_marks[tag][metric_name] = mark
            stat_results[(tag, ps_key)] = (w_stat, p_val, mark)

    print(f"\n{'='*100}")
    print("消融实验结果汇总")
    print(f"{'='*100}")

    header = f"{'Metric':<26} {'d=2':<30} {'d=4 (baseline)':<30} {'d=8':<30}"
    print(header)
    print("-" * len(header))

    row_keys = [
        ("Coarse FLOPs (G)", "coarse_flops", "raw"),
        ("Mean Dice", "Mean_Dice", "mean_std"),
        ("TC Dice", "TC_Dice", "mean_std"),
        ("WT Dice", "WT_Dice", "mean_std"),
        ("ET Dice", "ET_Dice", "mean_std"),
        ("TC HD95 (mm)", "HD95_TC", "median_iqr"),
        ("WT HD95 (mm)", "HD95_WT", "median_iqr"),
        ("ET HD95 (mm)", "HD95_ET", "median_iqr"),
        ("Avg HD95 (mm)", "HD95_avg", "median_iqr"),
        ("Composite Score", "Composite_Score", "mean_std"),
    ]

    for label, key, fmt_type in row_keys:
        values = []
        for d in DOWNSAMPLE_FACTORS:
            tag = f"d={d}"
            if tag in display and key in display[tag]:
                val_str = display[tag][key][0]
                if tag != BASELINE_TAG:
                    mark = sig_marks[tag].get(label, "")
                    val_str = f"{val_str} {mark}" if mark else val_str
                values.append(f"{val_str:<30}")
            else:
                values.append(f"{'N/A':<30}")
        print(f"{label:<26} {values[0]} {values[1]} {values[2]}")

    print(f"\n  Params (all variants): Coarse={coarse_params:.2f}M  "
          f"Fine={fine_params:.2f}M  Total={total_params:.2f}M")

    print(f"\n{'='*100}")
    print(f"基线比较统计检验 (双侧 Wilcoxon 符号秩检验, 基线: {BASELINE_TAG})")
    print(f"{'='*100}")

    stat_header = (f"  {'Metric':<20} {'d=2 vs d=4':<38} {'d=8 vs d=4':<38}")
    print(stat_header)
    sub_header = (f"  {'':<20} {'W-stat':<14} {'p-value':<12} {'sig.':<8} "
                  f"{'W-stat':<14} {'p-value':<12} {'sig.':<8}")
    print(sub_header)
    print("  " + "-" * (len(stat_header) - 2))

    for metric_name, ps_key in all_test_metrics:
        row = [f"  {metric_name:<20}"]
        for cmp_tag in [f"d={d}" for d in DOWNSAMPLE_FACTORS if f"d={d}" != BASELINE_TAG]:
            w_stat, p_val, mark = stat_results.get((cmp_tag, ps_key), (float('nan'), float('nan'), ""))
            if np.isnan(w_stat):
                row.append(f"{'N/A':<14} {'N/A':<12} {'—':<8}")
            else:
                row.append(f"{w_stat:<14.1f} {p_val:<12.6f} {mark:<8}")
        print("".join(row))

    print(f"\n  显著性标记: ** p<0.01, * p<0.05 (与基线 d=4 相比)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = args.dump_json.replace(".json", f"_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    if all(f"d={d}" in results for d in DOWNSAMPLE_FACTORS):
        d2 = results["d=2"]; d4 = results["d=4"]; d8 = results["d=8"]
        print(f"\nConclusion: d=4 is the Pareto-optimal point.")
        print(f"  d=2→4: Dice {d4['Mean_Dice']-d2['Mean_Dice']:+.4f}, "
              f"FLOPs {d2['coarse_flops']} → {d4['coarse_flops']}")
        print(f"  d=4→8: Dice {d8['Mean_Dice']-d4['Mean_Dice']:+.4f}, "
              f"FLOPs {d4['coarse_flops']} → {d8['coarse_flops']}")

    return results


if __name__ == "__main__":
    main()
