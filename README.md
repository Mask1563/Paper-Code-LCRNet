# A Lightweight Cascaded 3D Network with Context-Aware Localization and Multi-Dimensional Reliability Quantification for Brain Tumor Segmentation

Official PyTorch / MONAI implementation of the lightweight **coarse-to-fine cascade**
proposed in our paper. The network localizes the tumor on a heavily downsampled volume
with a tiny 3D UNet, crops a tumor-centered ROI, and segments the three regions
(**WT / TC / ET**) at full resolution with a ConvNeXt-style 3D U-Net — trading the cost
of sliding-window inference over the whole volume for high-resolution prediction on a
small patch.

```
                 ┌──────────────────────────┐
4-modality MRI ─▶│ Stage 1: CoarseNet3D     │  input downsampled 1/4, tiny 3D UNet
                 │  (whole-tumor heatmap)   │  → coarse WT mask
                 └───────────┬──────────────┘
                             │  bbox center → 128³ ROI crop
                 ┌───────────▼──────────────┐
  MRI patch + ─▶ │ Stage 2: FineNet_New     │  ConvNeXt-3D blocks + CBAM
  coarse mask    │  (deep supervision)      │  → WT / TC / ET
                 └──────────────────────────┘
```

The whole cascade is a single `nn.Module` (`models/cascade_both_new.py::CascadeBoth_New`)
and is trained **end-to-end**, so both stages receive gradient from the final loss.

---

## Requirements

```bash
conda create -n cascade python=3.9 -y
conda activate cascade
pip install torch==2.0.1 torchvision          # CUDA build matching your driver
pip install monai timm nibabel scipy matplotlib tqdm torchsummary thop
```

Tested on a single NVIDIA GPU with AMP (`torch.cuda.amp`) enabled.

> Note: several scripts start with `sys.modules['transformers'] = None`. This is an
> intentional guard that skips an unrelated transitive import which otherwise breaks
> `timm`/MONAI in our environment. Remove it if your setup does not need it.

## Data Preparation

Images/labels are read as NIfTI (`.nii` / `.nii.gz`) and resampled to `1×1×1 mm`,
reoriented to `RAS`, then cropped to `160×160×128`. Labels are mapped to the standard
BraTS 3-channel order **[TC, WT, ET]** inside the dataloader, so any BraTS-style
single-volume label map (1 = NCR/NET, 2 = EDMA, 3 = ET) works as-is.

Expected layout (edit `data_root` in `dataloader.py` / `dataloader_for_paper_test.py`):

```
data_root/
├── train/
│   ├── images/   # BraTS_flair*.nii.gz  (4 modalities stacked)
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`val` is split from `train` automatically (`val_split=0.125`, `torch.manual_seed(42)`).
For the domain-shift / generalization experiment, point `data_root` in
`dataloader_ssa.py` to the unseen-scanner cohort.

## Quick Sanity Check

Verify the architecture, tensor shapes and parameter counts before training:

```bash
python test_new_architecture.py        # full test suite (coarse / fine / cascade / memory)
python models/cascade_both_new.py      # minimal forward pass on a random (2,4,160,160,128) volume
```

## Training

```bash
python train_cascade_new.py     # clean, self-contained reference implementation
python train_both_new.py        # paper version (heavy augmentation + sliding-window validation)
```

Default configuration (all constants are at the top of `train_cascade_new.py`):

| Item | Value |
| --- | --- |
| input / ROI size | `160×160×128` → `128³` ROI |
| coarse downsample factor | `4` (`use_tiny_coarse` switch for a lighter variant) |
| optimizer | AdamW, `lr=1e-4`, `weight_decay=1e-5` |
| scheduler | `CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-6)` |
| epochs / batch size | `300` / `1` |
| coarse loss | `0.5 × Dice` against `WT` derived from the GT label |
| fine loss | `0.5 × BCE + Dice + 0.5 × BoundaryLoss` (`bdLoss.py`) |
| deep supervision | weights `[1.0, 0.5, 0.25]` over `[final, 1/4, 1/8]` heads |
| checkpoints | best-Dice / best-Composite / every 20 epochs → `WEIGHTS_DIR` |

Change `DATA_ROOT`, `WEIGHTS_DIR` and `CUDA_VISIBLE_DEVICES` at the top of the script.

## Evaluation

```bash
python demo_ori.py                          # Dice / HD95 / CompositeScore on the held-out test set
python demo_ssa.py                          # generalization to the unseen-scanner cohort
python demo_ssa_var.py                      # same, with variant models
```

```bash
python demo_ori.py --max_samples 20         # quick smoke test on the first N cases
```

## Ablation Studies

```bash
# ROI-strategy × fine-backbone variants (Var_1..Var_4, see models/Var_*.py)
python demo_Var.py --variant Var_1
python demo_Var.py --variant Var_3          # bbox crop vs. centroid crop, FineNet vs. FineNet_New
python demo_Var.py --variant Var_1 --skip_flops

# Coarse-stage downsample factor {2, 4, 8} — reuses one trained checkpoint, no retraining
python ablation_downsample.py --model_weight /path/to/best.pth --dump_json ablation_results.json
```

## Efficiency Benchmark

`bench_fps.py` measures inference throughput of our cascade against the compared
baselines under the same protocol:

```bash
python bench_fps.py --model cascade_new --weights /path/to/best.pth --max_samples 50 --gpu 0
# choices: cascade_new | cascade_new_centroid | cascade_old | uxnet | swinunetr | glims | nnunet
```

The baseline code weights/paths are hard-coded near the bottom of the file; the
baseline repositories (`UXNet`, `Modules_GLMs`, `nnunetv2`) are **not** bundled — install
them from their upstream sources or delete those branches if you only need our model.

## Metrics

`cal_composite.py` / `cal_composite_for_paper.py` / `cal_composite_for_paper_strict.py`
implement the region-wise Dice and HD95 plus the **CompositeScore** used in the paper —
an accuracy term (Dice + HD95) and a reliability term (low-confidence voxel ratio,
boundary entropy on the WT shell, confidence distribution). Use
`cal_composite_for_paper_strict.py` to reproduce the strict, un-truncated HD95 definition
reported in the tables.

## Repository Layout

```
├── models/
│   ├── cascade_both_new.py        # ★ proposed cascade (coarse 3D + ROI crop + fine)
│   ├── coarseNet_3d.py            # Stage 1: lightweight 3D UNet (+ Tiny variant)
│   ├── fineNet_new.py             # Stage 2: ConvNeXt-3D U-Net w/ CBAM + deep supervision
│   ├── cascade_both_new_centroid.py  # centroid-crop ablation
│   ├── Var_1.py ... Var_4.py      # ablation variants
│   ├── c2.py, cbam.py             # CBAM attention modules
│   ├── uputil.py, util.py         # decoder / conv building blocks
│   └── UNet3D.py, fineNet.py, fuse2d.py, cascadeNet.py, cascade_both.py  # baselines & earlier versions
├── train_cascade_new.py           # reference training script
├── train_both_new.py              # paper training script
├── train.py, train_both*.py, train_fuse2D.py, train_Var_*.py
├── dataloader.py / dataloader_for_paper_test.py / dataloader_ssa.py
├── demo_ori.py, demo_ssa.py, demo_ssa_var.py, demo_Var.py   # evaluation
├── ablation_downsample.py, bench_fps.py, test_new_architecture.py
├── bdLoss.py, boundary.py         # boundary-aware loss & distance helpers
├── cal_composite*.py              # CompositeScore metric
└── crop_util.py                   # ROI crop helpers
```

## Known Limitations / Notes

- Data and weight paths are hard-coded per machine; there is no CLI for them yet.
- `CascadeBoth_New` pads/crops internally so any volume size works, but Stage 1 requires
  divisibility by 8 and Stage 2 by 32 (handled automatically).
- If the coarse stage finds no tumor above threshold, the ROI falls back to the volume center.
- Batch size 1 is used because of the 3D ROI patch; larger batches need more memory.
- Results were produced on **BraTS 2021**; other BraTS editions work with the same label mapping.

## License

Released for research purposes. BraTS data is distributed separately under the
[Medical Segmentation Decathlon / BraTS license](https://www.medicaldecathlon.com/) terms.

## Acknowledgements

Built on [MONAI](https://github.com/Project-MONAI/MONAI) and
[timm](https://github.com/huggingface/pytorch-image-models); the fine network follows the
ConvNeXt design of [Li et al.](https://arxiv.org/abs/2201.03545) and the attention module of CBAM.
