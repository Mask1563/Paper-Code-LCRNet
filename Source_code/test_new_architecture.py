"""Cascade architecture test suite."""
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cascade_both_new import CascadeBoth_New
from models.coarseNet_3d import CoarseNet3D, CoarseNet3D_Tiny
from models.fineNet_new import FineNet_New


def test_coarse_net():
    """Test CoarseNet3D."""
    print("\n" + "="*60)
    print("Testing CoarseNet3D...")
    print("="*60)
    
    model = CoarseNet3D(in_channels=4, base_channels=8)
    x = torch.randn(2, 4, 40, 40, 32)  # 4x downsampled input
    
    print(f"Input shape: {x.shape}")
    out = model(x)
    print(f"Output shape: {out.shape}")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    assert out.shape == (2, 1, 40, 40, 32), f"Expected (2, 1, 40, 40, 32), got {out.shape}"
    print("✓ CoarseNet3D test passed!")
    
    model_tiny = CoarseNet3D_Tiny(in_channels=4, base_channels=4)
    out_tiny = model_tiny(x)
    params_tiny = sum(p.numel() for p in model_tiny.parameters()) / 1e6
    print(f"\nTiny version parameters: {params_tiny:.2f}M")
    print(f"Parameter reduction: {(1 - params_tiny/params)*100:.1f}%")
    print("✓ CoarseNet3D_Tiny test passed!")


def test_fine_net():
    """Test FineNet_New."""
    print("\n" + "="*60)
    print("Testing FineNet_New...")
    print("="*60)
    
    model = FineNet_New(num_classes=3, use_deep_supervision=True)
    x = torch.randn(2, 5, 128, 128, 128)  # 4 channels + 1 coarse mask
    
    print(f"Input shape: {x.shape}")
    
    model.train()
    out_train = model(x)
    print(f"Training mode output (deep supervision):")
    for i, out in enumerate(out_train):
        print(f"  Level {i}: {out.shape}")
    
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f"Inference mode output: {out_eval.shape}")
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
    
    assert isinstance(out_train, list), "Training mode should return list"
    assert len(out_train) == 3, "Should have 3 outputs for deep supervision"
    assert out_eval.shape == (2, 3, 128, 128, 128), f"Expected (2, 3, 128, 128, 128), got {out_eval.shape}"
    print("✓ FineNet_New test passed!")


def test_cascade_both_new():
    """Test full CascadeBoth_New."""
    print("\n" + "="*60)
    print("Testing CascadeBoth_New (Full Pipeline)...")
    print("="*60)
    
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
    coarse_mask, fine_output = model(x)
    
    print(f"\nCoarse mask shape: {coarse_mask.shape}")
    print(f"Fine output (deep supervision):")
    for i, out in enumerate(fine_output):
        print(f"  Level {i}: {out.shape}")
    
    model.eval()
    with torch.no_grad():
        coarse_mask_eval, fine_output_eval = model(x)
    
    print(f"\nInference mode:")
    print(f"  Coarse mask: {coarse_mask_eval.shape}")
    print(f"  Fine output: {fine_output_eval.shape}")
    
    coarse_params = sum(p.numel() for p in model.coarse_unet.parameters()) / 1e6
    fine_params = sum(p.numel() for p in model.fine_unet.parameters()) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    print(f"\nParameter breakdown:")
    print(f"  Coarse Network: {coarse_params:.2f}M ({coarse_params/total_params*100:.1f}%)")
    print(f"  Fine Network: {fine_params:.2f}M ({fine_params/total_params*100:.1f}%)")
    print(f"  Total: {total_params:.2f}M")
    
    assert coarse_mask.shape == (2, 1, 160, 160, 128), f"Coarse mask shape mismatch"
    assert isinstance(fine_output, list), "Training mode should return list"
    assert fine_output_eval.shape == (2, 3, 160, 160, 128), f"Fine output shape mismatch"
    
    print("✓ CascadeBoth_New test passed!")


def test_with_different_configs():
    """Test different configurations."""
    print("\n" + "="*60)
    print("Testing Different Configurations...")
    print("="*60)
    
    configs = [
        {"use_tiny_coarse": False, "use_deep_supervision": True, "name": "Standard + DS"},
        {"use_tiny_coarse": True, "use_deep_supervision": True, "name": "Tiny + DS"},
        {"use_tiny_coarse": False, "use_deep_supervision": False, "name": "Standard No DS"},
        {"use_tiny_coarse": True, "use_deep_supervision": False, "name": "Tiny No DS"},
    ]
    
    x = torch.randn(1, 4, 160, 160, 128)
    
    print(f"{'Configuration':<25} {'Total Params':<15} {'Coarse Params':<15} {'Fine Params':<15}")
    print("-" * 70)
    
    for config in configs:
        model = CascadeBoth_New(
            num_out_classes=3,
            input_channels=4,
            use_tiny_coarse=config["use_tiny_coarse"],
            use_deep_supervision=config["use_deep_supervision"]
        )
        
        coarse_params = sum(p.numel() for p in model.coarse_unet.parameters()) / 1e6
        fine_params = sum(p.numel() for p in model.fine_unet.parameters()) / 1e6
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        print(f"{config['name']:<25} {total_params:>6.2f}M        {coarse_params:>6.2f}M        {fine_params:>6.2f}M")
        
        model.eval()
        with torch.no_grad():
            _, _ = model(x)
    
    print("✓ All configurations work correctly!")


def test_memory_usage():
    """Estimate memory usage."""
    print("\n" + "="*60)
    print("Memory Usage Estimation...")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = CascadeBoth_New(
            num_out_classes=3,
            input_channels=4,
            use_deep_supervision=True
        ).to(device)
        
        # Use batch_size=2 to avoid BatchNorm issues in training mode
        x = torch.randn(2, 4, 160, 160, 128).to(device)
        
        model.train()
        coarse, fine = model(x)
        
        if isinstance(fine, list):
            loss = sum(f.sum() for f in fine)
        else:
            loss = fine.sum()
        
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory usage (batch_size=2): {peak_memory:.2f} GB")
        print(f"Estimated for batch_size=1: {peak_memory*0.6:.2f} GB")
        
        torch.cuda.empty_cache()
        print("✓ Memory test completed!")
    else:
        print("CUDA not available, skipping GPU memory test")
        print("✓ Skipped (CPU mode)")


def compare_with_original():
    """Compare parameter counts with original architecture."""
    print("\n" + "="*60)
    print("Comparison with Original Architecture...")
    print("="*60)
    
    try:
        from models.cascade_both import CascadeBoth
        
        model_old = CascadeBoth(num_out_classes=3, input_channels=4)
        params_old = sum(p.numel() for p in model_old.parameters()) / 1e6
        
        model_new = CascadeBoth_New(num_out_classes=3, input_channels=4)
        params_new = sum(p.numel() for p in model_new.parameters()) / 1e6
        
        print(f"Original architecture: {params_old:.2f}M parameters")
        print(f"New architecture: {params_new:.2f}M parameters")
        print(f"Difference: {params_new - params_old:+.2f}M ({(params_new/params_old - 1)*100:+.1f}%)")
        
        if params_new < params_old * 1.2:
            print("✓ New architecture maintains similar parameter count!")
        else:
            print("⚠ New architecture has significantly more parameters")
    
    except ImportError:
        print("Original CascadeBoth not available for comparison")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print(" "*20 + "CASCADE ARCHITECTURE TEST SUITE")
    print("="*80)
    
    try:
        test_coarse_net()
        test_fine_net()
        test_cascade_both_new()
        test_with_different_configs()
        test_memory_usage()
        compare_with_original()
        
        print("\n" + "="*80)
        print(" "*25 + "ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nYou can now use the new architecture for training:")
        print("  python train_cascade_new.py")
        print("\n")
    
    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
