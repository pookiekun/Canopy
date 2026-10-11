# src/explainability/visual_maps.py

import os
import uuid
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Prevents GUI popup errors during server execution
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Union
from src.explainability.counterfactual import simulate_canopy_removal

def generate_explainability_map(
    image_path: str,
    mask: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    save_dir: str = "data/processed",
    counterfactual_image: Optional[Union[Image.Image, np.ndarray]] = None
) -> str:
    """
    Generates a high-fidelity 4-panel explainability map for urban canopy verification (Phase 4):
      1. Optical Satellite Imagery
      2. XAI Grounded Canopy Mask Overlay
      3. Canopy Spatial Density Heatmap
      4. Counterfactual Impact: Cleared Canopy & Thermal Spike Simulation
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load original image and ensure shape compatibility
    original = Image.open(image_path).convert("RGB")
    orig_np = np.array(original)
    
    if mask.shape != orig_np.shape[:2]:
        mask_pil = Image.fromarray(mask.astype(np.uint8)).resize(
            (orig_np.shape[1], orig_np.shape[0]),
            resample=Image.Resampling.NEAREST
        )
        mask = np.array(mask_pil)

    # 1. Overlay (Forest Green RGB: [34, 139, 34])
    overlay = orig_np.copy()
    overlay[mask == 1] = [34, 139, 34]
    alpha = 0.52
    blended = (alpha * overlay + (1.0 - alpha) * orig_np).astype(np.uint8)

    # 2. Density Heatmap from mask convolution/blur
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(mask.astype(float), sigma=8.0)
    density = (density - density.min()) / (density.max() - density.min() + 1e-6)

    # 3. Counterfactual Image generation / formatting
    cf_data = (metadata or {}).get("counterfactual") or {}
    temp_rise = cf_data.get("simulated_surface_temp_rise_c", round(float((metadata or {}).get("coverage_pct", 0.0)) * 0.12, 2))
    utci_rise = cf_data.get("simulated_utci_thermal_stress_rise_c", round(float((metadata or {}).get("coverage_pct", 0.0)) * 0.08, 2))
    fragility = cf_data.get("ecological_fragility", "High" if float((metadata or {}).get("coverage_pct", 0.0)) > 20 else "Moderate")

    if counterfactual_image is not None:
        if isinstance(counterfactual_image, Image.Image):
            cf_np = np.array(counterfactual_image.convert("RGB"))
        else:
            cf_np = np.array(counterfactual_image)
    else:
        cf_pil, _ = simulate_canopy_removal(orig_np, mask)
        cf_np = np.array(cf_pil)

    # Add red thermal stress warning overlay on cleared canopy areas
    cf_display = cf_np.copy()
    thermal_tint = np.array([239, 68, 68], dtype=np.uint8)  # Bright amber-red
    cf_mask_area = (mask == 1)
    if np.any(cf_mask_area):
        cf_display[cf_mask_area] = (0.65 * cf_display[cf_mask_area] + 0.35 * thermal_tint).astype(np.uint8)

    # 4. Create 4-Panel Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), facecolor="#0e1117")
    plt.subplots_adjust(wspace=0.15)

    # Panel 1: Original
    axes[0].imshow(orig_np)
    axes[0].set_title("1. Optical Satellite Imagery", fontsize=13, fontweight='bold', color="#ffffff", pad=12)
    axes[0].axis('off')

    # Panel 2: XAI Overlay
    axes[1].imshow(blended)
    cov_text = f"{metadata.get('coverage_pct', 'N/A')}%" if metadata else "Segmented"
    axes[1].set_title(f"2. XAI Canopy Grounding ({cov_text})", fontsize=13, fontweight='bold', color="#4ade80", pad=12)
    axes[1].axis('off')

    # Panel 3: Spatial Density Heatmap
    axes[2].imshow(density, cmap="magma")
    axes[2].set_title("3. Canopy Density Heatmap", fontsize=13, fontweight='bold', color="#fbbf24", pad=12)
    axes[2].axis('off')

    # Panel 4: Counterfactual Impact
    axes[3].imshow(cf_display)
    axes[3].set_title(f"4. Counterfactual Impact (+{temp_rise}°C)", fontsize=13, fontweight='bold', color="#f87171", pad=12)
    axes[3].axis('off')

    # Add annotation box on Panel 4
    annotation_text = f"Δ Temp: +{temp_rise}°C\nΔ UTCI: +{utci_rise}°C\nFragility: {fragility}"
    axes[3].text(
        0.05, 0.08, annotation_text,
        transform=axes[3].transAxes,
        fontsize=10,
        fontweight='bold',
        color="#ffffff",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#ef4444", alpha=0.9)
    )

    # Add suptitle if metadata is present
    if metadata:
        verdict = "COMPLIANT" if metadata.get("is_compliant") else "DEFICIT / NON-COMPLIANT"
        verdict_color = "#22c55e" if metadata.get("is_compliant") else "#ef4444"
        fig.suptitle(
            f"CanopyReason Inspection Report  |  Status: {verdict}  |  TES: {metadata.get('tree_equity_score', 'N/A')}/100  |  XAI 4-Panel Grounding",
            fontsize=15,
            fontweight='bold',
            color=verdict_color,
            y=0.98
        )

    plt.tight_layout()

    # Unique output filename
    base_name = os.path.basename(image_path).split('.')[0]
    unique_id = str(uuid.uuid4())[:8]
    output_filename = f"xai_{base_name}_{unique_id}.png"
    output_path = os.path.abspath(os.path.join(save_dir, output_filename))

    plt.savefig(output_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

    print(f"Explainability map successfully generated (4-panel): {output_path}")
    return output_path

if __name__ == "__main__":
    dummy_mask = np.zeros((512, 512), dtype=np.uint8)
    dummy_mask[120:380, 140:420] = 1
    mock_img = Image.new("RGB", (512, 512), color=(140, 140, 130))
    test_path = "data/temp_test.png"
    os.makedirs("data", exist_ok=True)
    mock_img.save(test_path)
    res = generate_explainability_map(
        test_path,
        dummy_mask,
        metadata={"coverage_pct": 21.4, "is_compliant": True, "tree_equity_score": 82}
    )
    print("Saved:", res)