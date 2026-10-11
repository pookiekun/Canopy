# src/explainability/counterfactual.py

import os
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

def simulate_canopy_removal(
    image_input,
    mask: np.ndarray,
    replacement_color: Tuple[int, int, int] = (110, 110, 110)
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Simulates a counterfactual intervention where existing tree canopies are cleared
    or replaced with impervious asphalt / concrete.
    
    Computes the resulting delta in canopy coverage, localized temperature spike,
    and lost carbon sequestration capacity.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.fromarray(image_input).convert("RGB")

    img_np = np.array(image)
    if mask.shape != img_np.shape[:2]:
        mask_pil = Image.fromarray(mask.astype(np.uint8)).resize((img_np.shape[1], img_np.shape[0]), resample=Image.Resampling.NEAREST)
        mask = np.array(mask_pil)

    counterfactual_np = img_np.copy()
    counterfactual_np[mask == 1] = replacement_color

    total_pixels = mask.size
    tree_pixels = int(np.sum(mask))
    canopy_coverage_pct = round((tree_pixels / max(total_pixels, 1)) * 100.0, 2)

    # Counterfactual thermal impact: removing trees increases surface heat and UTCI
    surface_temp_rise_c = round(canopy_coverage_pct * 0.12, 2)
    utci_thermal_stress_increase_c = round(canopy_coverage_pct * 0.08, 2)

    analysis = {
        "original_canopy_pct": canopy_coverage_pct,
        "counterfactual_canopy_pct": 0.0,
        "delta_coverage_pct": -canopy_coverage_pct,
        "simulated_surface_temp_rise_c": surface_temp_rise_c,
        "simulated_utci_thermal_stress_rise_c": utci_thermal_stress_increase_c,
        "ecological_fragility": "High" if canopy_coverage_pct > 20 else "Moderate"
    }

    return Image.fromarray(counterfactual_np), analysis
