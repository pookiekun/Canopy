# src/utils/equity_score.py

from typing import Dict, Any, Optional

BIOME_TARGETS = {
    "forest": 40.0,
    "grassland": 20.0,
    "mediterranean": 20.0,
    "desert": 15.0
}

def calculate_canopy_gap(
    current_coverage_pct: float,
    biome_type: str = "forest",
    building_density: float = 0.5
) -> float:
    """
    Calculates the tree canopy gap = density-adjusted biome target - existing coverage.
    """
    base_target = BIOME_TARGETS.get(biome_type.lower(), 40.0)
    # Density adjustment: high building density (> 0.7) reduces realistic achievable canopy space
    density_factor = max(0.6, 1.0 - (0.35 * min(max(building_density, 0.0), 1.0)))
    adjusted_target = round(base_target * density_factor, 1)
    
    gap = max(0.0, adjusted_target - current_coverage_pct)
    return round(gap, 2)

def calculate_heat_disparity(local_temp_c: float, urban_avg_temp_c: float = 28.5) -> float:
    """
    Calculates surface temperature disparity relative to the metropolitan average.
    Positive values indicate an amplified Urban Heat Island (UHI) effect.
    """
    return round(local_temp_c - urban_avg_temp_c, 2)

def calculate_priority_index(
    poverty_pct: float = 20.0,
    unemployment_pct: float = 6.0,
    minority_pct: float = 40.0,
    dependent_ratio: float = 30.0,
    heat_disparity_c: float = 2.0
) -> float:
    """
    Calculates the socio-demographic Priority Index (E) normalized between 0.1 and 1.0.
    Higher values signify greater systemic vulnerability and acute need for ecological investment.
    """
    # Normalize indicators against standard municipal percentiles
    norm_poverty = min(poverty_pct / 60.0, 1.0)
    norm_unemp = min(unemployment_pct / 20.0, 1.0)
    norm_minority = min(minority_pct / 100.0, 1.0)
    norm_dependents = min(dependent_ratio / 50.0, 1.0)
    norm_heat = min(max(heat_disparity_c, 0.0) / 6.0, 1.0)

    # Multi-variable weighted sum
    composite = (
        0.30 * norm_poverty +
        0.25 * norm_heat +
        0.20 * norm_minority +
        0.15 * norm_unemp +
        0.10 * norm_dependents
    )

    priority_e = 0.1 + (0.9 * composite)
    return round(min(max(priority_e, 0.1), 1.0), 3)

def calculate_tree_equity_score(
    canopy_gap: float,
    priority_index: float,
    max_expected_gap: float = 35.0
) -> int:
    """
    Derives the Tree Equity Score (0 to 100) based on the American Forests methodology.
    A score of 100 represents full canopy equity; lower scores denote urgent intervention priority.
    """
    gap_ratio = min(canopy_gap / max_expected_gap, 1.0)
    # Disparity penalty
    penalty = gap_ratio * priority_index * 100.0
    score = int(round(max(0.0, min(100.0, 100.0 - penalty))))
    return score

def assess_tree_equity(
    current_coverage_pct: float,
    demographic_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Full pipeline evaluator for Tree Equity Score and demographic prioritization.
    """
    demo = demographic_data or {}
    biome = demo.get("biome", "forest")
    density = demo.get("building_density", 0.5)
    local_temp = demo.get("surface_temp_c", 30.5)
    urban_avg_temp = demo.get("urban_avg_temp_c", 28.5)
    
    poverty = demo.get("poverty_rate_pct", 22.0)
    unemp = demo.get("unemployment_rate_pct", 6.5)
    minority = demo.get("minority_pct", 45.0)
    dependents = demo.get("dependent_ratio_pct", 32.0)

    gap = calculate_canopy_gap(current_coverage_pct, biome, density)
    heat_disp = calculate_heat_disparity(local_temp, urban_avg_temp)
    priority_e = calculate_priority_index(poverty, unemp, minority, dependents, heat_disp)
    tes = calculate_tree_equity_score(gap, priority_e)

    priority_category = "Urgent Intervention Required" if tes < 50 else (
        "Moderate Priority" if tes < 75 else "Adequate Canopy Equity"
    )

    return {
        "tree_equity_score": tes,
        "priority_index": priority_e,
        "canopy_gap_pct": gap,
        "heat_disparity_c": heat_disp,
        "equity_category": priority_category,
        "methodology": "American Forests Tree Equity Score Framework"
    }
