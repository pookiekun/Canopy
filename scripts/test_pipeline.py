# scripts/test_pipeline.py

import os
import sys
import json
from dotenv import load_dotenv

# Ensure safe output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

from src.utils.image_utils import create_synthetic_satellite_image
from src.vision.segmentor import CanopySegmenter
from src.reasoning.graph import build_reasoning_engine
from src.explainability.visual_maps import generate_explainability_map
from src.explainability.counterfactual import simulate_canopy_removal

def run_e2e_pipeline_test():
    print("==================================================")
    print("CanopyReason MVP: End-to-End Pipeline Test")
    print("==================================================")
    
    # 1. Prepare sample imagery
    os.makedirs("data/raw_imagery", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    sample_image_path = os.path.abspath("data/raw_imagery/sample_commercial_zone.png")
    
    print("\n[Step 1] Preparing high-resolution synthetic satellite tile...")
    create_synthetic_satellite_image(sample_image_path)
    print(f"  Created sample image at: {sample_image_path}")

    # 2. Vision Segmentation
    print("\n[Step 2] Running SegFormer Vision Inference...")
    segmenter = CanopySegmenter()
    coverage_pct, tree_mask = segmenter.analyze_canopy(sample_image_path)
    print(f"  Identified Canopy Coverage: {coverage_pct}%")

    # 3. Counterfactual Spatial Reasoning
    print("\n[Step 3] Computing Counterfactual Perturbation...")
    _, counterfactual_res = simulate_canopy_removal(sample_image_path, tree_mask)
    print(f"  Counterfactual Surface Temp Rise: +{counterfactual_res['simulated_surface_temp_rise_c']} deg C")

    # 4. Multi-Agent Reasoning Graph
    print("\n[Step 4] Invoking LangGraph Multi-Agent Engine...")
    print("  Graph Nodes: evaluate_compliance -> estimate_biomass -> assess_equity -> simulate_afforestation")
    engine = build_reasoning_engine()
    
    state_input = {
        "current_coverage": coverage_pct,
        "policy_text": "Section 4.1: All commercial zones must maintain a minimum tree canopy coverage of 20% to mitigate urban heat islands.",
        "demographic_context": {
            "poverty_rate_pct": 28.5,
            "unemployment_rate_pct": 7.8,
            "minority_pct": 52.0,
            "dependent_ratio_pct": 34.0,
            "surface_temp_c": 31.8,
            "urban_avg_temp_c": 28.5
        },
        "latitude": 37.7749,
        "longitude": -122.4194,
        "total_area_sq_m": 12000.0  # 1.2 hectares
    }
    
    result = engine.invoke(state_input)

    # 5. Explainability Map Generation
    print("\n[Step 5] Generating 4-Panel XAI Grounding Map...")
    xai_path = generate_explainability_map(
        sample_image_path,
        tree_mask,
        metadata={
            "coverage_pct": coverage_pct,
            "is_compliant": result.get("is_compliant"),
            "tree_equity_score": result.get("tree_equity_score")
        }
    )
    print(f"  Saved XAI Grounding Map: {xai_path}")

    # 6. Display Synthesized MVP Report
    print("\n==================================================")
    print("CanopyReason Synthesized Assessment Report")
    print("==================================================")
    
    report = {
        "vision_metrics": {
            "canopy_coverage_pct": coverage_pct,
            "total_area_sq_m": state_input["total_area_sq_m"]
        },
        "biomass_and_carbon": result.get("biomass_metrics"),
        "tree_equity_and_microclimate": {
            "tree_equity_score": result.get("tree_equity_score"),
            "priority_index": result.get("priority_index"),
            "equity_details": result.get("equity_metrics"),
            "microclimate_details": result.get("microclimate_metrics")
        },
        "policy_and_intervention": {
            "target_coverage_pct": result.get("target_coverage"),
            "is_compliant": result.get("is_compliant"),
            "rationale": result.get("rationale"),
            "deficit_pct": result.get("deficit"),
            "intervention_plan": result.get("intervention_plan")
        },
        "counterfactual_analysis": counterfactual_res,
        "xai_map_path": xai_path
    }
    
    print(json.dumps(report, indent=2))
    print("\n[SUCCESS] End-to-end pipeline test passed successfully!")

if __name__ == "__main__":
    run_e2e_pipeline_test()