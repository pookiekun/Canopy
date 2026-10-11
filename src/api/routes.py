# src/api/routes.py

import os
import json
import uuid
from flask import Blueprint, request, jsonify, send_file
from src.vision.segmentor import CanopySegmenter
from src.reasoning.graph import build_reasoning_engine
from src.explainability.visual_maps import generate_explainability_map
from src.explainability.counterfactual import simulate_canopy_removal
from src.utils.image_utils import create_synthetic_satellite_image

api_blueprint = Blueprint('api', __name__)

# Lazy initialization placeholders
_segmenter = None
_reasoning_engine = None

def get_segmenter():
    global _segmenter
    if _segmenter is None:
        print("Initializing Vision Model (SegFormer)...")
        _segmenter = CanopySegmenter()
    return _segmenter

def get_reasoning_engine():
    global _reasoning_engine
    if _reasoning_engine is None:
        print("Initializing LangGraph Reasoning Engine...")
        _reasoning_engine = build_reasoning_engine()
    return _reasoning_engine

@api_blueprint.route('/analyze', methods=['POST'])
@api_blueprint.route('/assess', methods=['POST'])
def analyze_canopy():
    """
    Main endpoint for urban tree cover assessment (Phases 1 - 4).
    Accepts multipart/form-data or application/json.
    """
    os.makedirs("data/raw_imagery", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    image_path = None
    policy_text = None
    demographic_context = {}
    lat = None
    lon = None
    total_area_sq_m = 10000.0  # 1 hectare default

    # 1. Parse Multipart or JSON request
    if request.is_json:
        data = request.get_json() or {}
        policy_text = data.get("policy_text") or data.get("policy")
        image_path = data.get("image_path")
        demographic_context = data.get("demographic_context", {})
        lat = data.get("latitude")
        lon = data.get("longitude")
        total_area_sq_m = float(data.get("total_area_sq_m", 10000.0))
    else:
        # Form / Multipart
        policy_text = request.form.get("policy_text") or request.form.get("policy")
        demo_str = request.form.get("demographic_context")
        if demo_str:
            try:
                demographic_context = json.loads(demo_str)
            except Exception:
                demographic_context = {}
                
        if "latitude" in request.form:
            try: lat = float(request.form.get("latitude"))
            except ValueError: pass
        if "longitude" in request.form:
            try: lon = float(request.form.get("longitude"))
            except ValueError: pass
        if "total_area_sq_m" in request.form:
            try: total_area_sq_m = float(request.form.get("total_area_sq_m"))
            except ValueError: pass

        if 'image' in request.files and request.files['image'].filename:
            image_file = request.files['image']
            ext = os.path.splitext(image_file.filename)[1] or ".png"
            filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
            image_path = os.path.abspath(os.path.join("data/raw_imagery", filename))
            image_file.save(image_path)

    # Defaults if missing
    if not policy_text:
        policy_text = "Section 4.1: All commercial zones must maintain a minimum tree canopy coverage of 20% to mitigate urban heat islands."

    # If no image was provided, use or generate a sample satellite tile
    if not image_path or not os.path.isfile(image_path):
        sample_path = os.path.abspath("data/raw_imagery/sample_satellite_tile.png")
        if not os.path.isfile(sample_path):
            create_synthetic_satellite_image(sample_path)
        image_path = sample_path

    try:
        segmenter = get_segmenter()
        reasoning_engine = get_reasoning_engine()

        # 2. Vision Inference
        coverage_percentage, tree_mask = segmenter.analyze_canopy(image_path)

        # 3. Counterfactual Spatial Reasoning (Phase 4 Simulation)
        cf_img, counterfactual_res = simulate_canopy_removal(image_path, tree_mask)

        # 4. Multi-Agent Reasoning Execution (LangGraph Pipeline)
        state_input = {
            "current_coverage": coverage_percentage,
            "policy_text": policy_text,
            "demographic_context": demographic_context,
            "latitude": lat,
            "longitude": lon,
            "total_area_sq_m": total_area_sq_m,
            "counterfactual_data": counterfactual_res
        }

        result_state = reasoning_engine.invoke(state_input)

        # 5. Generate Multi-Panel XAI Map (Upgraded to 4 Panels in Phase 4)
        xai_map_path = generate_explainability_map(
            image_path,
            tree_mask,
            metadata={
                "coverage_pct": coverage_percentage,
                "is_compliant": result_state.get("is_compliant", False),
                "tree_equity_score": result_state.get("tree_equity_score", 50),
                "counterfactual": counterfactual_res
            },
            counterfactual_image=cf_img
        )

        map_filename = os.path.basename(xai_map_path)

        # 6. Assemble Comprehensive Response Payload (Phases 1-4)
        response_payload = {
            "vision_metrics": {
                "current_canopy_coverage_percent": coverage_percentage,
                "total_analyzed_area_sq_m": total_area_sq_m
            },
            "biomass_output": result_state.get("biomass_metrics", {
                "mean_canopy_height_m": result_state.get("canopy_height_m"),
                "total_agb_kg": result_state.get("estimated_agb_kg"),
                "total_carbon_stored_kg": result_state.get("carbon_stored_kg")
            }),
            "equity_output": result_state.get("equity_metrics", {
                "tree_equity_score": result_state.get("tree_equity_score"),
                "priority_index": result_state.get("priority_index")
            }),
            "microclimate_simulation": result_state.get("microclimate_metrics", {}),
            "reasoning_output": {
                "target_coverage_percent": result_state.get("target_coverage"),
                "is_compliant": result_state.get("is_compliant"),
                "compliance_rationale": result_state.get("rationale"),
                "afforestation_deficit_percent": result_state.get("deficit"),
                "intervention_plan": result_state.get("intervention_plan")
            },
            "counterfactual_analysis": counterfactual_res,
            "graphrag_analysis": result_state.get("graphrag_output", {}),
            "xai_rationale": result_state.get("xai_rationale", {}),
            "xai_map_url": f"/api/maps/{map_filename}"
        }

        return jsonify(response_payload), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@api_blueprint.route('/maps/<filename>', methods=['GET'])
def get_map(filename):
    """Serves the generated XAI overlay maps to users and frontend dashboards."""
    map_path = os.path.abspath(os.path.join("data", "processed", filename))
    if os.path.isfile(map_path):
        return send_file(map_path, mimetype='image/png')
    return jsonify({"error": "Map file not found"}), 404