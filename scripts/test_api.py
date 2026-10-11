# scripts/test_api.py

import os
import sys
import json
from dotenv import load_dotenv

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

load_dotenv()

from src.api.app import create_app
from src.utils.image_utils import create_synthetic_satellite_image

def test_api_endpoints():
    print("==================================================")
    print("Testing CanopyReason API Endpoints (Phases 1-4)")
    print("==================================================")

    app = create_app()
    client = app.test_client()

    # 1. Test /health
    print("\n[1] Testing GET /health ...")
    res = client.get('/health')
    print("  Status code:", res.status_code)
    print("  Response:", res.get_json())
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"

    # 2. Test POST /api/analyze with sample image & policy
    print("\n[2] Testing POST /api/analyze ...")
    os.makedirs("data/raw_imagery", exist_ok=True)
    sample_img_path = os.path.abspath("data/raw_imagery/sample_commercial_zone.png")
    if not os.path.isfile(sample_img_path):
        print("  Generating synthetic satellite tile for test...")
        create_synthetic_satellite_image(sample_img_path)
    
    with open(sample_img_path, "rb") as f:
        data = {
            "image": (f, "sample_commercial_zone.png"),
            "policy_text": "Section 4.1: Commercial zones must achieve a minimum 20% canopy cover to mitigate urban heat islands.",
            "demographic_context": json.dumps({
                "poverty_rate_pct": 25.0,
                "unemployment_rate_pct": 5.5,
                "surface_temp_c": 31.0
            }),
            "latitude": "37.77",
            "longitude": "-122.42"
        }
        res = client.post('/api/analyze', data=data, content_type='multipart/form-data')

    print("  Status code:", res.status_code)
    res_json = res.get_json()
    assert res.status_code == 200, f"Error: {res_json}"
    
    print("  Response Keys:", list(res_json.keys()))
    print("  Vision metrics:", res_json.get("vision_metrics"))
    print("  Biomass output:", res_json.get("biomass_output"))
    print("  Equity output:", res_json.get("equity_output"))
    print("  Reasoning output:", res_json.get("reasoning_output"))
    print("  GraphRAG analysis:", res_json.get("graphrag_analysis"))
    print("  XAI rationale:", res_json.get("xai_rationale"))
    print("  Counterfactual analysis:", res_json.get("counterfactual_analysis"))
    print("  XAI Map URL:", res_json.get("xai_map_url"))

    # Assert Phase 4 additions
    assert "graphrag_analysis" in res_json, "Missing graphrag_analysis in response"
    assert "xai_rationale" in res_json, "Missing xai_rationale in response"
    assert "counterfactual_analysis" in res_json, "Missing counterfactual_analysis in response"
    
    graphrag = res_json.get("graphrag_analysis", {})
    assert "binding_verdict" in graphrag or "extracted_entities" in graphrag, "GraphRAG missing required fields"
    
    xai = res_json.get("xai_rationale", {})
    assert "scenario_a_analysis" in xai or "scenario_b_analysis" in xai, "XAI rationale missing scenario analysis"

    # 3. Test GET /api/maps/<filename>
    map_url = res_json.get("xai_map_url")
    print(f"\n[3] Testing GET {map_url} ...")
    map_res = client.get(map_url)
    print("  Status code:", map_res.status_code)
    print("  Content-Type:", map_res.headers.get("Content-Type"))
    print("  Payload bytes:", len(map_res.data))
    assert map_res.status_code == 200, f"Expected 200, got {map_res.status_code}"
    assert map_res.headers.get("Content-Type") == "image/png"

    print("\n==================================================")
    print("[SUCCESS] All Phase 1-4 API endpoints validated successfully!")
    print("==================================================")

if __name__ == "__main__":
    test_api_endpoints()
