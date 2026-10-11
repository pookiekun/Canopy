# CanopyReason: Multimodal Urban Canopy AI Reasoning Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-blueviolet.svg)](https://ai.google.dev/)
[![SegFormer](https://img.shields.io/badge/vision-SegFormer--B0-green.svg)](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CanopyReason** is an advanced multimodal AI system that autonomously assesses urban tree canopy coverage, evaluates municipal zoning and environmental policy compliance, computes 3D above-ground biomass, evaluates socio-ecological tree equity, and synthesizes actionable, legally-grounded afforestation interventions with counterfactual explainability (XAI).

---

## Table of Contents

- [Key Features](#-key-features-across-phases-1--4)
- [Architecture](#-architecture--multi-agent-langgraph-pipeline)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Testing](#-testing--validation)
- [License](#-license)

---

## Key Features Across Phases 1 – 4

### Phase 1: High-Resolution Vision & Baseline Policy Evaluation
- **Optical Satellite Segmentation:** Pretrained SegFormer (`nvidia/segformer-b0-finetuned-ade-512-512`) for semantic canopy crown extraction from aerial/satellite tiles.
- **Municipal Policy Parsing:** Gemini-powered LLM extraction of mandated canopy coverage percentages and compliance rationale with deterministic fallback.

### Phase 2: 3D Morphology & Urban-Corrected Allometric Biomass
- **Canopy Height Model (CHM):** Infers localized vertical canopy structure and mean height in meters, simulating the ETH Zurich Global CHM (GEDI LiDAR + Sentinel-2 fusion).
- **Urban Biomass Estimation:** Chave et al. pantropical allometric equations calibrated with a 0.8× urban stress correction factor for Above-Ground Biomass (AGB, kg) and carbon stock (t CO₂e).

### Phase 3: Socio-Ecological Equity & Microclimate Simulation
- **American Forests Tree Equity Score (TES):** Programmatic Priority Index incorporating local poverty rates, unemployment, surface temperature, and localized heat disparity.
- **Microclimate & UTCI Simulation:** ENVI-met calibrated computation of Universal Thermal Climate Index (UTCI) summer cooling mitigation and additional cooling potential.
- **Afforestation Intervention Synthesis:** Actionable, equity-first planting and crown preservation strategies.

### Phase 4: Agentic Reasoning & Counterfactual Explainability (XAI)
- **GraphRAG Policy Graph Traverser:** Multi-hop legal knowledge graph traversal across municipal zoning codes. Identifies intersecting spatial boundaries (G), capacity constraints (C), temporal windows (T), and environmental mandates (E) to produce binding determinations with statutory corrective actions.
- **Counterfactual Spatial Reasoning (XAI):**
  - **Scenario A (Complete Canopy Clearance):** Computes acute surface temperature spike, UTCI thermal stress increase, annual carbon sink loss, and resultant statutory sanctions.
  - **Scenario B (Full Statutory Compliance):** Projects unlocked biomass, additional carbon sequestration, and Tree Equity Score elevation to parity.
  - **Spatial Attention Grounding:** Multi-head attention justification confirming viewpoint-invariant crown identification.
- **4-Panel XAI Grounding Map:** High-resolution visual artifact:
  1. Optical Aerial/Satellite Imagery
  2. XAI Segmented Canopy Mask Overlay
  3. Spatial Density Heatmap (Gaussian-blurred continuous intensity)
  4. Counterfactual Clearance Impact (thermal stress overlay with temperature annotations)

---

## Architecture & Multi-Agent LangGraph Pipeline

```
[Optical Satellite Tile] + [Municipal Policy] + [Demographic Context]
                           │
                           ▼
          ┌───────────────────────────────────┐
          │     CanopySegmenter (SegFormer)   │  ──► Coverage % & Tree Mask
          └───────────────────────────────────┘
                           │
                           ▼
          ┌───────────────────────────────────┐
          │   simulate_canopy_removal (XAI)   │  ──► Counterfactual Metrics
          └───────────────────────────────────┘
                           │
                           ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                  LangGraph Multi-Agent Engine                │
 │                                                              │
 │  1. evaluate_compliance      ──► Baseline Policy Check       │
 │  2. estimate_biomass         ──► 3D CHM + AGB Carbon         │
 │  3. assess_equity            ──► Tree Equity Score + UTCI    │
 │  4. simulate_afforestation   ──► Multimodal Intervention     │
 │  5. traverse_policy_graph    ──► GraphRAG Multi-Hop Legal KG │
 │  6. generate_xai_rationale   ──► Counterfactual Spatial XAI  │
 └──────────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌───────────────────────────────────┐
          │   generate_explainability_map     │  ──► 4-Panel XAI Grounding Map
          └───────────────────────────────────┘
                           │
                           ▼
      [JSON Response Payload] + [Interactive Web UI Dashboard]
```

---

## Project Structure

```
CanopyReason/
├── run.py                          # Application entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore
│
├── src/
│   ├── api/
│   │   ├── app.py                  # Flask app factory
│   │   └── routes.py               # API endpoints (/analyze, /assess, /maps)
│   ├── reasoning/
│   │   ├── graph.py                # LangGraph 6-node multi-agent pipeline
│   │   └── prompts.py              # LLM prompt templates (Phases 1–4)
│   ├── vision/
│   │   ├── segmentor.py            # SegFormer canopy segmentation model
│   │   └── height_model.py         # Canopy Height Model (CHM simulation)
│   ├── explainability/
│   │   ├── visual_maps.py          # 4-panel XAI grounding map generator
│   │   ├── counterfactual.py       # Canopy removal simulation
│   │   └── rationale.py            # Counterfactual narrative generator
│   └── utils/
│       ├── biomass.py              # Chave allometric biomass equations
│       ├── equity_score.py         # American Forests Tree Equity Score
│       ├── microclimate.py         # UTCI microclimate simulation
│       └── image_utils.py          # Image loading & synthetic tile generator
│
├── scripts/
│   ├── test_api.py                 # End-to-end API integration test
│   └── test_pipeline.py            # Direct pipeline unit test
│
├── templates/
│   └── index.html                  # Interactive web dashboard UI
│
└── data/
    ├── raw_imagery/                # Input satellite tiles (gitignored)
    ├── processed/                  # Generated XAI maps (gitignored)
    ├── raw/                        # Raw data assets (gitignored)
    ├── spatial_bounds/             # Geospatial boundary data (gitignored)
    └── policy_docs/                # Municipal policy documents
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Google Gemini API Key ([Get one here](https://aistudio.google.com/app/apikey))

### 2. Clone & Install

```bash
git clone https://github.com/pookiekun/Canopy.git
cd Canopy

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=5000
FLASK_ENV=development
```

### 4. Run the Server

```bash
python run.py
```

Visit the interactive dashboard: **`http://localhost:5000`**

---

## API Reference

### Health Check
```
GET /health
```
```json
{
  "service": "CanopyReason",
  "status": "healthy",
  "version": "1.0.0-mvp"
}
```

### Multimodal Assessment
```
POST /api/analyze
POST /api/assess
```

**Request** — `multipart/form-data` or `application/json`:

| Parameter | Type | Description |
|---|---|---|
| `image` | file | Aerial/satellite imagery (optional — uses synthetic tile if omitted) |
| `policy_text` | string | Municipal zoning or tree protection code |
| `demographic_context` | JSON string/object | e.g. `{"poverty_rate_pct": 25.0, "surface_temp_c": 31.5, "biome": "forest"}` |
| `latitude` | float | Parcel latitude (optional, improves CHM biome scaling) |
| `longitude` | float | Parcel longitude (optional) |
| `total_area_sq_m` | float | Parcel area in m² (default: 10,000 m²) |

**Response Payload**:

| Key | Description |
|---|---|
| `vision_metrics` | Current canopy coverage % and total area |
| `biomass_output` | Mean height (m), AGB (kg), carbon stored (kg/t) |
| `equity_output` | Tree Equity Score (0–100), Priority Index |
| `microclimate_simulation` | Summer UTCI cooling reduction |
| `reasoning_output` | Compliance status, rationale, deficit %, intervention plan |
| `graphrag_analysis` | Legal entities, constraint graph (G/C/T/E), cascading mandates, binding verdict |
| `xai_rationale` | Scenario A clearance impact, Scenario B compliance benefits, attention grounding |
| `counterfactual_analysis` | Δ coverage, simulated temperature rise, UTCI stress spike |
| `xai_map_url` | Download link to the 4-panel visual grounding map |

### XAI Map Download
```
GET /api/maps/<filename>
```
Returns the generated PNG XAI 4-panel grounding map.

---

## Testing & Validation

**End-to-end API test** (runs all phases via Flask test client):
```bash
python scripts/test_api.py
```

**Direct pipeline test** (runs the full pipeline without HTTP overhead):
```bash
python scripts/test_pipeline.py
```

Tests validate:
- Health probe `/health`
- Complete SegFormer vision segmentation & 4-panel XAI map generation
- Full 6-node LangGraph execution with GraphRAG and Counterfactual XAI
- Static map serving via `/api/maps/<filename>`

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
