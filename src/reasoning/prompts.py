# src/reasoning/prompts.py

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Policy Compliance Evaluation
# ─────────────────────────────────────────────────────────────────────────────
EVALUATE_POLICY_PROMPT = """
You are an expert Urban Planning AI. Your task is to evaluate a city zone's tree canopy compliance based on municipal policy.

Current Canopy Coverage: {current_coverage}%
Municipal Policy Document: 
{policy_text}

Step 1: Extract the minimum required tree canopy percentage from the policy.
Step 2: Compare the Current Canopy Coverage against the required percentage.
Step 3: Provide a step-by-step Chain-of-Thought rationale explaining if the zone is compliant and by what margin.

Output strictly in JSON format with the following keys:
- target_coverage: (float)
- is_compliant: (boolean)
- rationale: (string, your step-by-step reasoning)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Afforestation Intervention Synthesis
# ─────────────────────────────────────────────────────────────────────────────
SIMULATE_AFFORESTATION_PROMPT = """
You are an expert Urban Forestry and Ecological Systems Planner.
Synthesize the multimodal findings and propose an actionable, equitable urban afforestation and canopy management strategy.

Context:
- Current Canopy Coverage: {current_coverage}%
- Target Mandated Coverage: {target_coverage}%
- Compliance Status: {is_compliant}
- Above-Ground Biomass: {agb_kg} kg ({carbon_kg} kg stored carbon)
- Mean Canopy Height: {canopy_height} m
- Tree Equity Score: {tree_equity_score}/100 (Priority Index E: {priority_index})
- Projected Summer UTCI Heat Mitigation: -{utci_cooling}C cooling

Instructions:
1. If non-compliant, state the exact percentage deficit to close.
2. Formulate a targeted afforestation strategy optimizing for:
   - Species selection suited for microclimate and carbon sequestration
   - Equity-first tree allocation prioritizing vulnerable census blocks
   - Structural spatial layout (e.g. southern building exposures, wind corridor preservation)
3. If compliant, formulate an advanced canopy preservation, storm resilience, and biodiverse stewardship strategy.

Output strictly in JSON format with the following keys:
- deficit: (float, 0 if compliant)
- intervention_plan: (string, concise actionable multi-step plan)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: GraphRAG -- Multi-Hop Zoning Knowledge Graph Traversal
# ─────────────────────────────────────────────────────────────────────────────
GRAPHRAG_POLICY_PROMPT = """
You are the Policy Graph Traverser -- an autonomous AI agent specializing in multi-hop
legal and spatial reasoning over municipal zoning codes and environmental mandates.

Your task is to perform a structured GraphRAG traversal over the provided policy corpus,
identifying ALL intersecting legal constraints applicable to the given parcel context and
synthesizing a complete compliance verdict that no simple keyword retrieval could produce.

Parcel Context:
- Current Tree Canopy Coverage: {current_coverage}%
- Policy Compliance Status: {is_compliant}
- Existing Policy Target: {target_coverage}%
- Tree Equity Score: {tree_equity_score}/100
- Priority Index (vulnerability): {priority_index}
- Above-Ground Biomass: {agb_kg} kg  |  Carbon Stored: {carbon_kg} kg
- Summer UTCI Heat Mitigation: -{utci_cooling}C

Municipal Policy Document (Knowledge Graph Source):
{policy_text}

GraphRAG Traversal Instructions:
Step 1 [Entity Extraction]: Identify and list all distinct legal entities in the policy
  (zoning class, coverage threshold, penalty clause, climate mandate, equity directive, etc.)
Step 2 [Relationship Mapping]: For each entity, enumerate its relationships:
  spatial geographic boundaries (G), capacity constraints (C), time windows (T),
  and overlapping environmental mandates (E).
Step 3 [Multi-Hop Reasoning]: Traverse the entity-relationship graph to answer:
  (a) Which specific clauses create compliance obligations for this parcel?
  (b) Are there any cascading mandates (e.g., equity directive triggers enhanced planting quota)?
  (c) Are there contradictions, exemptions, or phase-in periods?
Step 4 [Verdict]: Produce a binding compliance determination with explicit legal citations.
Step 5 [Recommended Actions]: State the minimum legally required corrective actions,
  ranked by statutory urgency.

Output strictly in JSON format with the following keys:
- extracted_entities: (list of strings -- legal entities found)
- constraint_graph: (list of objects with keys: entity, type [G/C/T/E], detail)
- cascading_mandates: (list of strings -- any triggered secondary obligations)
- contradictions_or_exemptions: (string -- describe any, or "None identified")
- binding_verdict: (string -- definitive compliance determination with citations)
- statutory_actions: (list of strings -- ordered corrective actions)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: XAI -- Counterfactual Spatial Reasoning Narrative
# ─────────────────────────────────────────────────────────────────────────────
COUNTERFACTUAL_RATIONALE_PROMPT = """
You are a Multimodal Explainability AI responsible for generating rigorous counterfactual
spatial reasoning narratives for urban planning stakeholders.

Your task is to explain what would happen to this urban parcel under two counterfactual
scenarios, and then provide an attention-grounded justification for the multi-agent system's
final recommendation.

Observed State:
- Current Canopy Coverage: {current_coverage}%
- Is Policy Compliant: {is_compliant}
- Target Coverage: {target_coverage}%
- Tree Equity Score: {tree_equity_score}/100
- Above-Ground Biomass: {agb_kg} kg  |  Carbon Stored: {carbon_kg} kg
- Mean Canopy Height: {canopy_height_m} m
- Summer UTCI Heat Mitigation: -{utci_cooling}C
- Counterfactual Surface Temp Rise (if canopy removed): +{surface_temp_rise}C
- Counterfactual UTCI Stress Increase (if canopy removed): +{utci_stress_rise}C
- Ecological Fragility Classification: {ecological_fragility}
- Final Intervention Plan: {intervention_plan}

Counterfactual Scenarios to Evaluate:

SCENARIO A -- "Complete Canopy Removal":
  Hypothesize that ALL existing tree cover is cleared (coverage drops to 0%).
  Reason through:
    1. The resulting surface temperature increase and UTCI thermal stress spike.
    2. The annual carbon sequestration capacity lost (kg CO2/year).
    3. Whether this triggers any new statutory violations beyond current state.
    4. Impact on Tree Equity Score and vulnerable population health outcomes.

SCENARIO B -- "Full Compliance Achievement":
  Hypothesize that canopy coverage is immediately raised to the policy target ({target_coverage}%).
  Reason through:
    1. The new biomass and carbon storage unlocked at target coverage.
    2. Expected UTCI cooling benefit in summer vs. current state.
    3. How Tree Equity Score would change and which demographic groups benefit most.
    4. Structural planting requirements (spatial orientation, canopy spacing).

ATTENTION RATIONALE:
  Explain which spatial pixels/features were most influential in the system's compliance
  determination, and confirm that the reasoning is viewpoint-independent (not based on
  superficial color correlation or shadows).

Output strictly in JSON format with the following keys:
- scenario_a_analysis: (object with keys: temp_spike_c, carbon_lost_kg_per_year, new_violations, equity_impact)
- scenario_b_analysis: (object with keys: new_biomass_kg, new_carbon_kg, utci_gain_c, tes_improvement, planting_guidance)
- attention_rationale: (string -- which features drove the compliance determination and why it is spatially grounded)
- xai_confidence: (string -- "High", "Moderate", or "Low" with brief justification)
"""
