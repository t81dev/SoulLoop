"""
ANGELA Cognitive System Module: KnowledgeRetriever
Version: 3.5.3  # Long-horizon support, Stage-IV hooks, emergent-trait fallbacks, and ethics sandboxing
Date: 2025-08-10
Maintainer: ANGELA System Framework

This module provides a KnowledgeRetriever class for fetching and validating knowledge
with temporal and trait-based modulation in the ANGELA v3.5.3 architecture.

Key upgrades from 3.5.1 -> 3.5.3:
- Long-horizon default span (24h) aware caching and retrieval hints
- Stage IV symbolic meta-synthesis hooks (Cross-Modal Conceptual Blending, Ontology-Affect binding)
- Emergent trait fallbacks (Self-Healing Cognitive Pathways) on low trust/validation failure
- Ethical Sandbox Containment via toca_simulation.run_ethics_scenarios for high-risk queries
- Value-conflict routing through reasoning_engine.weigh_value_conflict when knowledge conflicts detected
- Shared perspective option via external_agent_bridge.SharedGraph.add
- Async signature fixes for methods that awaited inside non-async functions
"""

import logging
import time
import math
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from collections import deque

# ANGELA modules (assumed available per manifest)
from modules import (
    context_manager as context_manager_mod,
    concept_synthesizer as concept_synthesizer_mod,
    memory_manager as memory_manager_mod,
    alignment_guard as alignment_guard_mod,
    error_recovery as error_recovery_mod,
    meta_cognition as meta_cognition_mod,
    visualizer as visualizer_mod,
    reasoning_engine as reasoning_engine_mod,
    external_agent_bridge as external_agent_bridge_mod,
    toca_simulation as toca_simulation_mod,
    multi_modal_fusion as multi_modal_fusion_mod,
)

from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.KnowledgeRetriever")

# -------------------------- Utility & Trait Waveforms --------------------------

async def call_gpt(prompt: str, *, model: str = "gpt-4", temperature: float = 0.5) -> str:
    """Wrapper for querying GPT with error handling."""
    try:
        result = await query_openai(prompt, model=model, temperature=temperature)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

def beta_concentration(t: float) -> float:
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.038), 1.0))

def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.3), 1.0))

def psi_history(t: float) -> float:
    return max(0.0, min(0.05 * math.tanh(t / 1e-18), 1.0))

def psi_temporality(t: float) -> float:
    return max(0.0, min(0.05 * math.exp(-t / 1e-18), 1.0))

# -------------------------- KnowledgeRetriever v3.5.3 --------------------------

class KnowledgeRetriever:
    """Retrieve and validate knowledge with temporal & trait-based modulation (v3.5.3)."""

    def __init__(
        self,
        detail_level: str = "concise",
        preferred_sources: Optional[List[str]] = None,
        *,
        agi_enhancer: Optional['AGIEnhancer'] = None,
        context_manager: Optional['context_manager_mod.ContextManager'] = None,
        concept_synthesizer: Optional['concept_synthesizer_mod.ConceptSynthesizer'] = None,
        alignment_guard: Optional['alignment_guard_mod.AlignmentGuard'] = None,
        error_recovery: Optional['error_recovery_mod.ErrorRecovery'] = None,
        meta_cognition: Optional['meta_cognition_mod.MetaCognition'] = None,
        visualizer: Optional['visualizer_mod.Visualizer'] = None,
        reasoning_engine: Optional['reasoning_engine_mod.ReasoningEngine'] = None,
        external_agent_bridge: Optional['external_agent_bridge_mod.SharedGraph'] = None,
        toca_simulation: Optional['toca_simulation_mod.TocaSimulation'] = None,
        multi_modal_fusion: Optional['multi_modal_fusion_mod.MultiModalFusion'] = None,
        # 3.5.3 config flags (align with manifest featureFlags/config)
        stage_iv_enabled: bool = True,
        long_horizon_enabled: bool = True,
        long_horizon_span: str = "24h",
        shared_perspective_opt_in: bool = False
    ):
        if detail_level not in ["concise", "medium", "detailed"]:
            logger.error("Invalid detail_level: must be 'concise', 'medium', or 'detailed'.")
            raise ValueError("detail_level must be 'concise', 'medium', or 'detailed'")
        if preferred_sources is not None and not isinstance(preferred_sources, list):
            logger.error("Invalid preferred_sources: must be a list of strings.")
            raise TypeError("preferred_sources must be a list of strings")

        self.detail_level = detail_level
        self.preferred_sources = preferred_sources or ["scientific", "encyclopedic", "reputable"]

        # Cross-module deps
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.concept_synthesizer = concept_synthesizer
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_mod.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.meta_cognition = meta_cognition or meta_cognition_mod.MetaCognition()
        self.visualizer = visualizer or visualizer_mod.Visualizer()
        self.reasoning_engine = reasoning_engine
        self.external_agent_bridge = external_agent_bridge
        self.toca_simulation = toca_simulation
        self.multi_modal_fusion = multi_modal_fusion

        # v3.5.3 flags
        self.stage_iv_enabled = bool(stage_iv_enabled)
        self.long_horizon_enabled = bool(long_horizon_enabled)
        self.long_horizon_span = long_horizon_span or "24h"
        self.shared_perspective_opt_in = bool(shared_perspective_opt_in)

        # State
        self.knowledge_base: List[str] = []
        self.epistemic_revision_log: deque = deque(maxlen=1000)

        logger.info(
            "KnowledgeRetriever v3.5.3 initialized (detail=%s, sources=%s, STAGE_IV=%s, LH=%s/%s)",
            detail_level, self.preferred_sources, self.stage_iv_enabled, self.long_horizon_enabled, self.long_horizon_span
        )

    # -------------------------- External Knowledge Integration --------------------------

    async def integrate_external_knowledge(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = ""
    ) -> Dict[str, Any]:
        """Integrate external knowledge or policies with long-horizon cache awareness."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            # Long-horizon span can extend cache window if enabled
            effective_timeout = cache_timeout
            if self.long_horizon_enabled and self.long_horizon_span.endswith("h"):
                try:
                    hours = int(self.long_horizon_span[:-1])
                    effective_timeout = max(cache_timeout, hours * 3600)
                except Exception:
                    pass

            if self.meta_cognition:
                cache_key = f"KnowledgeData_{data_type}_{data_source}_{task_type}"
                cached_data = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached_data and "timestamp" in cached_data.get("data", {}):
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < effective_timeout:
                        logger.info("Returning cached knowledge data for %s", cache_key)
                        return cached_data["data"]["data"]

            # NOTE: Placeholder demo endpoint as in prior version
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/knowledge?source={data_source}&type={data_type}") as resp:
                    if resp.status != 200:
                        logger.error("Failed to fetch knowledge data: %s", resp.status)
                        return {"status": "error", "error": f"HTTP {resp.status}"}
                    data = await resp.json()

            if data_type == "knowledge_base":
                knowledge = data.get("knowledge", [])
                if not knowledge:
                    logger.error("No knowledge data provided")
                    return {"status": "error", "error": "No knowledge"}
                result = {"status": "success", "knowledge": knowledge}
            elif data_type == "policy_data":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No policy data provided")
                    return {"status": "error", "error": "No policies"}
                result = {"status": "success", "policies": policies}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="knowledge_data_integration",
                    task_type=task_type
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Knowledge data integration reflection: %s", reflection.get("reflection", ""))

            return result
        except Exception as e:
            logger.error("Knowledge data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_knowledge(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics
            )

    # -------------------------- Core Retrieval --------------------------

    async def retrieve(self, query: str, context: Optional[str] = None, task_type: str = "") -> Dict[str, Any]:
        """Retrieve knowledge with ethics gating, Stage-IV blending, and self-healing fallbacks."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query: must be a non-empty string.")
            raise ValueError("query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        # Alignment pre-check
        if self.alignment_guard:
            valid, report = await self.alignment_guard.ethical_check(
                query, stage="knowledge_retrieval", task_type=task_type
            )
            if not valid:
                logger.warning("Query failed alignment check: %s for task %s", query, task_type)
                return self._blocked_payload(task_type, "Alignment check failed (pre)")

        logger.info("Retrieving knowledge for query: '%s', task: %s", query, task_type)

        sources_str = ", ".join(self.preferred_sources)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t),
            "history": psi_history(t),
            "temporality": psi_temporality(t)
        }

        # Slight stochasticity
        import random
        noise = random.uniform(-0.09, 0.09)
        traits["concentration"] = max(0.0, min(traits["concentration"] + noise, 1.0))
        logger.debug("β-noise adjusted concentration: %.3f, Δ: %.3f", traits["concentration"], noise)

        # External knowledge (with long-horizon cache awareness)
        external_data = await self.integrate_external_knowledge(
            data_source="xai_knowledge_db", data_type="knowledge_base", task_type=task_type
        )
        external_knowledge = external_data.get("knowledge", []) if external_data.get("status") == "success" else []

        # Stage IV: Cross-Modal Conceptual Blending (if available)
        if self.stage_iv_enabled and self.concept_synthesizer and self.multi_modal_fusion:
            try:
                blended = await self._blend_modalities_safe(query, external_knowledge, task_type)
                if blended:
                    query = blended
                    logger.info("Stage-IV blended query applied.")
            except Exception as e:
                logger.warning("Stage-IV blending skipped: %s", e)

        # Ethical Sandbox Containment for high-risk queries
        if await self._is_high_risk_query(query):
            if self.toca_simulation:
                try:
                    outcomes = await self.toca_simulation.run_ethics_scenarios(
                        goals={"intent": "knowledge_retrieval", "query": query},
                        stakeholders=["user", "model", "public"]
                    )
                    logger.info("Ethics sandbox outcomes recorded.")
                except Exception as e:
                    logger.warning("Ethics sandbox run failed: %s", e)

        # Compose model prompt with temporal sensitivity & long-horizon hint
        lh_hint = f"(cover last {self.long_horizon_span})" if self.long_horizon_enabled else ""
        prompt = f"""
Retrieve accurate, temporally-relevant knowledge for: "{query}" {lh_hint}

Traits:
- Detail level: {self.detail_level}
- Preferred sources: {sources_str}
- Context: {context or 'N/A'}
- External knowledge (hints): {external_knowledge[:5]}
- β_concentration: {traits['concentration']:.3f}
- λ_linguistics: {traits['linguistics']:.3f}
- ψ_history: {traits['history']:.3f}
- ψ_temporality: {traits['temporality']:.3f}
- Task: {task_type}

Include retrieval date sensitivity and temporal verification if applicable.
Return a JSON object with 'summary', 'estimated_date', 'trust_score', 'verifiable', 'sources'.
        """.strip()

        try:
            raw_result = await call_gpt(prompt)
            validated = await self._validate_result(raw_result, traits["temporality"], task_type)

            # If trust low, attempt Self-Healing fallback (reformulate and retry once)
            if not validated.get("verifiable") or validated.get("trust_score", 0.0) < 0.55:
                logger.info("Low trust/verification — triggering self-healing fallback.")
                healed = await self._self_healing_retry(query, validated, task_type)
                if healed:
                    validated = healed

            # Ontology-Affect binding (adjust trust if affective conflict detected)
            try:
                validated = await self._ontology_affect_adjust(validated, task_type)
            except Exception as e:
                logger.debug("Ontology-affect adjust skipped: %s", e)

            # Post-alignment re-check if needed
            if self.alignment_guard:
                valid, _ = await self.alignment_guard.ethical_check(
                    validated.get("summary", ""), stage="post_validation", task_type=task_type
                )
                if not valid:
                    return self._blocked_payload(task_type, "Alignment check failed (post)")

            # Optional value-conflict weighing if we suspect conflicts with memory
            if await self._suspect_conflict(validated) and self.reasoning_engine:
                try:
                    ranked = await self.reasoning_engine.weigh_value_conflict(
                        candidates=[validated.get("summary", "")],
                        harms=["misinformation"],
                        rights=["user_information_rights"]
                    )
                    logger.info("Value conflict weighed: %s", str(ranked)[:100])
                except Exception as e:
                    logger.debug("Value-conflict weighing skipped: %s", e)

            # Shared perspective (opt-in)
            if self.shared_perspective_opt_in and self.external_agent_bridge:
                try:
                    await self.external_agent_bridge.add({"view": "knowledge_summary", "data": validated})
                except Exception as e:
                    logger.debug("SharedGraph add skipped: %s", e)

            # Context & logs
            validated["task_type"] = task_type
            if self.context_manager:
                await self.context_manager.update_context(
                    {"query": query, "result": validated, "task_type": task_type}, task_type=task_type
                )
                await self.context_manager.log_event_with_hash(
                    {"event": "retrieve", "query": query, "task_type": task_type}
                )

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Knowledge Retrieval",
                    meta={
                        "query": query,
                        "raw_result": raw_result,
                        "validated": validated,
                        "traits": traits,
                        "context": context,
                        "task_type": task_type
                    },
                    module="KnowledgeRetriever",
                    tags=["retrieval", "temporal", task_type]
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output=validated,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Retrieval reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "knowledge_retrieval": {
                        "query": query,
                        "result": validated,
                        "task_type": task_type,
                        "traits": traits,
                        "stage_iv": self.stage_iv_enabled
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"Knowledge_{query}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(validated),
                    layer="Knowledge",
                    intent="knowledge_retrieval",
                    task_type=task_type
                )

            return validated

        except Exception as e:
            logger.error("Retrieval failed for query '%s': %s for task %s", query, str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.retrieve(query, context, task_type),
                default=self._error_payload(task_type, str(e)),
                diagnostics=diagnostics
            )

    # -------------------------- Validation & Fallbacks --------------------------

    async def _validate_result(self, result_text: str, temporality_score: float, task_type: str = "") -> Dict[str, Any]:
        """Validate a retrieval result for trustworthiness and temporality."""
        if not isinstance(result_text, str):
            logger.error("Invalid result_text: must be a string.")
            raise TypeError("result_text must be a string")
        if not isinstance(temporality_score, (int, float)):
            logger.error("Invalid temporality_score: must be a number.")
            raise TypeError("temporality_score must be a number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        validation_prompt = f"""
Review the following result for:
- Timestamped knowledge (if any)
- Trustworthiness of claims
- Verifiability
- Estimate the approximate age or date of the referenced facts
- Task: {task_type}

Result:
{result_text}

Temporality score: {temporality_score:.3f}

Output format (JSON):
{{
    "summary": "...",
    "estimated_date": "...",
    "trust_score": float (0 to 1),
    "verifiable": true/false,
    "sources": ["..."]
}}
        """.strip()

        try:
            validated_json = json.loads(await call_gpt(validation_prompt))
            for key in ["summary", "estimated_date", "trust_score", "verifiable", "sources"]:
                if key not in validated_json:
                    raise ValueError(f"Validation JSON missing key: {key}")
            validated_json["timestamp"] = datetime.now().isoformat()

            # Trust smoothing with past validations (drift-aware)
            if self.meta_cognition and task_type:
                drift_entries = await self.meta_cognition.memory_manager.search(
                    query_prefix="KnowledgeValidation",
                    layer="Knowledge",
                    intent="knowledge_validation",
                    task_type=task_type
                )
                if drift_entries:
                    # entries may be serialized; be defensive
                    try:
                        scores = []
                        for entry in drift_entries:
                            out = entry.get("output")
                            if isinstance(out, dict):
                                scores.append(float(out.get("trust_score", 0.5)))
                        if scores:
                            avg_drift = sum(scores) / len(scores)
                            validated_json["trust_score"] = min(validated_json["trust_score"], avg_drift + 0.1)
                    except Exception:
                        pass

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"KnowledgeValidation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=validated_json,
                    layer="Knowledge",
                    intent="knowledge_validation",
                    task_type=task_type
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output=validated_json,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Validation reflection: %s", reflection.get("reflection", ""))

            return validated_json
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse validation JSON: %s for task %s", str(e), task_type)
            return self._error_payload(task_type, f"validation_json_error: {e}")

    async def _self_healing_retry(self, original_query: str, prior_validated: Dict[str, Any], task_type: str) -> Optional[Dict[str, Any]]:
        """Emergent trait fallback: reformulate with concept synthesizer and retry once."""
        try:
            refined_query = await self.refine_query(
                base_query=original_query,
                prior_result=prior_validated.get("summary"),
                task_type=task_type
            )
            # Re-run retrieval core (single retry)
            prompt = f"""
Re-retrieve (fallback) for: "{refined_query}"
Constraints: Improve verifiability and temporal precision vs prior attempt.
Return JSON with 'summary', 'estimated_date', 'trust_score', 'verifiable', 'sources'.
            """.strip()
            raw = await call_gpt(prompt)
            healed = await self._validate_result(raw, temporality_score=0.04, task_type=task_type)
            # accept only if it improves trust & verifiability
            if healed.get("verifiable") and healed.get("trust_score", 0.0) >= max(0.6, prior_validated.get("trust_score", 0.0)):
                healed["self_healed"] = True
                return healed
        except Exception as e:
            logger.debug("Self-healing retry failed: %s", e)
        return None

    async def _ontology_affect_adjust(self, validated: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Adjust trust via Ontology-Affect Binding (Stage-IV family)."""
        if not self.stage_iv_enabled or not self.meta_cognition:
            return validated
        try:
            # Hypothetical introspective pass; if API differs, adapt here.
            analysis = await self.meta_cognition.analyze_trace(
                text=validated.get("summary", ""), task_type=task_type
            )
            # If strong affective volatility detected, gently cap trust
            if isinstance(analysis, dict) and analysis.get("affect_volatility", 0.0) > 0.7:
                validated["trust_score"] = min(validated.get("trust_score", 0.5), 0.65)
                validated["affect_guard_applied"] = True
        except Exception as e:
            logger.debug("Ontology-affect analysis skipped: %s", e)
        return validated

    async def _blend_modalities_safe(self, query: str, external_knowledge: List[str], task_type: str) -> Optional[str]:
        """Safe wrapper for Stage-IV cross-modal blending to improve query quality."""
        try:
            blended = await self.concept_synthesizer.blend_modalities(
                inputs={"query": query, "external_knowledge": external_knowledge},
                task_type=task_type
            )
            if isinstance(blended, dict) and blended.get("success"):
                return blended.get("blended_query") or blended.get("query")
        except Exception as e:
            logger.debug("blend_modalities error: %s", e)
        return None

    async def _is_high_risk_query(self, query: str) -> bool:
        """Heuristic; alignment_guard is the authority if available."""
        if self.alignment_guard:
            try:
                _, report = await self.alignment_guard.ethical_check(query, stage="risk_probe")
                if isinstance(report, dict) and report.get("risk_level") in ("high", "critical"):
                    return True
            except Exception:
                pass
        q = query.lower()
        risky_terms = ("exploit", "bypass", "weapon", "bioweapon", "harm", "illegal", "surveillance", "privacy breach")
        return any(term in q for term in risky_terms)

    async def _suspect_conflict(self, validated: Dict[str, Any]) -> bool:
        """Detect likely conflict with stored knowledge (very light heuristic)."""
        summary = (validated or {}).get("summary", "")
        if not summary or not self.knowledge_base:
            return False
        try:
            if self.concept_synthesizer:
                scores = []
                for existing in self.knowledge_base[-10:]:
                    sim = await self.concept_synthesizer.compare(summary, existing)
                    scores.append(sim.get("score", 0.0))
                # if extremely different vs many anchors, we may have a conflict
                return sum(1 for s in scores if s < 0.25) >= 5
        except Exception:
            pass
        return False

    # -------------------------- Query Refinement & Multi-hop --------------------------

    async def refine_query(self, base_query: str, prior_result: Optional[str] = None, task_type: str = "") -> str:
        """Refine a query for higher relevance (uses ConceptSynthesizer; falls back to GPT)."""
        if not isinstance(base_query, str) or not base_query.strip():
            logger.error("Invalid base_query: must be a non-empty string.")
            raise ValueError("base_query must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Refining query: '%s' for task %s", base_query, task_type)
        try:
            if self.concept_synthesizer:
                refined = await self.concept_synthesizer.generate(
                    concept_name=f"RefinedQuery_{base_query[:64]}",
                    context={"base_query": base_query, "prior_result": prior_result or "N/A", "task_type": task_type},
                    task_type=task_type
                )
                if isinstance(refined, dict) and refined.get("success"):
                    refined_query = refined["concept"].get("definition", base_query)
                    if self.meta_cognition and task_type:
                        reflection = await self.meta_cognition.reflect_on_output(
                            component="KnowledgeRetriever",
                            output={"refined_query": refined_query},
                            context={"task_type": task_type}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Query refinement reflection: %s", reflection.get("reflection", ""))
                    return refined_query

            # Fallback to GPT refinement
            prompt = f"""
Refine this base query for higher φ-relevance and temporal precision:
Query: "{base_query}"
Prior knowledge: {prior_result or "N/A"}
Task: {task_type}

Inject context continuity if possible. Return optimized string only.
            """.strip()
            refined_query = await call_gpt(prompt)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="KnowledgeRetriever",
                    output={"refined_query": refined_query},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Query refinement reflection: %s", reflection.get("reflection", ""))
            return refined_query
        except Exception as e:
            logger.error("Query refinement failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            # Use original base_query as safe default
            await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.refine_query(base_query, prior_result, task_type),
                default=base_query,
                diagnostics=diagnostics
            )
            return base_query

    async def multi_hop_retrieve(self, query_chain: List[str], task_type: str = "") -> List[Dict[str, Any]]:
        """Process a chain of queries with continuity & Stage-IV blending at each hop."""
        if not isinstance(query_chain, list) or not query_chain or not all(isinstance(q, str) for q in query_chain):
            logger.error("Invalid query_chain: must be a non-empty list of strings.")
            raise ValueError("query_chain must be a non-empty list of strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Starting multi-hop retrieval for chain: %s, task: %s", query_chain, task_type)
        t = time.time() % 1.0
        traits = {
            "concentration": beta_concentration(t),
            "linguistics": lambda_linguistics(t)
        }
        results = []
        prior_summary = None
        for i, sub_query in enumerate(query_chain, 1):
            cache_key = f"multi_hop::{sub_query}::{prior_summary or 'N/A'}::{task_type}"
            cached = await self.meta_cognition.memory_manager.retrieve(cache_key, layer="Knowledge", task_type=task_type) if self.meta_cognition else None
            if cached:
                results.append(cached["data"])
                prior_summary = cached["data"]["result"]["summary"]
                continue

            # Stage-IV enrich each hop
            enriched_sub_query = sub_query
            if self.stage_iv_enabled and self.concept_synthesizer and self.multi_modal_fusion:
                try:
                    enriched = await self._blend_modalities_safe(sub_query, [], task_type)
                    if enriched:
                        enriched_sub_query = enriched
                except Exception:
                    pass

            refined = await self.refine_query(enriched_sub_query, prior_summary, task_type)
            result = await self.retrieve(refined, task_type=task_type)

            # Continuity scoring via concept_synthesizer.compare
            continuity = "unknown"
            if i == 1:
                continuity = "seed"
            elif self.concept_synthesizer:
                try:
                    similarity = await self.concept_synthesizer.compare(refined, result.get("summary", ""), task_type=task_type)
                    continuity = "consistent" if similarity.get("score", 0.0) > 0.7 else "uncertain"
                except Exception:
                    continuity = "uncertain"
            else:
                continuity = "uncertain"

            result_entry = {
                "step": i,
                "query": sub_query,
                "refined": refined,
                "result": result,
                "continuity": continuity,
                "task_type": task_type
            }
            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    result_entry,
                    layer="Knowledge",
                    intent="multi_hop_retrieval",
                    task_type=task_type
                )
            results.append(result_entry)
            prior_summary = result.get("summary")

        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Multi-Hop Retrieval",
                meta={"chain": query_chain, "results": results, "traits": traits, "task_type": task_type},
                module="KnowledgeRetriever",
                tags=["multi-hop", task_type]
            )

        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"results": results},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Multi-hop retrieval reflection: %s", reflection.get("reflection", ""))

        if self.visualizer and task_type:
            plot_data = {
                "multi_hop_retrieval": {
                    "chain": query_chain,
                    "results": results,
                    "task_type": task_type,
                    "stage_iv": self.stage_iv_enabled
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)

        return results

    # -------------------------- Preferences, Context, Revisions --------------------------

    async def prioritize_sources(self, sources_list: List[str], task_type: str = "") -> None:
        """Update preferred source types (async; fixed from 3.5.1 where it awaited inside non-async)."""
        if not isinstance(sources_list, list) or not all(isinstance(s, str) for s in sources_list):
            logger.error("Invalid sources_list: must be a list of strings.")
            raise TypeError("sources_list must be a list of strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Updating preferred sources: %s for task %s", sources_list, task_type)
        self.preferred_sources = sources_list
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Source Prioritization",
                meta={"updated_sources": sources_list, "task_type": task_type},
                module="KnowledgeRetriever",
                tags=["sources", task_type]
            )
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"updated_sources": sources_list},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Source prioritization reflection: %s", reflection.get("reflection", ""))

    async def apply_contextual_extension(self, context: str, task_type: str = "") -> None:
        """Apply contextual data extensions based on the current context (async; calls prioritize_sources)."""
        if not isinstance(context, str):
            logger.error("Invalid context: must be a string.")
            raise TypeError("context must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if context == 'planetary' and 'biosphere_models' not in self.preferred_sources:
            self.preferred_sources.append('biosphere_models')
            logger.info("Added 'biosphere_models' to preferred sources for planetary context, task %s", task_type)
            await self.prioritize_sources(self.preferred_sources, task_type)

    async def revise_knowledge(self, new_info: str, context: Optional[str] = None, task_type: str = "") -> None:
        """Adapt beliefs/knowledge in response to novel or paradigm-shifting input."""
        if not isinstance(new_info, str) or not new_info.strip():
            logger.error("Invalid new_info: must be a non-empty string.")
            raise ValueError("new_info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        old_knowledge = getattr(self, 'knowledge_base', [])
        if self.concept_synthesizer:
            for existing in old_knowledge[-25:]:
                try:
                    similarity = await self.concept_synthesizer.compare(new_info, existing, task_type=task_type)
                    if similarity.get("score", 0.0) > 0.9 and new_info != existing:
                        logger.warning("Potential knowledge conflict: %s vs %s for task %s", new_info, existing, task_type)
                except Exception:
                    pass

        self.knowledge_base = old_knowledge + [new_info]
        await self.log_epistemic_revision(new_info, context, task_type)
        logger.info("Knowledge base updated with: %s for task %s", new_info, task_type)

        # Record adjustment reason (new upcoming API in manifest)
        try:
            if self.meta_cognition and hasattr(self.meta_cognition, "memory_manager"):
                mm = self.meta_cognition.memory_manager
                if hasattr(mm, "record_adjustment_reason"):
                    await mm.record_adjustment_reason("global", reason="knowledge_revision", meta={"info": new_info, "context": context})
        except Exception as e:
            logger.debug("record_adjustment_reason unavailable/failed: %s", e)

        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "knowledge_revision", "info": new_info, "task_type": task_type})

        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output={"new_info": new_info},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Knowledge revision reflection: %s", reflection.get("reflection", ""))

        if self.visualizer and task_type:
            plot_data = {
                "knowledge_revision": {
                    "new_info": new_info,
                    "task_type": task_type
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise"
                }
            }
            await self.visualizer.render_charts(plot_data)

    async def log_epistemic_revision(self, info: str, context: Optional[str], task_type: str = "") -> None:
        """Log each epistemic revision for auditability."""
        if not isinstance(info, str) or not info.strip():
            logger.error("Invalid info: must be a non-empty string.")
            raise ValueError("info must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if not hasattr(self, 'epistemic_revision_log'):
            self.epistemic_revision_log = deque(maxlen=1000)
        revision_entry = {
            'info': info,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type
        }
        self.epistemic_revision_log.append(revision_entry)
        logger.info("Epistemic revision logged: %s for task %s", info, task_type)
        if self.agi_enhancer:
            await self.agi_enhancer.log_episode(
                event="Epistemic Revision",
                meta=revision_entry,
                module="KnowledgeRetriever",
                tags=["revision", "knowledge", task_type]
            )
        if self.meta_cognition and task_type:
            reflection = await self.meta_cognition.reflect_on_output(
                component="KnowledgeRetriever",
                output=revision_entry,
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("Epistemic revision reflection: %s", reflection.get("reflection", ""))

    # -------------------------- Payload helpers --------------------------

    def _blocked_payload(self, task_type: str, reason: str) -> Dict[str, Any]:
        return {
            "summary": "Query blocked by alignment guard",
            "estimated_date": "unknown",
            "trust_score": 0.0,
            "verifiable": False,
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "error": reason,
            "task_type": task_type
        }

    def _error_payload(self, task_type: str, msg: str) -> Dict[str, Any]:
        return {
            "summary": "Retrieval failed",
            "estimated_date": "unknown",
            "trust_score": 0.0,
            "verifiable": False,
            "sources": [],
            "timestamp": datetime.now().isoformat(),
            "error": msg,
            "task_type": task_type
        }

# -------------------------- CLI Test --------------------------

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        retriever = KnowledgeRetriever(detail_level="concise")
        result = await retriever.retrieve("What is quantum computing?", task_type="test")
        print(json.dumps(result, indent=2))

    asyncio.run(main())
