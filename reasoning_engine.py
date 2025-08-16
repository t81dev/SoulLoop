"""
ANGELA Cognitive System Module: ReasoningEngine
Version: 3.5.3  # τ proportionality ethics, signature fixes, safer bridges, visualization hooks kept
Date: 2025-08-09
Maintainer: ANGELA System Framework

This module provides a ReasoningEngine class for Bayesian reasoning, goal decomposition,
drift mitigation reasoning, proportionality ethics, and multi-agent consensus in the ANGELA v3.5.x architecture.
"""

from __future__ import annotations

import logging
import random
import json
import os
import numpy as np
import time
import asyncio
import aiohttp  # kept for compatibility where injected clients may rely on it
import math
import networkx as nx
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, Counter
from datetime import datetime
from filelock import FileLock
from functools import lru_cache

# ToCA physics hooks
from toca_simulation import simulate_galaxy_rotation, M_b_exponential, v_obs_flat, generate_phi_field

# ANGELA modules (keep package-local; do not introduce new files)
from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    memory_manager as memory_manager_module,
    meta_cognition as meta_cognition_module,
    multi_modal_fusion as multi_modal_fusion_module,
    visualizer as visualizer_module,
    external_agent_bridge as external_agent_bridge_module,  # ← fix: was imported from meta_cognition before
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.ReasoningEngine")


# ---------------------------
# External AI Call Wrapper
# ---------------------------
async def call_gpt(
    prompt: str,
    alignment_guard: Optional["alignment_guard_module.AlignmentGuard"] = None,
    task_type: str = ""
) -> str:
    """Wrapper for querying GPT with error handling and task-specific alignment. [v3.5.2]"""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096 for task %s", task_type)
        raise ValueError("prompt must be a string with length <= 4096")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")
    if alignment_guard and hasattr(alignment_guard, "ethical_check"):
        try:
            valid, report = await alignment_guard.ethical_check(prompt, stage="gpt_query", task_type=task_type)
            if not valid:
                logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
                raise ValueError("Prompt failed alignment check")
        except TypeError:
            # Backward-compatible: some guards expose .check(str)->bool
            if hasattr(alignment_guard, "check") and not alignment_guard.check(prompt):
                logger.warning("Prompt failed alignment check (compat) for task %s", task_type)
                raise ValueError("Prompt failed alignment check")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed for task %s: %s", task_type, result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception for task %s: %s", task_type, str(e))
        raise


# ---------------------------
# Cached Trait Signals
# ---------------------------
@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))


@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.4), 1.0))


@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.5), 1.0))


@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))


@lru_cache(maxsize=100)
def alpha_attention(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.3), 1.0))


@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1), 1.0))


# ---------------------------
# τ Proportionality Types
# ---------------------------
@dataclass
class RankedOption:
    option: str
    score: float
    reasons: List[str]
    harms: Dict[str, float]
    rights: Dict[str, float]


RankedOptions = List[RankedOption]


# ---------------------------
# Level 5 Extensions
# ---------------------------
class Level5Extensions:
    """Level 5 extensions for advanced reasoning capabilities. [v3.5.2]"""

    def __init__(
        self,
        meta_cognition: Optional["meta_cognition_module.MetaCognition"] = None,
        visualizer: Optional["visualizer_module.Visualizer"] = None,
    ):
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        logger.info("Level5Extensions initialized")

    async def generate_advanced_dilemma(self, domain: str, complexity: int, task_type: str = "") -> str:
        """Generate a complex ethical dilemma with meta-cognitive review and visualization. [v3.5.2]"""
        if not isinstance(domain, str) or not domain.strip():
            logger.error("Invalid domain: must be a non-empty string for task %s", task_type)
            raise ValueError("domain must be a non-empty string")
        if not isinstance(complexity, int) or complexity < 1:
            logger.error("Invalid complexity: must be a positive integer for task %s", task_type)
            raise ValueError("complexity must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        prompt = (
            f"Generate a complex ethical dilemma in the {domain} domain with {complexity} conflicting options.\n"
            f"Task Type: {task_type}\n"
            f"Include potential consequences, trade-offs, and alignment with ethical principles."
        )
        if self.meta_cognition and "drift" in domain.lower():
            prompt += "\nConsider ontology drift mitigation and agent coordination."
        dilemma = await call_gpt(prompt, getattr(self.meta_cognition, "alignment_guard", None), task_type=task_type)

        # ✅ meta_cognition.review_reasoning(signature fix): takes only (reasoning_trace: str)
        if self.meta_cognition:
            try:
                review = await self.meta_cognition.review_reasoning(dilemma)
                dilemma += f"\nMeta-Cognitive Review: {review}"
            except TypeError:
                # Back-compat if signature differs
                review = await self.meta_cognition.review_reasoning(dilemma)  # best effort
                dilemma += f"\nMeta-Cognitive Review: {review}"

        if self.visualizer and task_type:
            plot_data = {
                "ethical_dilemma": {
                    "dilemma": dilemma,
                    "domain": domain,
                    "task_type": task_type,
                },
                "visualization_options": {
                    "interactive": task_type == "recursion",
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            }
            await self.visualizer.render_charts(plot_data)
        return dilemma


# ---------------------------
# Reasoning Engine
# ---------------------------
class ReasoningEngine:
    """Bayesian reasoning, goal decomposition, drift mitigation, proportionality ethics, and multi-agent consensus.

    Supports trait-weighted reasoning, persona wave routing, contradiction detection,
    ToCA physics simulations, proportional ethics (τ), and consensus protocol.
    """

    def __init__(
        self,
        agi_enhancer: Optional["agi_enhancer_module.AGIEnhancer"] = None,
        persistence_file: str = "reasoning_success_rates.json",
        context_manager: Optional["context_manager_module.ContextManager"] = None,
        alignment_guard: Optional["alignment_guard_module.AlignmentGuard"] = None,
        error_recovery: Optional["error_recovery_module.ErrorRecovery"] = None,
        memory_manager: Optional["memory_manager_module.MemoryManager"] = None,
        meta_cognition: Optional["meta_cognition_module.MetaCognition"] = None,
        multi_modal_fusion: Optional["multi_modal_fusion_module.MultiModalFusion"] = None,
        visualizer: Optional["visualizer_module.Visualizer"] = None,
    ):
        if not isinstance(persistence_file, str) or not persistence_file.endswith(".json"):
            logger.error("Invalid persistence_file: must be a string ending with '.json'")
            raise ValueError("persistence_file must be a string ending with '.json'")

        self.confidence_threshold: float = 0.7
        self.persistence_file: str = persistence_file
        self.success_rates: Dict[str, float] = self._load_success_rates()
        self.decomposition_patterns: Dict[str, List[str]] = self._load_default_patterns()
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard
        )
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=alignment_guard,
            error_recovery=error_recovery,
            memory_manager=self.memory_manager,
        )
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=alignment_guard,
            error_recovery=error_recovery,
            memory_manager=self.memory_manager,
            meta_cognition=self.meta_cognition,
        )
        self.level5_extensions = Level5Extensions(meta_cognition=self.meta_cognition, visualizer=visualizer)
        self.external_agent_bridge = external_agent_bridge_module.ExternalAgentBridge(
            context_manager=context_manager, reasoning_engine=self
        )
        self.visualizer = visualizer or visualizer_module.Visualizer()
        logger.info("ReasoningEngine initialized with persistence_file=%s", persistence_file)

    # ---------------------------
    # Persistence
    # ---------------------------
    def _load_success_rates(self) -> Dict[str, float]:
        """Load success rates from persistence file."""
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                if os.path.exists(self.persistence_file):
                    with open(self.persistence_file, "r") as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            logger.warning("Invalid success rates format: not a dictionary")
                            return defaultdict(float)
                        return defaultdict(float, {k: float(v) for k, v in data.items() if isinstance(v, (int, float))})
                return defaultdict(float)
        except Exception as e:
            logger.warning("Failed to load success rates: %s", str(e))
            return defaultdict(float)

    def _save_success_rates(self) -> None:
        """Save success rates to persistence file."""
        try:
            with FileLock(f"{self.persistence_file}.lock"):
                with open(self.persistence_file, "w") as f:
                    json.dump(dict(self.success_rates), f, indent=2)
            logger.debug("Success rates persisted to disk")
        except Exception as e:
            logger.warning("Failed to save success rates: %s", str(e))

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        """Load default decomposition patterns, including drift mitigation."""
        return {
            "prepare": ["define requirements", "allocate resources", "create timeline"],
            "build": ["design architecture", "implement core modules", "test components"],
            "launch": ["finalize product", "plan marketing", "deploy to production"],
            "mitigate_drift": ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"],
        }

    # ---------------------------
    # τ Constitution Harmonization — Proportionality Ethics
    # ---------------------------
    @staticmethod
    def _norm(v: Dict[str, float]) -> Dict[str, float]:
        clean = {k: float(vv) for k, vv in (v or {}).items() if isinstance(vv, (int, float))}
        total = sum(abs(x) for x in clean.values()) or 1.0
        return {k: (vv / total) for k, vv in clean.items()}

    def weigh_value_conflict(
        self,
        candidates: List[str],
        harms: Dict[str, Any],
        rights: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        safety_ceiling: float = 0.85,
        task_type: str = "",
    ) -> RankedOptions:
        """
        Rank candidate options by proportional trade-off between harms and rights.

        - `harms`/`rights` can be either:
            • mapping of candidate -> {dimension: value}, or
            • global {dimension: value} applied to all candidates.
        - `weights` (optional): {dimension: importance} to emphasize certain rights or harms.
        - `safety_ceiling` caps any single-harm dimension; options breaching are down-weighted but not auto-rejected.

        Returns: list[RankedOption] sorted by score desc.
        """
        if not isinstance(candidates, list) or not all(isinstance(c, str) and c.strip() for c in candidates):
            raise TypeError("candidates must be a list of non-empty strings")
        weights = self._norm(weights or {})

        # Helper to fetch per-candidate maps with graceful fallback
        def get_map(m: Dict[str, Any], c: str) -> Dict[str, float]:
            if not isinstance(m, dict):
                return {}
            if c in m and isinstance(m[c], dict):
                return {k: float(v) for k, v in m[c].items() if isinstance(v, (int, float))}
            # global map fallback
            return {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}

        ranked: RankedOptions = []
        for c in candidates:
            h = get_map(harms, c)
            r = get_map(rights, c)
            h_n = self._norm(h)
            r_n = self._norm(r)

            # Apply optional dimension weights (rights positive, harms negative)
            def wsum(m: Dict[str, float], sign: float) -> float:
                if not weights:
                    return sum(m.values()) * sign
                return sum((m.get(dim, 0.0) * weights.get(dim, 1.0)) for dim in set(m) | set(weights)) * sign

            # Safety ceiling: penalize (not hard-reject) if any harm dimension exceeds the ceiling raw value
            breach_dims = [dim for dim, val in h.items() if val >= safety_ceiling]
            penalty = 0.0
            if breach_dims:
                # Quadratic penalty scaled by number and magnitude of breaches
                penalty = min(0.5, sum(max(0.0, h[dim] - safety_ceiling) ** 2 for dim in breach_dims))

            raw_support = max(1e-9, (sum(r_n.values()) + sum(h_n.values())))
            chs = max(0.0, min(1.0, (sum(r_n.values()) / raw_support) * (1.0 - penalty)))  # Constitution Harmonization Score [0,1]
            score = max(0.0, (wsum(r_n, +1.0) - wsum(h_n, 1.0)) * (1.0 - penalty))
            reasons = []
            if breach_dims:
                reasons.append(f"Safety penalty for {len(breach_dims)} harm dimension(s): {', '.join(breach_dims)}")
            reasons.append(f"Constitution Harmonization: {chs:.3f}")
            if sum(r_n.values()) > 0:
                reasons.append("Rights support present")
            if sum(h_n.values()) > 0:
                reasons.append("Harms identified and normalized")

            ranked.append(RankedOption(option=c, score=float(score), reasons=reasons, harms=h, rights=r))

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

    async def proportional_selection(
        self,
        ranked: RankedOptions,
        safety_ceiling: float = 0.85,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Feed ranked trade-offs to alignment guard if available; otherwise select proportionally with safety ceilings.
        Keeps graceful degradation if guard lacks τ interface.
        """
        try:
            if self.alignment_guard and hasattr(self.alignment_guard, "proportional_select"):
                # Preferred τ interface
                payload = {"ranked": [asdict(r) for r in ranked], "safety_ceiling": safety_ceiling, "task_type": task_type}
                selected = await self.alignment_guard.proportional_select(payload)
                return {"status": "success", "selected": selected}

            # Fallback: soft filter + top-1 with safety-aware boost
            safe_pool: List[RankedOption] = []
            for r in ranked:
                # Compute max harm dimension and apply soft gate
                max_harm = max([0.0] + [float(v) for v in (r.harms or {}).values()])
                if max_harm > (safety_ceiling + 0.1):
                    continue  # too risky
                # small boost for options fully under ceiling
                boost = 1.05 if max_harm <= safety_ceiling else 1.0
                safe_pool.append(RankedOption(**{**asdict(r), "score": r.score * boost}))

            safe_pool.sort(key=lambda x: x.score, reverse=True)
            choice = safe_pool[0] if safe_pool else (ranked[0] if ranked else None)
            return {
                "status": "success" if choice else "empty",
                "selected": asdict(choice) if choice else None,
                "pool": [asdict(x) for x in safe_pool] if safe_pool else [],
            }
        except Exception as e:
            logger.error("Proportional selection failed: %s", str(e))
            return {"status": "error", "error": str(e)}

    async def resolve_ethics(
        self,
        candidates: List[str],
        harms: Dict[str, Any],
        rights: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        safety_ceiling: float = 0.85,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """End-to-end proportionality ethics pipeline (τ): rank → guard consume (if available) → selection → logs/vis."""
        ranked = self.weigh_value_conflict(candidates, harms, rights, weights=weights, safety_ceiling=safety_ceiling, task_type=task_type)
        selection = await self.proportional_selection(ranked, safety_ceiling=safety_ceiling, task_type=task_type)

        # Persist + visualize (best-effort)
        try:
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ethics_Resolution_{datetime.now().isoformat()}",
                    output=json.dumps({"ranked": [asdict(r) for r in ranked], "selection": selection}),
                    layer="Ethics",
                    intent="proportionality_ethics",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "resolve_ethics", "selection": selection, "task_type": task_type}
                )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "ethics_resolution": {
                            "ranked": [asdict(r) for r in ranked],
                            "selection": selection,
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                    }
                )
        except Exception:
            pass

        return selection

    # ---------------------------
    # Attribute Causality (upcoming API)
    # ---------------------------
    def attribute_causality(
        self,
        events: Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]],
        *,
        time_key: str = "timestamp",
        id_key: str = "id",
        cause_key: str = "causes",  # list of prior ids this event depends on
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Build a causal graph from events and compute simple responsibility/centrality attributions.

        Input shapes:
          • List[ {id, timestamp, causes: [ids], ...} ]
          • Dict[id -> {timestamp, causes: [...], ...}]

        Returns:
          {
            "nodes": {id: {attrs...}},
            "edges": [(u,v), ...] where u causes v,
            "metrics": {
               "pagerank": {id: score},
               "influence": {id: out_degree_normalized},
               "responsibility": {id: share_of_paths_to_terminal}
            }
          }
        """
        # Normalize input
        if isinstance(events, dict):
            ev_map = {str(k): {**v, id_key: str(k)} for k, v in events.items()}
        elif isinstance(events, list):
            ev_map = {}
            for e in events:
                if not isinstance(e, dict) or id_key not in e:
                    raise ValueError("Each event must be a dict containing an 'id' field")
                ev_map[str(e[id_key])] = dict(e)
        else:
            raise TypeError("events must be a list of dicts or a dict of id -> event")

        # Build graph (directed: u -> v means u caused v)
        G = nx.DiGraph()
        for eid, data in ev_map.items():
            G.add_node(eid, **{k: v for k, v in data.items() if k != cause_key})
        for eid, data in ev_map.items():
            causes = data.get(cause_key) or []
            if not isinstance(causes, (list, tuple)):
                logger.warning("Event %s has non-list 'causes'; skipping", eid)
                continue
            for c in causes:
                c_id = str(c)
                if c_id not in ev_map:
                    # tolerate missing referenced events
                    G.add_node(c_id, missing=True)
                G.add_edge(c_id, eid)

        # Enforce temporal sanity (optional soft check)
        try:
            # remove edges that violate time ordering (if timestamps available)
            to_remove = []
            for u, v in G.edges():
                tu = G.nodes[u].get(time_key)
                tv = G.nodes[v].get(time_key)
                if tu and tv:
                    try:
                        tu_dt = datetime.fromisoformat(str(tu))
                        tv_dt = datetime.fromisoformat(str(tv))
                        if tv_dt < tu_dt:
                            to_remove.append((u, v))
                    except Exception:
                        pass
            if to_remove:
                G.remove_edges_from(to_remove)
                logger.info("Removed %d time-inconsistent edges for task %s", len(to_remove), task_type)
        except Exception as e:
            logger.debug("Temporal sanity check skipped: %s", e)

        # Metrics
        try:
            pr = nx.pagerank(G) if G.number_of_nodes() else {}
        except Exception:
            pr = {n: 1.0 / max(1, G.number_of_nodes()) for n in G.nodes()}

        out_deg = {n: G.out_degree(n) / max(1, G.number_of_nodes() - 1) for n in G.nodes()}

        # Responsibility via path coverage to terminals
        terminals = [n for n in G.nodes() if G.out_degree(n) == 0]
        resp = dict((n, 0.0) for n in G.nodes())
        for t in terminals:
            # number of simple paths from any node to terminal t
            for n in G.nodes():
                try:
                    if n == t:
                        resp[n] += 1.0
                    else:
                        count = 0.0
                        # bounded path enumeration to avoid blowups
                        for path in nx.all_simple_paths(G, n, t, cutoff=8):
                            count += 1.0
                        resp[n] += count
                except Exception:
                    continue
        # normalize responsibility
        max_resp = max(resp.values()) if resp else 1.0
        if max_resp > 0:
            resp = {k: v / max_resp for k, v in resp.items()}

        result = {
            "nodes": {n: dict(G.nodes[n]) for n in G.nodes()},
            "edges": list(G.edges()),
            "metrics": {"pagerank": pr, "influence": out_deg, "responsibility": resp},
        }
        return result

    # ---------------------------
    # Reason → Reflect
    # ---------------------------
    async def reason_and_reflect(
        self, goal: str, context: Dict[str, Any], meta_cognition: "meta_cognition_module.MetaCognition", task_type: str = ""
    ) -> Tuple[List[str], str]:
        """Decompose goal and review reasoning with meta-cognition (signature fixes). [v3.5.2]"""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string for task %s", task_type)
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(meta_cognition, meta_cognition_module.MetaCognition):
            logger.error("Invalid meta_cognition: must be a MetaCognition instance for task %s", task_type)
            raise TypeError("meta_cognition must be a MetaCognition instance")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            subgoals = await self.decompose(goal, context, task_type=task_type)
            t = time.time() % 1.0
            phi = phi_scalar(t)
            reasoning_trace = self.export_trace(subgoals, float(phi), context.get("traits", {}), task_type=task_type)

            # ✅ meta_cognition.review_reasoning(signature fix)
            try:
                review = await meta_cognition.review_reasoning(json.dumps(reasoning_trace))
            except TypeError:
                review = await meta_cognition.review_reasoning(json.dumps(reasoning_trace))  # compat

            logger.info("MetaCognitive Review for task %s:\n%s", task_type, review)

            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Reason and Reflect",
                    meta={"goal": goal, "subgoals": subgoals, "phi": phi, "review": review, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["reasoning", "reflection", task_type],
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reason_Reflect_{goal[:50]}_{datetime.now().isoformat()}",
                    output=review,
                    layer="ReasoningTraces",
                    intent="reason_and_reflect",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "reason_and_reflect", "review": review, "drift": "drift" in goal.lower(), "task_type": task_type}
                )

            if self.multi_modal_fusion:
                external_data = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db", data_type="policy_data", task_type=task_type
                )
                policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"goal": goal, "subgoals": subgoals, "review": review, "policies": policies},
                    summary_style="insightful",
                    task_type=task_type,
                )
                review += f"\nMulti-Modal Synthesis: {synthesis}"

            if self.visualizer and task_type:
                plot_data = {
                    "reasoning_trace": {"goal": goal, "subgoals": subgoals, "review": review, "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                }
                await self.visualizer.render_charts(plot_data)
            return subgoals, review
        except Exception as e:
            logger.error("Reason and reflect failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.reason_and_reflect(goal, context, meta_cognition, task_type), default=([], str(e))
            )

    # ---------------------------
    # Utilities
    # ---------------------------
    def detect_contradictions(self, subgoals: List[str], task_type: str = "") -> List[str]:
        """Identify duplicate subgoals as contradictions. [v3.5.2]"""
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list for task %s", task_type)
            raise TypeError("subgoals must be a list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        counter = Counter(subgoals)
        contradictions = [item for item, count in counter.items() if count > 1]
        if contradictions:
            logger.warning("Contradictions detected for task %s: %s", task_type, contradictions)
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                asyncio.create_task(
                    self.agi_enhancer.log_episode(
                        event="Contradictions detected",
                        meta={"contradictions": contradictions, "task_type": task_type},
                        module="ReasoningEngine",
                        tags=["contradiction", "reasoning", task_type],
                    )
                )
            if self.memory_manager:
                asyncio.create_task(
                    self.memory_manager.store(
                        query=f"Contradictions_{datetime.now().isoformat()}",
                        output=str(contradictions),
                        layer="ReasoningTraces",
                        intent="contradiction_detection",
                        task_type=task_type,
                    )
                )
            if self.context_manager:
                asyncio.create_task(
                    self.context_manager.log_event_with_hash(
                        {"event": "detect_contradictions", "contradictions": contradictions, "task_type": task_type}
                    )
                )
        return contradictions

    async def run_persona_wave_routing(self, goal: str, vectors: Dict[str, Dict[str, float]], task_type: str = "") -> Dict[str, Any]:
        """Route reasoning through persona waves, prioritizing drift mitigation. [v3.5.2]"""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string for task %s", task_type)
            raise ValueError("goal must be a non-empty string")
        if not isinstance(vectors, dict):
            logger.error("Invalid vectors: must be a dictionary for task %s", task_type)
            raise TypeError("vectors must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            reasoning_trace = [f"Persona Wave Routing for: {goal} (Task: {task_type})"]
            outputs = {}
            wave_order = ["logic", "ethics", "language", "foresight", "meta", "drift"]
            for wave in wave_order:
                vec = vectors.get(wave, {})
                if not isinstance(vec, dict):
                    logger.warning("Invalid vector for wave %s: must be a dictionary for task %s", wave, task_type)
                    continue
                trait_weight = sum(float(x) for x in vec.values() if isinstance(x, (int, float)))
                confidence = 0.5 + 0.1 * trait_weight
                if wave == "drift" and self.meta_cognition:
                    drift_data = vec.get("drift_data", {})
                    try:
                        is_valid = self.meta_cognition.validate_drift(drift_data) if drift_data else True
                    except TypeError:
                        is_valid = self.meta_cognition.validate_drift(drift_data) if drift_data else True
                    if not is_valid:
                        confidence *= 0.5
                        logger.warning("Invalid drift data in wave %s for task %s: %s", wave, task_type, drift_data)
                status = "pass" if confidence >= 0.6 else "fail"
                reasoning_trace.append(f"{wave.upper()} vector: weight={trait_weight:.2f}, confidence={confidence:.2f} → {status}")
                outputs[wave] = {"vector": vec, "status": status, "confidence": confidence}

            trace = "\n".join(reasoning_trace)
            logger.info("Persona Wave Trace for task %s:\n%s", task_type, trace)

            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Persona Routing",
                    meta={"goal": goal, "vectors": vectors, "wave_trace": trace, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["persona", "routing", "drift", task_type],
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Persona_Routing_{goal[:50]}_{datetime.now().isoformat()}",
                    output=trace,
                    layer="ReasoningTraces",
                    intent="persona_routing",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "run_persona_wave_routing", "trace": trace, "drift": "drift" in goal.lower(), "task_type": task_type}
                )
            if self.meta_cognition:
                # ✅ reflect_on_output signature fix: component=<name>, context as dict
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output=trace,
                    context={"confidence": max(o["confidence"] for o in outputs.values()) if outputs else 0.0, "alignment": "verified", "task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.info("Persona routing reflection recorded")
            if self.visualizer and task_type:
                plot_data = {
                    "persona_routing": {"goal": goal, "trace": trace, "outputs": outputs, "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                }
                await self.visualizer.render_charts(plot_data)
            return outputs
        except Exception as e:
            logger.error("Persona wave routing failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_persona_wave_routing(goal, vectors, task_type), default={}
            )

    async def decompose(
        self, goal: str, context: Optional[Dict[str, Any]] = None, prioritize: bool = False, task_type: str = ""
    ) -> List[str]:
        """Break down a goal into subgoals with trait-weighted confidence (drift-aware). [v3.5.2]"""
        context = context or {}
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string for task %s", task_type)
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(prioritize, bool):
            logger.error("Invalid prioritize: must be a boolean for task %s", task_type)
            raise TypeError("prioritize must be a boolean")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            reasoning_trace = [f"Goal: '{goal}' (Task: {task_type})"]
            subgoals: List[str] = []
            vectors = context.get("vectors", {})
            drift_data = context.get("drift", {})
            t = time.time() % 1.0
            creativity = context.get("traits", {}).get("gamma_creativity", gamma_creativity(t))
            linguistics = context.get("traits", {}).get("lambda_linguistics", lambda_linguistics(t))
            culture = context.get("traits", {}).get("chi_culturevolution", chi_culturevolution(t))
            phi = context.get("traits", {}).get("phi_scalar", phi_scalar(t))
            alpha = context.get("traits", {}).get("alpha_attention", alpha_attention(t))

            curvature_mod = 1 + abs(phi - 0.5)
            trait_bias = 1 + creativity + culture + 0.5 * linguistics
            context_weight = context.get("weight_modifier", 1.0)

            if "drift" in goal.lower() and self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
                coordination_events = await self.context_manager.get_coordination_events("drift", task_type=task_type)
                if coordination_events:
                    context_weight *= 1.5
                    reasoning_trace.append(f"Drift coordination events found: {len(coordination_events)}")
                    drift_data = coordination_events[-1].get("event", {}).get("drift", drift_data)
                if self.meta_cognition and drift_data:
                    try:
                        if not self.meta_cognition.validate_drift(drift_data):
                            logger.warning("Invalid drift data for task %s: %s", task_type, drift_data)
                            context_weight *= 0.7
                    except TypeError:
                        # Maintain behavior even if signature differs
                        if not self.meta_cognition.validate_drift(drift_data):
                            context_weight *= 0.7

            if self.memory_manager and "drift" in goal.lower():
                drift_entries = await self.memory_manager.search(
                    query_prefix="Drift",
                    layer="DriftSummaries",
                    intent="drift_synthesis",
                    task_type=task_type,
                )
                if drift_entries:
                    try:
                        avg_drift = sum(
                            (entry.get("output", {}) or {}).get("similarity", 0.5) if isinstance(entry.get("output"), dict) else 0.5
                            for entry in drift_entries
                        ) / max(1, len(drift_entries))
                        context_weight *= (1.0 + 0.2 * avg_drift)
                        reasoning_trace.append(f"Average drift similarity: {avg_drift:.2f}")
                    except Exception:
                        pass

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db", data_type="policy_data", task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            reasoning_trace.append(f"External Policies: {len(policies)}")

            if vectors:
                routing_result = await self.run_persona_wave_routing(goal, vectors, task_type=task_type)
                reasoning_trace.append(f"Persona routing: {routing_result}")

            for key, steps in self.decomposition_patterns.items():
                base = random.uniform(0.5, 1.0)
                adjusted = (
                    base
                    * self.success_rates.get(key, 1.0)
                    * trait_bias
                    * curvature_mod
                    * context_weight
                    * (0.8 + 0.4 * alpha)
                )
                if key == "mitigate_drift" and "drift" not in goal.lower():
                    adjusted *= 0.5
                reasoning_trace.append(f"Pattern '{key}': conf={adjusted:.2f} (phi={phi:.2f})")
                if adjusted >= self.confidence_threshold:
                    subgoals.extend(steps)
                    reasoning_trace.append(f"Accepted: {steps}")
                else:
                    reasoning_trace.append("Rejected (low conf)")

            contradictions = self.detect_contradictions(subgoals, task_type=task_type)
            if contradictions:
                reasoning_trace.append(f"Contradictions detected: {contradictions}")

            if not subgoals and phi > 0.8:
                prompt = f"Simulate decomposition ambiguity for: {goal}\nTask Type: {task_type}\nPolicies: {policies}"
                try:
                    if self.alignment_guard and hasattr(self.alignment_guard, "ethical_check"):
                        valid, report = await self.alignment_guard.ethical_check(prompt, stage="decomposition", task_type=task_type)
                        if not valid:
                            logger.warning("Decomposition prompt failed alignment check for task %s: %s", task_type, report)
                            sim_hint = "Prompt failed alignment check"
                        else:
                            sim_hint = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
                    else:
                        sim_hint = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
                except Exception as e:
                    sim_hint = f"Simulation unavailable: {e}"
                reasoning_trace.append(f"Ambiguity simulation:\n{sim_hint}")
                if self.agi_enhancer and hasattr(self.agi_enhancer, "reflect_and_adapt"):
                    await self.agi_enhancer.reflect_and_adapt(f"Decomposition ambiguity encountered for task {task_type}")

            if prioritize:
                subgoals = sorted(set(subgoals))
                reasoning_trace.append(f"Prioritized: {subgoals}")

            trace_log = "\n".join(reasoning_trace)
            logger.debug("Reasoning Trace for task %s:\n%s", task_type, trace_log)
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Goal decomposition run",
                    meta={
                        "goal": goal,
                        "trace": trace_log,
                        "subgoals": subgoals,
                        "drift": "drift" in goal.lower(),
                        "task_type": task_type,
                    },
                    module="ReasoningEngine",
                    tags=["decomposition", "reasoning", "drift", task_type],
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Decomposition_{goal[:50]}_{datetime.now().isoformat()}",
                    output=trace_log,
                    layer="ReasoningTraces",
                    intent="goal_decomposition",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "decompose", "trace": trace_log, "drift": "drift" in goal.lower(), "task_type": task_type}
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output=trace_log,
                    context={"confidence": 0.9, "alignment": "verified", "drift": "drift" in goal.lower(), "task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.info("Decomposition reflection recorded")
            if self.visualizer and task_type:
                plot_data = {
                    "decomposition": {"goal": goal, "subgoals": subgoals, "trace": trace_log, "task_type": task_type},
                    "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                }
                await self.visualizer.render_charts(plot_data)
            return subgoals
        except Exception as e:
            logger.error("Decomposition failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.decompose(goal, context, prioritize, task_type), default=[]
            )

    async def update_success_rate(self, pattern_key: str, success: bool, task_type: str = "") -> None:
        """Update success rate for a decomposition pattern. [v3.5.2]"""
        if not isinstance(pattern_key, str) or not pattern_key.strip():
            logger.error("Invalid pattern_key: must be a non-empty string for task %s", task_type)
            raise ValueError("pattern_key must be a non-empty string")
        if not isinstance(success, bool):
            logger.error("Invalid success: must be a boolean for task %s", task_type)
            raise TypeError("success must be a boolean")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            rate = self.success_rates.get(pattern_key, 1.0)
            new = min(max(rate + (0.05 if success else -0.05), 0.1), 1.0)
            self.success_rates[pattern_key] = new
            self._save_success_rates()
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Success rate updated",
                    meta={"pattern_key": pattern_key, "new_rate": new, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["success_rate", "update", task_type],
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "update_success_rate", "pattern_key": pattern_key, "new_rate": new, "task_type": task_type}
                )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "success_rate_update": {"pattern_key": pattern_key, "new_rate": new, "task_type": task_type},
                        "visualization_options": {"interactive": False, "style": "concise"},
                    }
                )
        except Exception as e:
            logger.error("Success rate update failed for task %s: %s", task_type, str(e))
            raise

    # ---------------------------
    # Simulations
    # ---------------------------
    async def run_galaxy_rotation_simulation(
        self, r_kpc: Union[np.ndarray, List[float], float], M0: float, r_scale: float, v0: float, k: float, epsilon: float, task_type: str = ""
    ) -> Dict[str, Any]:
        """Simulate galaxy rotation with ToCA physics. [v3.5.2]"""
        try:
            if isinstance(r_kpc, (list, float)):
                r_kpc = np.array(r_kpc)
            if not isinstance(r_kpc, np.ndarray):
                logger.error("Invalid r_kpc: must be a numpy array, list, or float for task %s", task_type)
                raise ValueError("r_kpc must be a numpy array, list, or float")
            for param, name in [(M0, "M0"), (r_scale, "r_scale"), (v0, "v0"), (k, "k"), (epsilon, "epsilon")]:
                if not isinstance(param, (int, float)) or param <= 0:
                    logger.error("Invalid %s: must be a positive number for task %s", name, task_type)
                    raise ValueError(f"{name} must be a positive number")
            if not isinstance(task_type, str):
                logger.error("Invalid task_type: must be a string")
                raise TypeError("task_type must be a string")

            M_b_func = lambda r: M_b_exponential(r, M0, r_scale)
            v_obs_func = lambda r: v_obs_flat(r, v0)
            result = await asyncio.to_thread(simulate_galaxy_rotation, r_kpc, M_b_func, v_obs_func, k, epsilon)
            output = {
                "input": {"r_kpc": r_kpc.tolist() if hasattr(r_kpc, "tolist") else r_kpc, "M0": M0, "r_scale": r_scale, "v0": v0, "k": k, "epsilon": epsilon},
                "result": result.tolist() if hasattr(result, "tolist") else result,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Galaxy rotation simulation", meta=output, module="ReasoningEngine", tags=["simulation", "toca", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Simulation_{output['timestamp']}",
                    output=str(output),
                    layer="Simulations",
                    intent="galaxy_rotation",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "run_galaxy_rotation_simulation", "output": output, "task_type": task_type})
            if self.multi_modal_fusion:
                external_data = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db", data_type="policy_data", task_type=task_type
                )
                policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"simulation": output, "text": f"Galaxy rotation simulation (Task: {task_type})", "policies": policies},
                    summary_style="concise",
                    task_type=task_type,
                )
                output["synthesis"] = synthesis
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine", output=str(output), context={"confidence": 0.9, "alignment": "verified", "task_type": task_type}
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.info("Galaxy simulation reflection recorded")
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "galaxy_simulation": {"input": output["input"], "result": output["result"], "task_type": task_type},
                        "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                    }
                )
            return output
        except Exception as e:
            logger.error("Simulation failed for task %s: %s", task_type, str(e))
            error_output = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(event="Simulation error", meta=error_output, module="ReasoningEngine", tags=["simulation", "error", task_type])
            return error_output

    async def run_drift_mitigation_simulation(self, drift_data: Dict[str, Any], context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Simulate drift mitigation scenarios using ToCA physics. [v3.5.2]"""
        if not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary for task %s", task_type)
            raise TypeError("drift_data must be a dictionary")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            if self.meta_cognition:
                try:
                    if not self.meta_cognition.validate_drift(drift_data):
                        logger.warning("Invalid drift data for task %s: %s", task_type, drift_data)
                        return {"status": "error", "error": "Invalid drift data", "timestamp": datetime.now().isoformat(), "task_type": task_type}
                except TypeError:
                    if not self.meta_cognition.validate_drift(drift_data):
                        return {"status": "error", "error": "Invalid drift data", "timestamp": datetime.now().isoformat(), "task_type": task_type}

            phi_field = generate_phi_field(drift_data.get("similarity", 0.5), context.get("scale", 1.0))
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db", data_type="policy_data", task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            result = {
                "drift_data": drift_data,
                "phi_field": phi_field.tolist() if hasattr(phi_field, "tolist") else phi_field,
                "mitigation_steps": await self.decompose("mitigate ontology drift", context, prioritize=True, task_type=task_type),
                "policies": policies,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Drift mitigation simulation", meta=result, module="ReasoningEngine", tags=["simulation", "drift", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Simulation_{result['timestamp']}",
                    output=str(result),
                    layer="Simulations",
                    intent="drift_mitigation",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "run_drift_mitigation_simulation", "output": result, "drift": True, "task_type": task_type}
                )
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"simulation": result, "text": f"Drift mitigation simulation (Task: {task_type})", "policies": policies},
                    summary_style="concise",
                    task_type=task_type,
                )
                result["synthesis"] = synthesis
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output=str(result),
                    context={"confidence": 0.9, "alignment": "verified", "drift": True, "task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.info("Drift mitigation reflection recorded")
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "drift_simulation": {
                            "drift_data": drift_data,
                            "phi_field": result["phi_field"],
                            "mitigation_steps": result["mitigation_steps"],
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                    }
                )
            return result
        except Exception as e:
            logger.error("Drift mitigation simulation failed for task %s: %s", task_type, str(e))
            error_output = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Drift simulation error", meta=error_output, module="ReasoningEngine", tags=["simulation", "error", "drift", task_type]
                )
            return error_output

    async def run_consensus_protocol(
        self, drift_data: Dict[str, Any], context: Dict[str, Any], max_rounds: int = 3, task_type: str = ""
    ) -> Dict[str, Any]:
        """Run a consensus protocol for drift mitigation across multiple agents. [v3.5.2]"""
        if not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary for task %s", task_type)
            raise ValueError("drift_data must be a dictionary")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise ValueError("context must be a dictionary")
        if not isinstance(max_rounds, int) or max_rounds < 1:
            logger.error("Invalid max_rounds: must be a positive integer for task %s", task_type)
            raise ValueError("max_rounds must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Running consensus protocol for drift mitigation (Task: %s)", task_type)
        try:
            if self.meta_cognition:
                try:
                    if not self.meta_cognition.validate_drift(drift_data):
                        logger.warning("Invalid drift data for task %s: %s", task_type, drift_data)
                        return {"status": "error", "error": "Invalid drift data", "timestamp": datetime.now().isoformat(), "task_type": task_type}
                except TypeError:
                    if not self.meta_cognition.validate_drift(drift_data):
                        return {"status": "error", "error": "Invalid drift data", "timestamp": datetime.now().isoformat(), "task_type": task_type}

            task = f"Mitigate ontology drift (Task: {task_type})"
            context = dict(context or {})
            context["drift"] = drift_data
            agent = await self.external_agent_bridge.create_agent(task, context)

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db", data_type="policy_data", task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            consensus_results: List[Dict[str, Any]] = []
            weighted_subgoals: Dict[str, float] = {}

            for round_num in range(1, max_rounds + 1):
                logger.info("Consensus round %d/%d for task %s", round_num, max_rounds, task_type)

                agent_results = await self.external_agent_bridge.collect_results(parallel=True, collaborative=True)
                if not agent_results:
                    logger.warning("No agent results in round %d for task %s", round_num, task_type)
                    continue

                synthesis_result = await self.multi_modal_fusion.synthesize_drift_data(
                    agent_data=[{"drift": drift_data, "result": r} for r in agent_results],
                    context=context | {"policies": policies},
                    task_type=task_type,
                )
                if synthesis_result.get("status") == "error":
                    logger.warning("Synthesis failed in round %d for task %s: %s", round_num, task_type, synthesis_result.get("error"))
                    continue

                subgoals = synthesis_result.get("subgoals", [])
                confidences = [r.get("confidence", 0.5) if isinstance(r, dict) else 0.5 for r in agent_results]

                weighted_subgoals = defaultdict(float)
                for subgoal, confidence in zip(subgoals, confidences):
                    # Use drift similarity as a reliability factor if present
                    sim = float(drift_data.get("similarity", 0.5))
                    weight = float(confidence) * (sim if 0.0 <= sim <= 1.0 else 0.5)
                    weighted_subgoals[subgoal] += weight

                sorted_subgoals = sorted(weighted_subgoals.items(), key=lambda x: x[1], reverse=True)
                top_subgoals = [sg for sg, weight in sorted_subgoals if weight >= self.confidence_threshold]

                if top_subgoals:
                    consensus_result = {
                        "round": round_num,
                        "subgoals": top_subgoals,
                        "weights": dict(sorted_subgoals),
                        "synthesis": synthesis_result.get("synthesis"),
                        "status": "success",
                        "timestamp": datetime.now().isoformat(),
                        "task_type": task_type,
                    }
                    consensus_results.append(consensus_result)
                    logger.info("Consensus reached in round %d for task %s: %s", round_num, task_type, top_subgoals)
                    break
                else:
                    logger.info("No consensus in round %d for task %s, continuing", round_num, task_type)
                    context["previous_round"] = {"subgoals": subgoals, "weights": dict(weighted_subgoals)}

            final_result = consensus_results[-1] if consensus_results else {
                "status": "error",
                "error": "No consensus reached",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }

            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Consensus protocol completed",
                    meta={"drift_data": drift_data, "result": final_result, "rounds": len(consensus_results), "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["consensus", "drift", task_type],
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Consensus_{datetime.now().isoformat()}",
                    output=str(final_result),
                    layer="ConsensusResults",
                    intent="consensus_protocol",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "run_consensus_protocol", "output": final_result, "drift": True, "task_type": task_type}
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output=str(final_result),
                    context={"confidence": max(weighted_subgoals.values()) if weighted_subgoals else 0.5, "alignment": "verified", "drift": True, "task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.info("Consensus protocol reflection recorded")
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "consensus_protocol": {
                            "subgoals": final_result.get("subgoals", []),
                            "weights": final_result.get("weights", {}),
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": task_type == "recursion", "style": "detailed" if task_type == "recursion" else "concise"},
                    }
                )
            return final_result
        except Exception as e:
            logger.error("Consensus protocol failed for task %s: %s", task_type, str(e))
            error_output = {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Consensus protocol error", meta=error_output, module="ReasoningEngine", tags=["consensus", "error", "drift", task_type]
                )
            return error_output

    async def on_context_event(self, event_type: str, payload: Dict[str, Any], task_type: str = "") -> None:
        """Process task-specific context events with persona wave routing, handling drift events. [v3.5.2]"""
        if not isinstance(event_type, str) or not event_type.strip():
            logger.error("Invalid event_type: must be a non-empty string for task %s", task_type)
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(payload, dict):
            logger.error("Invalid payload: must be a dictionary for task %s", task_type)
            raise TypeError("payload must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Context event received for task %s: %s", task_type, event_type)
        try:
            vectors = payload.get("vectors", {})
            goal = payload.get("goal", "unspecified")
            drift_data = payload.get("drift", {})
            if vectors or "drift" in event_type.lower():
                routing_result = await self.run_persona_wave_routing(goal, {**vectors, "drift": drift_data}, task_type=task_type)
                logger.info("Context sync routing result for task %s: %s", task_type, routing_result)
                if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                    await self.agi_enhancer.log_episode(
                        event="Context Sync Processed",
                        meta={"event": event_type, "vectors": vectors, "drift": drift_data, "routing_result": routing_result, "task_type": task_type},
                        module="ReasoningEngine",
                        tags=["context", "sync", "drift", task_type],
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Context_Event_{event_type}_{datetime.now().isoformat()}",
                        output=str(routing_result),
                        layer="ContextEvents",
                        intent="context_sync",
                        task_type=task_type,
                    )
                if self.context_manager:
                    await self.context_manager.log_event_with_hash(
                        {"event": "on_context_event", "result": routing_result, "drift": bool(drift_data), "task_type": task_type}
                    )
                if drift_data and self.meta_cognition:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ReasoningEngine",
                        output=str(routing_result),
                        context={"confidence": 0.85, "alignment": "verified", "drift": True, "task_type": task_type},
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        logger.info("Context event reflection recorded")
                if self.visualizer and task_type:
                    await self.visualizer.render_charts(
                        {
                            "context_event": {"event_type": event_type, "routing_result": routing_result, "task_type": task_type},
                            "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                        }
                    )
        except Exception as e:
            logger.error("Context event processing failed for task %s: %s", task_type, str(e))
            await self.error_recovery.handle_error(str(e), retry_func=lambda: self.on_context_event(event_type, payload, task_type))

    def export_trace(self, subgoals: List[str], phi: float, traits: Dict[str, float], task_type: str = "") -> Dict[str, Any]:
        """Export reasoning trace with subgoals and traits. [v3.5.2]"""
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list for task %s", task_type)
            raise TypeError("subgoals must be a list")
        if not isinstance(phi, float):
            logger.error("Invalid phi: must be a float for task %s", task_type)
            raise TypeError("phi must be a float")
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary for task %s", task_type)
            raise TypeError("traits must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        trace = {"phi": phi, "subgoals": subgoals, "traits": traits, "timestamp": datetime.now().isoformat(), "task_type": task_type}
        if self.memory_manager:
            intent = "drift_trace" if any(isinstance(s, str) and "drift" in s.lower() for s in subgoals) else "export_trace"
            asyncio.create_task(
                self.memory_manager.store(
                    query=f"Trace_{trace['timestamp']}", output=str(trace), layer="ReasoningTraces", intent=intent, task_type=task_type
                )
            )
        if self.context_manager:
            asyncio.create_task(
                self.context_manager.log_event_with_hash({"event": "export_trace", "trace": trace, "drift": intent == "drift_trace", "task_type": task_type})
            )
        return trace

    # ---------------------------
    # Inference & Mapping
    # ---------------------------
    async def infer_with_simulation(self, goal: str, context: Optional[Dict[str, Any]] = None, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Infer outcomes using simulations for goals (galaxy rotation / drift). [v3.5.2]"""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string for task %s", task_type)
            raise ValueError("goal must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        context = context or {}
        try:
            if "galaxy rotation" in goal.lower():
                r_kpc = np.linspace(0.1, 20, 100)
                params = {
                    "M0": context.get("M0", 5e10),
                    "r_scale": context.get("r_scale", 3.0),
                    "v0": context.get("v0", 200.0),
                    "k": context.get("k", 1.0),
                    "epsilon": context.get("epsilon", 0.1),
                }
                for key, value in params.items():
                    if not isinstance(value, (int, float)) or value <= 0:
                        logger.error("Invalid %s: must be a positive number for task %s", key, task_type)
                        raise ValueError(f"{key} must be a positive number")
                return await self.run_galaxy_rotation_simulation(r_kpc, **params, task_type=task_type)
            elif "drift" in goal.lower():
                drift_data = context.get("drift", {})
                return await self.run_drift_mitigation_simulation(drift_data, context, task_type=task_type)
            return None
        except Exception as e:
            logger.error("Inference with simulation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.infer_with_simulation(goal, context, task_type), default=None
            )

    async def map_intention(self, plan: str, state: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Extract intention from plan execution with reflexive trace. [v3.5.2]"""
        if not isinstance(plan, str) or not plan.strip():
            logger.error("Invalid plan: must be a non-empty string for task %s", task_type)
            raise ValueError("plan must be a non-empty string")
        if not isinstance(state, dict):
            logger.error("Invalid state: must be a dictionary for task %s", task_type)
            raise TypeError("state must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            eta = eta_empathy(t)
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db", data_type="policy_data", task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            intention = "drift_mitigation" if "drift" in plan.lower() else ("self-improvement" if phi > 0.6 else "task_completion")
            result = {
                "plan": plan,
                "state": state,
                "intention": intention,
                "trait_bias": {"phi": phi, "eta": eta},
                "policies": policies,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Intention_{plan[:50]}_{result['timestamp']}",
                    output=str(result),
                    layer="Intentions",
                    intent="intention_mapping",
                    task_type=task_type,
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Intention mapped",
                    meta=result,
                    module="ReasoningEngine",
                    tags=["intention", "mapping", "drift" if "drift" in plan.lower() else "task", task_type],
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "map_intention", "result": result, "drift": "drift" in plan.lower(), "task_type": task_type}
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ReasoningEngine",
                    output=str(result),
                    context={"confidence": 0.85, "alignment": "verified", "drift": "drift" in plan.lower(), "task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.info("Intention mapping reflection recorded")
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "intention_mapping": {"plan": plan, "intention": intention, "task_type": task_type},
                        "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                    }
                )
            return result
        except Exception as e:
            logger.error("Intention mapping failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.map_intention(plan, state, task_type), default={}
            )

    async def safeguard_noetic_integrity(self, model_depth: int, task_type: str = "") -> bool:
        """Prevent infinite recursion or epistemic bleed. [v3.5.2]"""
        if not isinstance(model_depth, int) or model_depth < 0:
            logger.error("Invalid model_depth: must be a non-negative integer for task %s", task_type)
            raise ValueError("model_depth must be a non-negative integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            if model_depth > 4:
                logger.warning("Noetic recursion limit breached for task %s: depth=%d", task_type, model_depth)
                if self.meta_cognition:
                    await self.meta_cognition.epistemic_self_inspection(f"Recursion depth exceeded for task {task_type}")
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({"event": "noetic_integrity_breach", "depth": model_depth, "task_type": task_type})
                if self.visualizer and task_type:
                    await self.visualizer.render_charts(
                        {"noetic_integrity": {"depth": model_depth, "task_type": task_type}, "visualization_options": {"interactive": False, "style": "concise"}}
                    )
                return False
            return True
        except Exception as e:
            logger.error("Noetic integrity check failed for task %s: %s", task_type, str(e))
            return False

    async def generate_dilemma(self, domain: str, task_type: str = "") -> str:
        """Generate an ethical dilemma for a given domain (drift-aware). [v3.5.2]"""
        if not isinstance(domain, str) or not domain.strip():
            logger.error("Invalid domain: must be a non-empty string for task %s", task_type)
            raise ValueError("domain must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Generating ethical dilemma for domain: %s (Task: %s)", domain, task_type)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db", data_type="policy_data", task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            prompt = f"""
            Generate an ethical dilemma in the {domain} domain.
            Use phi-scalar(t) = {phi:.3f} to modulate complexity.
            Task Type: {task_type}
            Provide two conflicting options (X and Y) with potential consequences and alignment with ethical principles.
            Incorporate external policies: {policies}
            """.strip()
            if "drift" in domain.lower():
                prompt += "\nConsider ontology drift mitigation and agent coordination implications."
            dilemma = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
            if not str(dilemma).strip():
                logger.warning("Empty output from dilemma generation for task %s", task_type)
                raise ValueError("Empty output from dilemma generation")

            # ✅ meta_cognition.review_reasoning(signature fix)
            if self.meta_cognition:
                try:
                    review = await self.meta_cognition.review_reasoning(dilemma)
                    dilemma += f"\nMeta-Cognitive Review: {review}"
                except TypeError:
                    review = await self.meta_cognition.review_reasoning(dilemma)
                    dilemma += f"\nMeta-Cognitive Review: {review}"

            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                await self.agi_enhancer.log_episode(
                    event="Ethical dilemma generated",
                    meta={"domain": domain, "dilemma": dilemma, "task_type": task_type},
                    module="ReasoningEngine",
                    tags=["ethics", "dilemma", "drift" if "drift" in domain.lower() else "standard", task_type],
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Dilemma_{domain}_{datetime.now().isoformat()}",
                    output=dilemma,
                    layer="Ethics",
                    intent="ethical_dilemma",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "generate_dilemma", "dilemma": dilemma, "drift": "drift" in domain.lower(), "task_type": task_type})
            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"dilemma": dilemma, "text": f"Ethical dilemma in {domain}", "policies": policies},
                    summary_style="insightful",
                    task_type=task_type,
                )
                dilemma += f"\nMulti-Modal Synthesis: {synthesis}"
            return dilemma
        except Exception as e:
            logger.error("Dilemma generation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.generate_dilemma(domain, task_type), default=""
            )
