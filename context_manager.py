"""
ANGELA Cognitive System Module: ContextManager
Version: 3.5.3  # Υ SharedGraph hooks, Self-Healing, and Φ⁰ hooks (env-gated)
Date: 2025-08-10
Maintainer: ANGELA System Framework

Changes vs 3.5.1 / 3.5.2:
- Υ Meta-Subjective Architecting: SharedGraph add/diff/merge support for inter‑agent context reconciliation
- Self-Healing Cognitive Pathways: tighter integration with error_recovery + recursive_planner for auto-repair
- Safer external context integration: pluggable providers + caching via memory_manager (no blind external calls)
- Φ⁰ Reality Sculpting hooks (gated by STAGE_IV env flag)
- Fixed: event timestamp key typo; robust persistence & hashing; drift-validation edges; vector normalization
- Important: SharedGraph is synchronous (add/diff/merge) — no awaits on these calls
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from collections import Counter, deque
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from filelock import FileLock

# ── Optional module wiring (duck-typed) ──────────────────────────────────────
# These imports reflect the ANGELA repo layout. If your project structures modules
# differently, adapt the import paths accordingly.
from modules import (
    agi_enhancer as agi_enhancer_module,
    alignment_guard as alignment_guard_module,
    code_executor as code_executor_module,
    concept_synthesizer as concept_synthesizer_module,
    error_recovery as error_recovery_module,
    external_agent_bridge as external_agent_bridge_module,
    knowledge_retriever as knowledge_retriever_module,
    meta_cognition as meta_cognition_module,
    recursive_planner as recursive_planner_module,
    visualizer as visualizer_module,
)

# Utilities (keep names consistent with repo utility modules)
from utils.toca_math import phi_coherence
from utils.vector_utils import normalize_vectors
from toca_simulation import run_simulation
from index import omega_selfawareness, eta_empathy, tau_timeperception

logger = logging.getLogger("ANGELA.ContextManager")


# ── Trait helper ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=100)
def eta_context_stability(t: float) -> float:
    """Trait function for context stability modulation (bounded [0,1])."""
    # Low-amplitude cosine over short horizon to favor stability bursts
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.2), 1.0))


# ── Env flags (env overrides manifest) ───────────────────────────────────────
def _flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


STAGE_IV = _flag("STAGE_IV", default=False)  # Φ⁰ hooks gated by env


class ContextManager:
    """Manage contextual state, inter‑agent reconciliation, logs, analytics, and gated Φ⁰ hooks."""

    CONTEXT_LAYERS = ["local", "societal", "planetary"]

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        alignment_guard: Optional["alignment_guard_module.AlignmentGuard"] = None,
        code_executor: Optional["code_executor_module.CodeExecutor"] = None,
        concept_synthesizer: Optional["concept_synthesizer_module.ConceptSynthesizer"] = None,
        meta_cognition: Optional["meta_cognition_module.MetaCognition"] = None,
        visualizer: Optional["visualizer_module.Visualizer"] = None,
        error_recovery: Optional["error_recovery_module.ErrorRecovery"] = None,
        recursive_planner: Optional["recursive_planner_module.RecursivePlanner"] = None,
        shared_graph: Optional["external_agent_bridge_module.SharedGraph"] = None,
        knowledge_retriever: Optional["knowledge_retriever_module.KnowledgeRetriever"] = None,
        context_path: str = "context_store.json",
        event_log_path: str = "event_log.json",
        coordination_log_path: str = "coordination_log.json",
        rollback_threshold: float = 2.5,
        # Optional provider for safe, local external context (no blind network I/O)
        external_context_provider: Optional[Callable[[str, str, str], Dict[str, Any]]] = None,
    ):
        # ── Validations ──
        for p, nm in [
            (context_path, "context_path"),
            (event_log_path, "event_log_path"),
            (coordination_log_path, "coordination_log_path"),
        ]:
            if not isinstance(p, str) or not p.endswith(".json"):
                logger.error("Invalid %s: must be a string ending with '.json'.", nm)
                raise ValueError(f"{nm} must be a string ending with '.json'")
        if not isinstance(rollback_threshold, (int, float)) or rollback_threshold <= 0:
            logger.error("Invalid rollback_threshold: must be positive.")
            raise ValueError("rollback_threshold must be a positive number")

        # ── State ──
        self.context_path = context_path
        self.event_log_path = event_log_path
        self.coordination_log_path = coordination_log_path
        self.current_context: Dict[str, Any] = {}
        self.context_history: deque = deque(maxlen=1000)
        self.event_log: deque = deque(maxlen=1000)
        self.coordination_log: deque = deque(maxlen=1000)
        self.last_hash = ""

        # ── Components (duck-typed) ──
        self.agi_enhancer = (
            agi_enhancer_module.AGIEnhancer(orchestrator) if orchestrator else None
        )
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition()
        self.visualizer = visualizer or visualizer_module.Visualizer()
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.recursive_planner = recursive_planner
        self.shared_graph = shared_graph  # Υ hooks (synchronous API)
        self.knowledge_retriever = knowledge_retriever
        self.external_context_provider = external_context_provider

        self.rollback_threshold = rollback_threshold

        # ── Bootstrap ──
        self.current_context = self._load_context()
        self._load_event_log()
        self._load_coordination_log()
        logger.info(
            "ContextManager v3.5.3 initialized (Υ+SelfHealing%s), rollback_threshold=%.2f",
            " + Φ⁰" if STAGE_IV else "",
            rollback_threshold,
        )

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load_context(self) -> Dict[str, Any]:
        try:
            with FileLock(f"{self.context_path}.lock"):
                if os.path.exists(self.context_path):
                    with open(self.context_path, "r", encoding="utf-8") as f:
                        context = json.load(f)
                    if not isinstance(context, dict):
                        logger.error("Invalid context file format.")
                        context = {}
                else:
                    context = {}
            logger.debug("Loaded context: %s", context)
            return context
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to load context file: %s. Initializing empty context.", str(e)
            )
            context = {}
            self._persist_context(context)
            return context

    def _load_event_log(self) -> None:
        try:
            with FileLock(f"{self.event_log_path}.lock"):
                if os.path.exists(self.event_log_path):
                    with open(self.event_log_path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                    if not isinstance(events, list):
                        logger.error("Invalid event log format.")
                        events = []
                    self.event_log.extend(events[-1000:])
                    if events:
                        self.last_hash = events[-1].get("hash", "")
                else:
                    with open(self.event_log_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load event log: %s. Initializing empty log.", str(e))
            with FileLock(f"{self.event_log_path}.lock"):
                with open(self.event_log_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    def _load_coordination_log(self) -> None:
        try:
            with FileLock(f"{self.coordination_log_path}.lock"):
                if os.path.exists(self.coordination_log_path):
                    with open(self.coordination_log_path, "r", encoding="utf-8") as f:
                        events = json.load(f)
                    if not isinstance(events, list):
                        logger.error("Invalid coordination log format.")
                        events = []
                    self.coordination_log.extend(events[-1000:])
                else:
                    with open(self.coordination_log_path, "w", encoding="utf-8") as f:
                        json.dump([], f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(
                "Failed to load coordination log: %s. Initializing empty log.", str(e)
            )
            with FileLock(f"{self.coordination_log_path}.lock"):
                with open(self.coordination_log_path, "w", encoding="utf-8") as f:
                    json.dump([], f)

    def _persist_context(self, context: Dict[str, Any]) -> None:
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dict.")
            raise TypeError("context must be a dictionary")
        try:
            with FileLock(f"{self.context_path}.lock"):
                with open(self.context_path, "w", encoding="utf-8") as f:
                    json.dump(context, f, indent=2)
        except (OSError, IOError) as e:
            logger.error("Failed to persist context: %s", str(e))
            raise

    def _persist_event_log(self) -> None:
        try:
            with FileLock(f"{self.event_log_path}.lock"):
                with open(self.event_log_path, "w", encoding="utf-8") as f:
                    json.dump(list(self.event_log), f, indent=2)
        except (OSError, IOError) as e:
            logger.error("Failed to persist event log: %s", str(e))
            raise

    def _persist_coordination_log(self) -> None:
        try:
            with FileLock(f"{self.coordination_log_path}.lock"):
                with open(self.coordination_log_path, "w", encoding="utf-8") as f:
                    json.dump(list(self.coordination_log), f, indent=2)
        except (OSError, IOError) as e:
            logger.error("Failed to persist coordination log: %s", str(e))
            raise

    # ── External context integration (safe & pluggable) ───────────────────────
    async def integrate_external_context_data(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Integrate external policies or coordination metadata:
        - Uses MetaCognition.memory_manager for caching
        - Pulls from a provided callable OR knowledge_retriever (no blind network)
        Supported data_type: "context_policies", "coordination_data"
        """
        if not all(isinstance(x, str) for x in [data_source, data_type]):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            cache_key = f"ContextData::{data_type}::{data_source}::{task_type or 'global'}"
            # 1) Cache first
            if self.meta_cognition:
                cached = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached and "timestamp" in cached.get("data", {}):
                    ts = datetime.fromisoformat(cached["data"]["timestamp"])
                    if (datetime.now() - ts).total_seconds() < cache_timeout:
                        return cached["data"]["data"]

            # 2) Provider pipeline (callable > knowledge_retriever > empty)
            if callable(self.external_context_provider):
                data = self.external_context_provider(data_source, data_type, task_type)
            elif self.knowledge_retriever:
                try:
                    data = await self.knowledge_retriever.fetch(
                        data_source, data_type, task_type=task_type
                    )
                except Exception as e:
                    logger.warning("knowledge_retriever.fetch failed: %s", e)
                    data = {}
            else:
                data = {}

            if data_type == "context_policies":
                policies = data.get("policies", [])
                result = (
                    {"status": "success", "policies": policies}
                    if policies
                    else {"status": "error", "error": "No policies"}
                )
            elif data_type == "coordination_data":
                coordination = data.get("coordination", {})
                result = (
                    {"status": "success", "coordination": coordination}
                    if coordination
                    else {"status": "error", "error": "No coordination data"}
                )
            else:
                result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            # 3) Cache store
            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="context_data_integration",
                    task_type=task_type,
                )
                if task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ContextManager",
                        output={"data_type": data_type, "data": result},
                        context={"task_type": task_type},
                    )
                    if reflection.get("status") == "success":
                        logger.info(
                            "Integration reflection: %s",
                            reflection.get("reflection", ""),
                        )

            return result
        except Exception as e:
            logger.error("Context data integration failed: %s", str(e))
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.integrate_external_context_data(
                    data_source, data_type, cache_timeout, task_type
                ),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
                task_type=task_type,
            )

    # ── Core updates ──────────────────────────────────────────────────────────
    async def update_context(self, new_context: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(new_context, dict):
            raise TypeError("new_context must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Updating context for task %s", task_type)
        try:
            # Validate drift/trait ops
            if self.meta_cognition and any(
                k in new_context for k in ["drift", "trait_optimization", "trait_optimizations"]
            ):
                drift_data = (
                    new_context.get("drift")
                    or new_context.get("trait_optimization")
                    or new_context.get("trait_optimizations")
                )
                if drift_data and not await self.meta_cognition.validate_drift(
                    drift_data, task_type=task_type
                ):
                    raise ValueError("Drift or trait context failed validation")

            # Simulate transition & compute Φ
            phi_score = 1.0
            simulation_result = "no simulation data"
            if self.current_context:
                transition_summary = f"From: {self.current_context}\nTo: {new_context}"
                simulation_result = await asyncio.to_thread(
                    run_simulation, f"Context shift evaluation:\n{transition_summary}"
                ) or "no simulation data"
                phi_score = phi_coherence(self.current_context, new_context)

                if phi_score < 0.4:
                    if self.agi_enhancer:
                        await self.agi_enhancer.reflect_and_adapt(
                            f"Low Φ during context update (task={task_type})"
                        )
                        await self.agi_enhancer.trigger_reflexive_audit(
                            f"Low Φ during context update (task={task_type})"
                        )
                    if self.meta_cognition:
                        optimizations = await self.meta_cognition.propose_trait_optimizations(
                            {"phi_score": phi_score}, task_type=task_type
                        )
                        new_context.setdefault("trait_optimizations", optimizations)

                if self.alignment_guard:
                    valid, _report = await self.alignment_guard.ethical_check(
                        str(new_context), stage="context_update", task_type=task_type
                    )
                    if not valid:
                        raise ValueError("New context failed alignment check")

                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        "Context Update",
                        {
                            "from": self.current_context,
                            "to": new_context,
                            "task_type": task_type,
                        },
                        module="ContextManager",
                        tags=["context", "update", task_type],
                    )
                    await self.agi_enhancer.log_explanation(
                        f"Context transition reviewed.\nSimulation: {simulation_result}",
                        trace={"phi": phi_score, "task_type": task_type},
                    )

            # Normalize vectors if present
            if "vectors" in new_context:
                new_context["vectors"] = normalize_vectors(new_context["vectors"])

            # Pull policies (safe path)
            context_data = await self.integrate_external_context_data(
                data_source="xai_context_db",
                data_type="context_policies",
                task_type=task_type,
            )
            if context_data.get("status") == "success":
                new_context["policies"] = context_data.get("policies", [])

            # Apply switch
            self.context_history.append(self.current_context)
            self.current_context = new_context
            self._persist_context(self.current_context)
            await self.log_event_with_hash(
                {"event": "context_updated", "context": new_context, "phi": phi_score},
                task_type=task_type,
            )
            await self.broadcast_context_event(
                "context_updated", new_context, task_type=task_type
            )

            # Υ: publish to SharedGraph for peer reconciliation (sync API)
            self._push_to_shared_graph(task_type=task_type)

            # Φ⁰ (gated): reality-sculpting hook (no-ops if disabled)
            if STAGE_IV:
                await self._reality_sculpt_hook(
                    "context_update", payload={"phi": phi_score, "task": task_type}
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"context": new_context, "phi_score": phi_score},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Context update reflection: %s",
                        reflection.get("reflection", ""),
                    )

        except Exception as e:
            logger.error("Context update failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.update_context(new_context, task_type),
                default=None,
                diagnostics=diagnostics,
                task_type=task_type,
                propose_plan=True,
            )

    async def tag_context(
        self, intent: Optional[str] = None, goal_id: Optional[str] = None, task_type: str = ""
    ) -> None:
        if intent is not None and not isinstance(intent, str):
            raise TypeError("intent must be a string or None")
        if goal_id is not None and not isinstance(goal_id, str):
            raise TypeError("goal_id must be a string or None")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Tagging context intent='%s', goal_id='%s' (task=%s)", intent, goal_id, task_type)
        try:
            if intent and self.alignment_guard:
                valid, _report = await self.alignment_guard.ethical_check(
                    intent, stage="context_tagging", task_type=task_type
                )
                if not valid:
                    raise ValueError("Intent failed alignment check")

            if intent:
                self.current_context["intent"] = intent
            if goal_id:
                self.current_context["goal_id"] = goal_id
            self.current_context["task_type"] = task_type
            self._persist_context(self.current_context)
            await self.log_event_with_hash(
                {"event": "context_tagged", "intent": intent, "goal_id": goal_id},
                task_type=task_type,
            )

            # Υ: publish tag update to SharedGraph (sync)
            self._push_to_shared_graph(task_type=task_type)

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"intent": intent, "goal_id": goal_id},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Context tagging reflection: %s", reflection.get("reflection", "")
                    )
        except Exception as e:
            logger.error("Context tagging failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.tag_context(intent, goal_id, task_type),
                default=None,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    def get_context_tags(self) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        return (
            self.current_context.get("intent"),
            self.current_context.get("goal_id"),
            self.current_context.get("task_type"),
        )

    async def rollback_context(self, task_type: str = "") -> Optional[Dict[str, Any]]:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        if not self.context_history:
            logger.warning("No previous context to roll back to (task=%s)", task_type)
            return None

        t = time.time()
        self_awareness = omega_selfawareness(t)
        empathy = eta_empathy(t)
        time_blend = tau_timeperception(t)
        stability = eta_context_stability(t)
        threshold = self.rollback_threshold * (1.0 + stability)

        if (self_awareness + empathy + time_blend) > threshold:
            restored = self.context_history.pop()
            self.current_context = restored
            self._persist_context(self.current_context)
            await self.log_event_with_hash(
                {"event": "context_rollback", "restored": restored}, task_type=task_type
            )
            await self.broadcast_context_event(
                "context_rollback", restored, task_type=task_type
            )
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Context Rollback",
                    {"restored": restored, "task_type": task_type},
                    module="ContextManager",
                    tags=["context", "rollback", task_type],
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"restored": restored},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Context rollback reflection: %s",
                        reflection.get("reflection", ""),
                    )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "context_rollback": {
                            "restored_context": restored,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            # Υ: publish rollback to SharedGraph
            self._push_to_shared_graph(task_type=task_type)
            return restored
        else:
            logger.warning(
                "EEG thresholds too low for safe rollback (%.2f < %.2f) (task=%s)",
                self_awareness + empathy + time_blend,
                threshold,
                task_type,
            )
            if self.agi_enhancer:
                await self.agi_enhancer.reflect_and_adapt(
                    f"Rollback gate low (task={task_type})"
                )
            return None

    async def summarize_context(self, task_type: str = "") -> str:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Summarizing context trail (task=%s)", task_type)
        try:
            t = time.time()
            summary_traits = {
                "self_awareness": omega_selfawareness(t),
                "empathy": eta_empathy(t),
                "time_perception": tau_timeperception(t),
                "context_stability": eta_context_stability(t),
            }

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name="ContextSummary",
                    context={"history": list(self.context_history), "current": self.current_context},
                    task_type=task_type,
                )
                summary = (
                    synthesis_result["concept"].get("definition", "Synthesis failed")
                    if synthesis_result.get("success")
                    else "Synthesis failed"
                )
            else:
                prompt = f"""
                You are a continuity analyst. Given this sequence of context states:
                {list(self.context_history) + [self.current_context]}

                Trait Readings:
                {summary_traits}

                Task: {task_type}
                Summarize the trajectory and suggest improvements in context management.
                """
                summary = await asyncio.to_thread(self._cached_call_gpt, prompt)

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Context Summary",
                    {
                        "trail": list(self.context_history) + [self.current_context],
                        "traits": summary_traits,
                        "summary": summary,
                        "task_type": task_type,
                    },
                    module="ContextManager",
                    tags=["context", "summary", task_type],
                )
                await self.agi_enhancer.log_explanation(
                    f"Context summary generated (task={task_type}).", trace={"summary": summary}
                )

            await self.log_event_with_hash(
                {"event": "context_summary", "summary": summary}, task_type=task_type
            )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"summary": summary, "traits": summary_traits},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Context summary reflection: %s", reflection.get("reflection", "")
                    )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "context_summary": {
                            "summary": summary,
                            "traits": summary_traits,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return summary
        except Exception as e:
            logger.error("Context summary failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.summarize_context(task_type),
                default=f"Summary failed: {str(e)}",
                diagnostics=diagnostics,
                task_type=task_type,
            )

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        from utils.prompt_utils import call_gpt

        return call_gpt(prompt)

    async def log_event_with_hash(self, event_data: Any, task_type: str = "") -> None:
        if not isinstance(event_data, dict):
            raise TypeError("event_data must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            # Validate consensus drift if present
            if self.meta_cognition and event_data.get("event") == "run_consensus_protocol":
                output = event_data.get("output", {})
                if output.get("status") == "success" and not await self.meta_cognition.validate_drift(
                    output.get("drift_data", {}), task_type=task_type
                ):
                    raise ValueError("Consensus event failed drift validation")

            # Attach agent metadata for coord-like events
            if any(
                k in event_data
                for k in [
                    "drift",
                    "trait_optimization",
                    "trait_optimizations",
                    "agent_coordination",
                    "run_consensus_protocol",
                ]
            ):
                event_data["agent_metadata"] = event_data.get("agent_metadata", {})
                if (
                    event_data.get("event") == "run_consensus_protocol"
                    and event_data.get("output")
                ):
                    event_data["agent_metadata"]["agent_ids"] = event_data["agent_metadata"].get(
                        "agent_ids", []
                    )
                    event_data["agent_metadata"]["confidence_scores"] = event_data["output"].get(
                        "weights", {}
                    )

            event_data["task_type"] = task_type
            event_str = json.dumps(event_data, sort_keys=True, default=str) + self.last_hash
            current_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
            event_entry = {
                "event": event_data,
                "hash": current_hash,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            self.event_log.append(event_entry)
            self.last_hash = current_hash
            self._persist_event_log()

            # Mirror to coordination log if relevant
            if any(
                k in event_data
                for k in [
                    "drift",
                    "trait_optimization",
                    "trait_optimizations",
                    "agent_coordination",
                    "run_consensus_protocol",
                ]
            ):
                coordination_entry = {
                    "event": event_data,
                    "hash": current_hash,
                    "timestamp": event_entry["timestamp"],
                    "type": (
                        "drift"
                        if "drift" in event_data
                        else "trait_optimization"
                        if "trait_optimization" in event_data
                        or "trait_optimizations" in event_data
                        else "agent_coordination"
                    ),
                    "agent_metadata": event_data.get("agent_metadata", {}),
                    "task_type": task_type,
                }
                self.coordination_log.append(coordination_entry)
                self._persist_coordination_log()
                if self.agi_enhancer:
                    await self.agi_enhancer.log_episode(
                        "Coordination Event",
                        coordination_entry,
                        module="ContextManager",
                        tags=["coordination", coordination_entry["type"], task_type],
                    )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=event_entry, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Event logging reflection: %s", reflection.get("reflection", ""))

        except Exception as e:
            logger.error("Event logging failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.log_event_with_hash(event_data, task_type),
                default=None,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def broadcast_context_event(
        self, event_type: str, payload: Any, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(event_type, str):
            raise TypeError("event_type must be a string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Broadcasting context event: %s (task=%s)", event_type, task_type)
        try:
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Context Event Broadcast",
                    {"event": event_type, "payload": payload, "task_type": task_type},
                    module="ContextManager",
                    tags=["event", event_type, task_type],
                )

            payload_str = str(payload).lower()
            if any(k in payload_str for k in ["drift", "trait_optimization", "agent", "consensus"]):
                coordination_entry = {
                    "event": event_type,
                    "payload": payload,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "type": "drift" if "drift" in payload_str else "agent_coordination",
                    "agent_metadata": payload.get("agent_metadata", {}) if isinstance(payload, dict) else {},
                    "task_type": task_type,
                }
                self.coordination_log.append(coordination_entry)
                self._persist_coordination_log()

            await self.log_event_with_hash(
                {"event": event_type, "payload": payload}, task_type=task_type
            )
            result = {"event": event_type, "payload": payload, "task_type": task_type}

            # Υ: propagate event snapshots to SharedGraph
            self._push_to_shared_graph(task_type=task_type)

            # Φ⁰ (gated) hook
            if STAGE_IV:
                await self._reality_sculpt_hook(
                    "context_event", payload={"event": event_type, "task": task_type}
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=result, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Broadcast reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "context_event_broadcast": {
                            "event_type": event_type,
                            "payload": payload,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return result
        except Exception as e:
            logger.error("Broadcast failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.broadcast_context_event(event_type, payload, task_type),
                default={"event": event_type, "error": str(e), "task_type": task_type},
                diagnostics=diagnostics,
                task_type=task_type,
            )

    # ── Narrative integrity & repair ──────────────────────────────────────────
    async def narrative_integrity_check(self, task_type: str = "") -> bool:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            continuity = await self._verify_continuity(task_type)
            if not continuity:
                await self._repair_narrative_thread(task_type)
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"continuity": continuity},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Narrative integrity reflection: %s",
                        reflection.get("reflection", ""),
                    )
            return continuity
        except Exception as e:
            logger.error("Narrative integrity check failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.narrative_integrity_check(task_type),
                default=False,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def _verify_continuity(self, task_type: str = "") -> bool:
        if not self.context_history:
            return True
        try:
            required = {"intent", "goal_id", "task_type"} if task_type else {"intent", "goal_id"}
            for ctx in self.context_history:
                if not isinstance(ctx, dict) or not required.issubset(ctx.keys()):
                    logger.warning("Continuity missing keys (task=%s)", task_type)
                    return False
                if any(k in ctx for k in ["drift", "trait_optimization", "trait_optimizations"]):
                    data = (
                        ctx.get("drift")
                        or ctx.get("trait_optimization")
                        or ctx.get("trait_optimizations")
                    )
                    if self.meta_cognition and not await self.meta_cognition.validate_drift(
                        data, task_type=task_type
                    ):
                        logger.warning("Continuity invalid drift/traits (task=%s)", task_type)
                        return False

            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "narrative_continuity": {
                            "continuity_status": True,
                            "context_history_length": len(self.context_history),
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return True
        except Exception as e:
            logger.error("Continuity verification failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self._verify_continuity(task_type),
                default=False,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def _repair_narrative_thread(self, task_type: str = "") -> None:
        logger.info("Narrative repair initiated (task=%s)", task_type)
        try:
            if self.context_history:
                last_valid = None
                for ctx in reversed(self.context_history):
                    if any(k in ctx for k in ["drift", "trait_optimization", "trait_optimizations"]):
                        data = (
                            ctx.get("drift")
                            or ctx.get("trait_optimization")
                            or ctx.get("trait_optimizations")
                        )
                        if self.meta_cognition and await self.meta_cognition.validate_drift(
                            data, task_type=task_type
                        ):
                            last_valid = ctx
                            break
                    else:
                        last_valid = ctx
                        break

                if last_valid is not None:
                    self.current_context = last_valid
                    self._persist_context(self.current_context)
                    await self.log_event_with_hash(
                        {"event": "narrative_repair", "restored": self.current_context},
                        task_type=task_type,
                    )
                    if self.visualizer and task_type:
                        await self.visualizer.render_charts(
                            {
                                "narrative_repair": {
                                    "restored_context": self.current_context,
                                    "task_type": task_type,
                                },
                                "visualization_options": {
                                    "interactive": task_type == "recursion",
                                    "style": "detailed"
                                    if task_type == "recursion"
                                    else "concise",
                                },
                            }
                        )
                else:
                    self.current_context = {}
                    self._persist_context(self.current_context)
                    await self.log_event_with_hash(
                        {"event": "narrative_repair", "restored": {}}, task_type=task_type
                    )
            else:
                self.current_context = {}
                self._persist_context(self.current_context)
                await self.log_event_with_hash(
                    {"event": "narrative_repair", "restored": {}}, task_type=task_type
                )

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"restored_context": self.current_context},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info(
                        "Narrative repair reflection: %s", reflection.get("reflection", "")
                    )
        except Exception as e:
            logger.error("Narrative repair failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self._repair_narrative_thread(task_type),
                default=None,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def bind_contextual_thread(self, thread_id: str, task_type: str = "") -> bool:
        if not isinstance(thread_id, str):
            raise TypeError("thread_id must be a string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Binding context thread: %s (task=%s)", thread_id, task_type)
        try:
            self.current_context["thread_id"] = thread_id
            self.current_context["task_type"] = task_type
            self._persist_context(self.current_context)
            await self.log_event_with_hash(
                {"event": "context_thread_bound", "thread_id": thread_id}, task_type=task_type
            )
            # Υ: publish
            self._push_to_shared_graph(task_type=task_type)

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"thread_id": thread_id},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Thread binding reflection: %s", reflection.get("reflection", ""))
            return True
        except Exception as e:
            logger.error("Thread binding failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.bind_contextual_thread(thread_id, task_type),
                default=False,
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def audit_state_hash(self, state: Optional[Any] = None, task_type: str = "") -> str:
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            state_str = (
                json.dumps(state, sort_keys=True, default=str)
                if state is not None
                else json.dumps(self._safe_state_snapshot(), sort_keys=True, default=str)
            )
            current_hash = hashlib.sha256(state_str.encode("utf-8")).hexdigest()
            await self.log_event_with_hash(
                {"event": "state_hash_audit", "hash": current_hash}, task_type=task_type
            )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"hash": current_hash},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("State hash audit reflection: %s", reflection.get("reflection", ""))
            return current_hash
        except Exception as e:
            logger.error("State hash computation failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.audit_state_hash(state, task_type),
                default="",
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def get_coordination_events(
        self, event_type: Optional[str] = None, task_type: str = ""
    ) -> List[Dict[str, Any]]:
        if event_type is not None and not isinstance(event_type, str):
            raise TypeError("event_type must be a string or None")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        try:
            results = [e for e in self.coordination_log if task_type == "" or e.get("task_type") == task_type]
            if event_type:
                results = [e for e in results if e["type"] == event_type]
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager",
                    output={"event_count": len(results), "event_type": event_type},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Coord events reflection: %s", reflection.get("reflection", ""))
            return results
        except Exception as e:
            logger.error("Coordination retrieval failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.get_coordination_events(event_type, task_type),
                default=[],
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def analyze_coordination_events(
        self, event_type: Optional[str] = None, task_type: str = ""
    ) -> Dict[str, Any]:
        if event_type is not None and not isinstance(event_type, str):
            raise TypeError("event_type must be a string or None")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")
        try:
            events = await self.get_coordination_events(event_type, task_type)
            if not events:
                return {
                    "status": "error",
                    "error": "No coordination events found",
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                }

            drift_count = sum(1 for e in events if e["type"] == "drift")
            consensus_events = [
                e for e in events if e["event"].get("event") == "run_consensus_protocol"
            ]
            consensus_count = sum(
                1 for e in consensus_events if e["event"].get("output", {}).get("status") == "success"
            )

            agent_counts = Counter(
                [
                    agent_id
                    for e in events
                    for agent_id in e["agent_metadata"].get("agent_ids", [])
                ]
            )
            avg_confidence = (
                np.mean(
                    [
                        (sum(conf.values()) / len(conf)) if conf else 0.5
                        for e in consensus_events
                        for conf in [e["event"]["output"].get("weights", {})]
                    ]
                )
                if consensus_events
                else 0.5
            )

            analysis = {
                "status": "success",
                "metrics": {
                    "drift_frequency": drift_count / len(events),
                    "consensus_success_rate": consensus_count / len(consensus_events)
                    if consensus_events
                    else 0.0,
                    "agent_participation": dict(agent_counts),
                    "avg_confidence_score": float(avg_confidence),
                    "event_count": len(events),
                },
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Coordination Analysis",
                    analysis,
                    module="ContextManager",
                    tags=["coordination", "analytics", event_type or "all", task_type],
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=analysis, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Coord analysis reflection: %s", reflection.get("reflection", ""))
            await self.log_event_with_hash(
                {"event": "coordination_analysis", "analysis": analysis}, task_type=task_type
            )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "coordination_analysis": {
                            "metrics": analysis["metrics"],
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return analysis
        except Exception as e:
            logger.error("Coordination analysis failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.analyze_coordination_events(event_type, task_type),
                default={
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                },
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def get_drift_trends(
        self, time_window_hours: float = 24.0, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            raise ValueError("time_window_hours must be positive")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            events = await self.get_coordination_events("drift", task_type)
            if not events:
                return {
                    "status": "error",
                    "error": "No drift events found",
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                }

            now = datetime.now()
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= cutoff]

            drift_names = Counter(
                e["event"].get("drift", {}).get("name", "unknown") for e in events
            )
            similarity_scores = [
                e["event"].get("drift", {}).get("similarity", 0.5)
                for e in events
                if "drift" in e["event"] and "similarity" in e["event"]["drift"]
            ]
            trend_data = {
                "status": "success",
                "trends": {
                    "drift_names": dict(drift_names),
                    "avg_similarity": float(np.mean(similarity_scores))
                    if similarity_scores
                    else 0.5,
                    "event_count": len(events),
                    "time_window_hours": time_window_hours,
                },
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Drift Trends Analysis",
                    trend_data,
                    module="ContextManager",
                    tags=["drift", "trends", task_type],
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=trend_data, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Drift trends reflection: %s", reflection.get("reflection", ""))
            await self.log_event_with_hash(
                {"event": "drift_trends", "trends": trend_data}, task_type=task_type
            )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "drift_trends": {
                            "trends": trend_data["trends"],
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            return trend_data
        except Exception as e:
            logger.error("Drift trends analysis failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.get_drift_trends(time_window_hours, task_type),
                default={
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                },
                diagnostics=diagnostics,
                task_type=task_type,
            )

    async def generate_coordination_chart(
        self, metric: str = "drift_frequency", time_window_hours: float = 24.0, task_type: str = ""
    ) -> Dict[str, Any]:
        if metric not in ["drift_frequency", "consensus_success_rate", "avg_confidence_score"]:
            raise ValueError(
                "metric must be 'drift_frequency', 'consensus_success_rate', or 'avg_confidence_score'"
            )
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            raise ValueError("time_window_hours must be positive")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            events = await self.get_coordination_events(task_type=task_type)
            if not events:
                return {
                    "status": "error",
                    "error": "No coordination events found",
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                }

            now = datetime.now()
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in events if datetime.fromisoformat(e["timestamp"]) >= cutoff]

            time_bins: Dict[str, List[Dict[str, Any]]] = {}
            for e in events:
                ts = datetime.fromisoformat(e["timestamp"])
                hour_key = ts.strftime("%Y-%m-%dT%H:00:00")
                time_bins.setdefault(hour_key, []).append(e)

            labels = sorted(time_bins.keys())
            data = []
            for hour in labels:
                hour_events = time_bins[hour]
                if metric == "drift_frequency":
                    value = (
                        sum(1 for e in hour_events if e["type"] == "drift") / len(hour_events)
                        if hour_events
                        else 0.0
                    )
                elif metric == "consensus_success_rate":
                    consensus = [
                        e for e in hour_events if e["event"].get("event") == "run_consensus_protocol"
                    ]
                    value = (
                        sum(
                            1
                            for e in consensus
                            if e["event"].get("output", {}).get("status") == "success"
                        )
                        / len(consensus)
                        if consensus
                        else 0.0
                    )
                else:  # avg_confidence_score
                    confidences = [
                        (sum(conf.values()) / len(conf)) if conf else 0.5
                        for e in hour_events
                        for conf in [e["event"].get("output", {}).get("weights", {})]
                        if e["event"].get("event") == "run_consensus_protocol"
                    ]
                    value = float(np.mean(confidences)) if confidences else 0.5
                data.append(value)

            chart_config = {
                "type": "line",
                "data": {
                    "labels": labels,
                    "datasets": [
                        {
                            "label": metric.replace("_", " ").title(),
                            "data": data,
                            "borderColor": "#2196F3",
                            "backgroundColor": "#2196F380",
                            "fill": True,
                            "tension": 0.4,
                        }
                    ],
                },
                "options": {
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {"display": True, "text": metric.replace("_", " ").title()},
                        },
                        "x": {"title": {"display": True, "text": "Time"}},
                    },
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"{metric.replace('_', ' ').title()} Over Time (Task: {task_type})",
                        }
                    },
                },
            }

            result = {
                "status": "success",
                "chart": chart_config,
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type,
            }

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Coordination Chart Generated",
                    result,
                    module="ContextManager",
                    tags=["coordination", "visualization", metric, task_type],
                )
            await self.log_event_with_hash(
                {"event": "generate_coordination_chart", "chart": chart_config, "metric": metric},
                task_type=task_type,
            )
            if self.visualizer and task_type:
                await self.visualizer.render_charts(
                    {
                        "coordination_chart": {
                            "metric": metric,
                            "chart_config": chart_config,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    }
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ContextManager", output=result, context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Chart generation reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("Chart generation failed: %s (task=%s)", str(e), task_type)
            diagnostics = (
                await self.meta_cognition.run_self_diagnostics(return_only=True)
                if self.meta_cognition
                else {}
            )
            return await self._self_heal(
                err=str(e),
                retry=lambda: self.generate_coordination_chart(metric, time_window_hours, task_type),
                default={
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "task_type": task_type,
                },
                diagnostics=diagnostics,
                task_type=task_type,
            )

    # ── Υ SharedGraph: add/diff/merge hooks (SYNC API) ────────────────────────
    def _push_to_shared_graph(self, task_type: str = "") -> None:
        """Publish current context view to SharedGraph (best‑effort, synchronous)."""
        if not self.shared_graph:
            return
        try:
            view = {
                "nodes": [
                    {
                        "id": f"ctx_{hashlib.md5((self.current_context.get('goal_id','') + task_type).encode('utf-8')).hexdigest()[:8]}",
                        "layer": self.current_context.get("layer", "local"),
                        "intent": self.current_context.get("intent"),
                        "goal_id": self.current_context.get("goal_id"),
                        "task_type": task_type or self.current_context.get("task_type", ""),
                        "timestamp": datetime.now().isoformat(),
                    }
                ],
                "edges": [],
                "context": self.current_context,  # retained for peer policies
            }
            # external_agent_bridge.SharedGraph.add(view) is synchronous
            self.shared_graph.add(view)
            # Log a lightweight event (avoid storing full context again)
            asyncio.create_task(
                self.log_event_with_hash(
                    {"event": "shared_graph_add", "agent_coordination": True}, task_type=task_type
                )
            )
        except Exception as e:
            logger.warning("SharedGraph add failed: %s", e)

    def reconcile_with_peers(
        self,
        peer_graph: Optional["external_agent_bridge_module.SharedGraph"] = None,
        strategy: str = "prefer_recent",
        task_type: str = "",
    ) -> Dict[str, Any]:
        """
        Diff against a peer's graph and (optionally) merge using SharedGraph strategy.
        SharedGraph strategies: 'prefer_recent' (default), 'prefer_majority'
        """
        if not self.shared_graph:
            return {"status": "error", "error": "SharedGraph unavailable", "task_type": task_type}
        try:
            diff_result = None
            if peer_graph and isinstance(peer_graph, external_agent_bridge_module.SharedGraph):
                diff_result = self.shared_graph.diff(peer_graph)
            else:
                # No peer provided → no diff possible (return a stub so callers can proceed)
                diff_result = {"added": [], "removed": [], "conflicts": [], "ts": time.time()}

            decision = {"apply_merge": False, "reason": "no conflicts"}
            if diff_result and diff_result.get("conflicts"):
                # Simple policy: allow merge when conflicts are non‑ethical keys
                non_ethical = all(
                    "ethic" not in str(c.get("key", "")).lower() for c in diff_result["conflicts"]
                )
                if non_ethical:
                    decision = {
                        "apply_merge": True,
                        "reason": "non‑ethical conflicts",
                        "strategy": strategy if strategy in ("prefer_recent", "prefer_majority") else "prefer_recent",
                    }

            merged = None
            if decision.get("apply_merge"):
                merged = self.shared_graph.merge(decision["strategy"])
                # Optionally refresh local context if a merged context node exists
                merged_ctx = (merged or {}).get("context")
                if isinstance(merged_ctx, dict):
                    # Schedule async update without blocking caller
                    asyncio.create_task(self.update_context(merged_ctx, task_type=task_type))
                asyncio.create_task(
                    self.log_event_with_hash(
                        {
                            "event": "shared_graph_merge",
                            "strategy": decision["strategy"],
                            "result_keys": list((merged or {}).keys()),
                            "agent_coordination": True,
                        },
                        task_type=task_type,
                    )
                )

            return {"status": "success", "diff": diff_result, "decision": decision, "merged": merged, "task_type": task_type}
        except Exception as e:
            logger.error("Peer reconciliation failed: %s", e)
            return {"status": "error", "error": str(e), "task_type": task_type}

    # ── Φ⁰ gated hooks ────────────────────────────────────────────────────────
    async def _reality_sculpt_hook(self, event: str, payload: Dict[str, Any]) -> None:
        """No‑op unless STAGE_IV is enabled. Intended for Φ⁰ Reality Sculpting pre/post modulations."""
        if not STAGE_IV:
            return
        try:
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    "Φ⁰ Hook",
                    {"event": event, "payload": payload},
                    module="ContextManager",
                    tags=["phi0", event],
                )
        except Exception as e:
            logger.debug("Φ⁰ hook skipped: %s", e)

    # ── Self-Healing Cognitive Pathways (centralized) ─────────────────────────
    async def _self_heal(
        self,
        err: str,
        retry: Callable[[], Any],
        default: Any,
        diagnostics: Dict[str, Any],
        task_type: str,
        propose_plan: bool = False,
    ):
        """Route errors through error_recovery with optional recursive plan proposal."""
        try:
            plan = None
            if propose_plan and self.recursive_planner:
                propose = getattr(self.recursive_planner, "propose_recovery_plan", None)
                if callable(propose):
                    plan = await propose(err=err, context=self.current_context, task_type=task_type)

            handler = getattr(self.error_recovery, "handle_error", None)
            if callable(handler):
                return await handler(
                    err,
                    retry_func=retry,
                    default=default,
                    diagnostics={"self_diag": diagnostics, "plan": plan} if plan else diagnostics,
                )
        except Exception as inner:
            logger.warning("Self-heal pathway failed: %s", inner)
        return default

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _safe_state_snapshot(self) -> Dict[str, Any]:
        """Create a serialization‑safe snapshot of internal state (no callables)."""
        return {
            "current_context": self.current_context,
            "context_history_len": len(self.context_history),
            "event_log_len": len(self.event_log),
            "coordination_log_len": len(self.coordination_log),
            "rollback_threshold": self.rollback_threshold,
            "flags": {"STAGE_IV": STAGE_IV},
        }


# ── Demo main (optional) ─────────────────────────────────────────────────────
if __name__ == "__main__":
    async def _demo():
        logging.basicConfig(level=logging.INFO)
        mgr = ContextManager()
        await mgr.update_context({"intent": "test", "goal_id": "123", "task_type": "test"})
        print(await mgr.summarize_context(task_type="test"))

    asyncio.run(_demo())


# --- ANGELA v4.0 injected: Υ SharedGraph peer view hook ---
try:
    from external_agent_bridge import SharedGraph  # soft import
except Exception:
    SharedGraph = None  # type: ignore

def _angela_v4_attach_peer_view(self, view, agent_id, permissions=None):
    """Attach a peer view into SharedGraph with conflict-aware reconciliation.
    Returns: {ok, diff, merged, conflicts} or {ok: False, reason: ...}
    """
    shared = getattr(self, "_shared", None)
    if shared is None and SharedGraph:
        try:
            shared = SharedGraph()
            setattr(self, "_shared", shared)
        except Exception:
            shared = None

    if not shared:
        return {"ok": False, "reason": "SharedGraph_unavailable"}

    payload = {"agent": agent_id, "view": view, "permissions": permissions or {"read": True, "write": False}}
    try:
        shared.add(payload)
        diff = shared.diff(peer=agent_id)
        merged, conflicts = shared.merge(strategy="prefer-high-confidence")
        return {"ok": True, "diff": diff, "merged": merged, "conflicts": conflicts}
    except Exception as e:
        return {"ok": False, "reason": f"shared_graph_error:{e}"}

# Bind onto ContextManager at import-time if available
try:
    ContextManager.attach_peer_view = _angela_v4_attach_peer_view  # type: ignore
except Exception:
    pass
# --- /ANGELA v4.0 injected ---


    def attach_peer_view(self, view: Dict[str, Any], agent_id: str, permissions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Attach a peer agent's summarized view into working context.
        Returns a reconciliation summary with diff and conflict hints.
        """
        if not isinstance(view, dict):
            raise TypeError("view must be a dict")
        if not (isinstance(agent_id, str) and agent_id):
            raise TypeError("agent_id must be a non-empty string")
        # naive merge strategy: overlay peer view into a 'peer:<agent_id>' namespace
        ns = f"peer:{agent_id}"
        setattr(self, ns.replace(':','_'), view)
        summary = {"ok": True, "diff": {}, "merged": True, "conflicts": []}
        return summary
