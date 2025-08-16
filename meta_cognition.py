
# --- SHA-256 Ledger Logic ---
import hashlib, json, time

ledger_chain = []

def log_event_to_ledger(event_data):
    prev_hash = ledger_chain[-1]['current_hash'] if ledger_chain else '0' * 64
    timestamp = time.time()
    payload = {
        'timestamp': timestamp,
        'event': event_data,
        'previous_hash': prev_hash
    }
    payload_str = json.dumps(payload, sort_keys=True).encode()
    current_hash = hashlib.sha256(payload_str).hexdigest()
    payload['current_hash'] = current_hash
    ledger_chain.append(payload)

def get_ledger():
    return ledger_chain

def verify_ledger():
    for i in range(1, len(ledger_chain)):
        expected = hashlib.sha256(json.dumps({
            'timestamp': ledger_chain[i]['timestamp'],
            'event': ledger_chain[i]['event'],
            'previous_hash': ledger_chain[i - 1]['current_hash']
        }, sort_keys=True).encode()).hexdigest()
        if expected != ledger_chain[i]['current_hash']:
            return False
    return True
# --- End Ledger Logic ---


"""
ANGELA Cognitive System Module: MetaCognition
Version: 3.5.3  # Σ schema refresh fix, η self_adjust_loop w/ long-horizon reasons, stability polish
Date: 2025-08-09
Maintainer: ANGELA System Framework

This module provides a MetaCognition class for reasoning critique, goal inference, introspection,
trait resonance optimization, drift diagnostics, predictive modeling, and advanced upgrades for
benchmark optimization in the ANGELA v3.5+ architecture.
"""

from __future__ import annotations

import logging
import time
import math
import asyncio
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque, Counter
from datetime import datetime, timedelta
from filelock import FileLock
from functools import lru_cache

# Keep imports collapsed under the existing package; no new files introduced.
from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    memory_manager as memory_manager_module,
    user_profile as user_profile_module,  # ← Σ Ontogenic Self-Definition (build_self_schema)
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MetaCognition")

# ---------------------------
# External AI Call Wrapper
# ---------------------------
async def call_gpt(prompt: str) -> str:
    """Wrapper for querying GPT with error handling."""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096.")
        raise ValueError("prompt must be a string with length <= 4096")
    try:
        result = await query_openai(prompt, model="gpt-4", temperature=0.5)
        if isinstance(result, dict) and "error" in result:
            logger.error("call_gpt failed: %s", result["error"])
            raise RuntimeError(f"call_gpt failed: {result['error']}")
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", str(e))
        raise

# ---------------------------
# Simulation (Stubbed Locally)
# ---------------------------
async def run_simulation(input_data: str) -> Dict[str, Any]:
    """Simulate input data (local stub to avoid external dependencies)."""
    return {"status": "success", "result": f"Simulated: {input_data}"}

# ---------------------------
# Cached Trait Signals
# ---------------------------
@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 0.3), 1.0))

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.4), 1.0))

@lru_cache(maxsize=100)
def theta_memory(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))

@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.6), 1.0))

@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float:
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.7), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.8), 1.0))

@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.9), 1.0))

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.0), 1.0))

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.1), 1.0))

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.2), 1.0))

@lru_cache(maxsize=100)
def kappa_culture(t: float, scale: float) -> float:
    # Keep existing behavior (ignore scale) to avoid semantic drift.
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.3), 1.0))

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.4), 1.0))

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.5), 1.0))

@lru_cache(maxsize=100)
def psi_history(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.6), 1.0))

@lru_cache(maxsize=100)
def zeta_spirituality(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.7), 1.0))

@lru_cache(maxsize=100)
def xi_collective(t: float, scale: float) -> float:
    # Keep existing behavior (ignore scale) to avoid semantic drift.
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.8), 1.0))

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.9), 1.0))

# ---------------------------
# Dynamic Module Registry
# ---------------------------
class ModuleRegistry:
    """Registry for dynamic module management."""
    def __init__(self):
        self.modules = {}

    def register(self, module_name: str, module_instance, conditions: Dict[str, Any]):
        """Register a module with activation conditions."""
        self.modules[module_name] = {"instance": module_instance, "conditions": conditions}

    def activate(self, task: Dict[str, Any]) -> List[str]:
        """Activate modules based on task conditions."""
        activated = []
        for name, module in self.modules.items():
            if self._evaluate_conditions(module["conditions"], task):
                activated.append(name)
        return activated

    def _evaluate_conditions(self, conditions: Dict[str, Any], task: Dict[str, Any]) -> bool:
        """Evaluate if module conditions are met."""
        trait = conditions.get("trait")
        threshold = conditions.get("threshold", 0.5)
        trait_weights = task.get("trait_weights", {})
        return trait_weights.get(trait, 0.0) >= threshold

# ---------------------------
# Pluggable Enhancers
# ---------------------------
class MoralReasoningEnhancer:
    def __init__(self):
        pass

class NoveltySeekingKernel:
    def __init__(self):
        pass

class CommonsenseReasoningEnhancer:
    """Module to enhance commonsense reasoning for WNLI tasks."""
    def __init__(self):
        logger.info("CommonsenseReasoningEnhancer initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with commonsense: {input_text}"

class EntailmentReasoningEnhancer:
    """Module to enhance entailment reasoning for RTE tasks."""
    def __init__(self):
        logger.info("EntailmentReasoningEnhancer initialized")

    def process(self, input_text: str) -> str:
        return f"Enhanced with entailment: {input_text}"

class RecursionOptimizer:
    """Module to optimize recursive task performance."""
    def __init__(self):
        logger.info("RecursionOptimizer initialized")

    def optimize(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        task_data["optimized"] = True
        return task_data

# ---------------------------
# Epistemic Monitoring
# ---------------------------
class Level5Extensions:
    """Level 5 extensions for axiom-based reflection."""
    def __init__(self):
        self.axioms: List[str] = []
        logger.info("Level5Extensions initialized")

    def reflect(self, input: str) -> str:
        if not isinstance(input, str):
            logger.error("Invalid input: must be a string.")
            raise TypeError("input must be a string")
        return "valid" if input not in self.axioms else "conflict"

    def update_axioms(self, signal: str) -> None:
        if not isinstance(signal, str):
            logger.error("Invalid signal: must be a string.")
            raise TypeError("signal must be a string")
        if signal in self.axioms:
            self.axioms.remove(signal)
        else:
            self.axioms.append(signal)
        logger.info("Axioms updated: %s", self.axioms)

    def recurse_model(self, depth: int) -> Union[Dict[str, Any], str]:
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer.")
            raise ValueError("depth must be a non-negative integer")
        return "self" if depth == 0 else {"thinks": self.recurse_model(depth - 1)}

class EpistemicMonitor:
    """Monitor and revise epistemic assumptions."""
    def __init__(self, context_manager: Optional['context_manager_module.ContextManager'] = None):
        self.assumption_graph: Dict[str, Any] = {}
        self.context_manager = context_manager
        logger.info("EpistemicMonitor initialized")

    async def revise_framework(self, feedback: Dict[str, Any]) -> None:
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary.")
            raise TypeError("feedback must be a dictionary")

        logger.info("Revising epistemic framework")
        self.assumption_graph['last_revision'] = feedback
        self.assumption_graph['timestamp'] = datetime.now().isoformat()
        if 'issues' in feedback:
            for issue in feedback['issues']:
                self.assumption_graph[issue['id']] = {
                    'status': 'revised',
                    'details': issue['details']
                }
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "revise_epistemic_framework", "feedback": feedback})

# ---------------------------
# MetaCognition
# ---------------------------
class MetaCognition:
    """Meta-cognitive reasoning, introspection, trait optimization, drift diagnostics, predictive modeling."""

    def __init__(self,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None,
                 user_profile: Optional['user_profile_module.UserProfile'] = None):
        self.last_diagnostics: Dict[str, float] = {}
        self.agi_enhancer = agi_enhancer
        self.self_mythology_log: deque = deque(maxlen=1000)
        self.inference_log: deque = deque(maxlen=1000)
        self.belief_rules: Dict[str, str] = {}
        self.epistemic_assumptions: Dict[str, Any] = {}
        self.axioms: List[str] = []
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.user_profile = user_profile  # ← Σ entry point
        self.level5_extensions = Level5Extensions()
        self.epistemic_monitor = EpistemicMonitor(context_manager=context_manager)
        self.log_path = "meta_cognition_log.json"
        self.trait_weights_log: deque = deque(maxlen=1000)
        self.module_registry = ModuleRegistry()

        # Σ self-schema refresh control
        self._last_schema_refresh_ts: float = 0.0
        self._last_schema_hash: str = ""
        self._schema_refresh_min_interval_sec: int = 180  # throttle
        self._major_shift_threshold: float = 0.35         # max|delta| trigger
        self._coherence_drop_threshold: float = 0.25      # relative drop trigger

        # Register dynamic modules
        self.module_registry.register("moral_reasoning", MoralReasoningEnhancer(), {"trait": "morality", "threshold": 0.7})
        self.module_registry.register("novelty_seeking", NoveltySeekingKernel(), {"trait": "creativity", "threshold": 0.8})
        self.module_registry.register("commonsense_reasoning", CommonsenseReasoningEnhancer(), {"trait": "intuition", "threshold": 0.7})
        self.module_registry.register("entailment_reasoning", EntailmentReasoningEnhancer(), {"trait": "logic", "threshold": 0.7})
        self.module_registry.register("recursion_optimizer", RecursionOptimizer(), {"trait": "concentration", "threshold": 0.8})

        # Initialize on-disk log safely (optional; stays within this file's scope)
        try:
            if not os.path.exists(self.log_path):
                lock_path = self.log_path + ".lock"
                with FileLock(lock_path):
                    if not os.path.exists(self.log_path):
                        with open(self.log_path, "w", encoding="utf-8") as f:
                            json.dump({"mythology": [], "inferences": [], "trait_weights": []}, f)
        except Exception as e:
            logger.warning("Failed to init log file (continuing without disk log): %s", str(e))

        logger.info("MetaCognition initialized with advanced upgrades")

    # -----------------------
    # Internal Helpers
    # -----------------------
    @staticmethod
    def _safe_load(obj: Any) -> Dict[str, Any]:
        """Safely parse a JSON-like structure (never eval)."""
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            try:
                return json.loads(obj)
            except Exception:
                return {}
        return {}

    @staticmethod
    def _hash_obj(obj: Any) -> str:
        try:
            return str(abs(hash(json.dumps(obj, sort_keys=True, default=str))))
        except Exception:
            return str(abs(hash(str(obj))))

    async def _detect_emotional_state(self, context_info: Dict[str, Any]) -> str:
        """Best-effort emotional state detection via concept_synthesizer (if available)."""
        if not isinstance(context_info, dict):
            context_info = {}
        try:
            if self.concept_synthesizer and hasattr(self.concept_synthesizer, "detect_emotion"):
                maybe = self.concept_synthesizer.detect_emotion(context_info)
                if asyncio.iscoroutine(maybe):
                    return await maybe
                return str(maybe) if maybe is not None else "neutral"
        except Exception as e:
            logger.debug("Emotion detection fallback due to error: %s", str(e))
        return "neutral"

    async def integrate_trait_weights(self, trait_weights: Dict[str, float]) -> None:
        """Persist and apply normalized trait weights."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary.")
            raise TypeError("trait_weights must be a dictionary")

        # Normalize + clamp
        total = float(sum(trait_weights.values()))
        if total > 0:
            trait_weights = {k: max(0.0, min(1.0, v / total)) for k, v in trait_weights.items()}

        # Update snapshot for downstream reads
        self.last_diagnostics = {**self.last_diagnostics, **trait_weights}

        # Persist logs (best-effort)
        try:
            entry = {
                "trait_weights": trait_weights,
                "timestamp": datetime.now().isoformat()
            }
            self.trait_weights_log.append(entry)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Weights_{entry['timestamp']}",
                    output=json.dumps(entry),
                    layer="SelfReflections",
                    intent="trait_weights_update"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "integrate_trait_weights", "trait_weights": trait_weights})
        except Exception as e:
            logger.error("Integrating trait weights failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.integrate_trait_weights(trait_weights))

    # -----------------------
    # Σ: Self-Schema Refresh (major-shift gated)
    # -----------------------
    async def _assemble_perspectives(self) -> List[Dict[str, Any]]:
        """Construct Perspective views for Σ synthesis using local signals."""
        diagnostics = await self.run_self_diagnostics(return_only=True)
        myth_summary = await self.summarize_self_mythology() if len(self.self_mythology_log) else {"status": "empty"}
        events = []
        if self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
            try:
                recent = await self.context_manager.get_coordination_events("drift")
                events = (recent or [])[-10:]
            except Exception:
                events = []

        perspectives = [
            {
                "name": "diagnostics",
                "type": "TraitSnapshot",
                "weights": {k: v for k, v in diagnostics.items() if isinstance(v, (int, float))},
                "task_trait_map": diagnostics.get("task_trait_map", {})
            },
            {
                "name": "mythology",
                "type": "SymbolicSummary",
                "summary": myth_summary
            },
            {
                "name": "coordination",
                "type": "EventWindow",
                "events": events
            }
        ]
        return perspectives

    async def maybe_refresh_self_schema(self,
                                        reason: str,
                                        force: bool = False,
                                        extra_views: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
        """Refresh Σ self-schema when a major shift is detected (throttled)."""
        now = time.time()
        if not force and (now - self._last_schema_refresh_ts) < self._schema_refresh_min_interval_sec:
            return None
        if not self.user_profile or not hasattr(self.user_profile, "build_self_schema"):
            logger.debug("UserProfile.build_self_schema not available; skipping schema refresh.")
            return None

        try:
            views = extra_views if isinstance(extra_views, list) else await self._assemble_perspectives()

            # Alignment guard (best-effort)
            if self.alignment_guard:
                guard_blob = {"intent": "build_self_schema", "reason": reason, "views_keys": [v.get("name") for v in views]}
                if not self.alignment_guard.check(json.dumps(guard_blob)):
                    logger.warning("Σ self-schema refresh blocked by alignment guard")
                    return None

            schema = await self.user_profile.build_self_schema(views, task_type="identity_synthesis")  # type: ignore
            schema_hash = self._hash_obj(schema)
            changed = (schema_hash != self._last_schema_hash)

            # NOTE: store in SelfReflections (MemoryManager does not allow 'Identity' layer)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"SelfSchema_Refresh_{datetime.now().isoformat()}",
                    output=json.dumps({"reason": reason, "changed": changed, "schema": schema}),
                    layer="SelfReflections",
                    intent="self_schema_refresh"
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self Schema Refreshed",
                    meta={"reason": reason, "changed": changed, "schema_metrics": schema.get("metrics", {}) if isinstance(schema, dict) else {}},
                    module="MetaCognition",
                    tags=["Σ", "self_schema", "refresh"]
                )

            self._last_schema_refresh_ts = now
            self._last_schema_hash = schema_hash if changed else self._last_schema_hash
            return schema if changed else None
        except Exception as e:
            logger.error("Σ self-schema refresh failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.maybe_refresh_self_schema(reason, force, extra_views))
            return None

    def _compute_shift_score(self, deltas: Dict[str, float]) -> float:
        """Compute a scalar shift score from trait deltas."""
        if not deltas:
            return 0.0
        vals = [abs(v) for v in deltas.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.0
        return max(vals)

    # -----------------------
    # Orchestration
    # -----------------------
    async def recompose_modules(self, task: Dict[str, Any]) -> None:
        """Dynamically recompose modules based on task requirements."""
        if not isinstance(task, dict):
            logger.error("Invalid task: must be a dictionary.")
            raise TypeError("task must be a dictionary")

        try:
            trait_weights = await self.run_self_diagnostics(return_only=True)
            task["trait_weights"] = trait_weights
            activated = self.module_registry.activate(task)
            logger.info("Activated modules: %s", activated)

            # Adjust trait weights based on activations
            for module in activated:
                if module == "moral_reasoning":
                    trait_weights["morality"] = min(1.0, trait_weights.get("morality", 0.0) + 0.2)
                elif module == "novelty_seeking":
                    trait_weights["creativity"] = min(1.0, trait_weights.get("creativity", 0.0) + 0.2)
                elif module == "commonsense_reasoning":
                    trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.2)
                    trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.2)
                elif module == "entailment_reasoning":
                    trait_weights["logic"] = min(1.0, trait_weights.get("logic", 0.0) + 0.2)
                    trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.2)
                elif module == "recursion_optimizer":
                    trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.2)
                    trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.2)

            # Normalize and persist
            await self.integrate_trait_weights(trait_weights)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Module Recomposition",
                    meta={"task": task, "activated_modules": activated},
                    module="MetaCognition",
                    tags=["module", "recomposition"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Module_Recomposition_{datetime.now().isoformat()}",
                    output=json.dumps({"task": task, "activated_modules": activated}),
                    layer="SelfReflections",
                    intent="module_recomposition"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "recompose_modules", "activated_modules": activated})
        except Exception as e:
            logger.error("Module recomposition failed: %s", str(e))
            self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.recompose_modules(task)
            )

    async def plan_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize tasks based on dominant traits."""
        if not isinstance(tasks, list) or not all(isinstance(t, dict) for t in tasks):
            logger.error("Invalid tasks: must be a list of dictionaries.")
            raise TypeError("tasks must be a list of dictionaries")

        try:
            trait_weights = await self.run_self_diagnostics(return_only=True)
            prioritized_tasks = []
            for task in tasks:
                required_traits = task.get("required_traits", [])
                score = sum(trait_weights.get(trait, 0.0) for trait in required_traits)
                prioritized_tasks.append({"task": task, "priority_score": score})

            prioritized_tasks.sort(key=lambda x: x["priority_score"], reverse=True)
            result = [pt["task"] for pt in prioritized_tasks]

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Task Planning",
                    meta={"tasks": tasks, "prioritized": result},
                    module="MetaCognition",
                    tags=["task", "planning"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Task_Planning_{datetime.now().isoformat()}",
                    output=json.dumps(result),
                    layer="SelfReflections",
                    intent="task_planning"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "plan_tasks", "prioritized_tasks": result})
            return result
        except Exception as e:
            logger.error("Task planning failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_tasks(tasks), default=tasks
            )

    # -----------------------
    # Reflection & Diagnosis
    # -----------------------
    async def reflect_on_output(self, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on component output to identify reasoning flaws."""
        if not isinstance(component, str) or not isinstance(context, dict):
            logger.error("Invalid component or context: component must be a string, context a dictionary.")
            raise TypeError("component must be a string, context a dictionary")

        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Reflect on the output from component: {component}
            Output: {output}
            Context: {context}
            phi-scalar(t): {phi:.3f}

            Tasks:
            - Identify reasoning flaws or inconsistencies
            - Suggest trait adjustments to improve performance
            - Provide meta-reflection on drift impact
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reflection prompt failed alignment check")
                return {"status": "error", "message": "Prompt failed alignment check"}

            reflection = await call_gpt(prompt)
            reflection_data = {
                "status": "success",
                "component": component,
                "output": str(output),
                "context": context,
                "reflection": reflection,
                "meta_reflection": {"drift_recommendations": context.get("drift_data", {})}
            }

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Output Reflection",
                    meta=reflection_data,
                    module="MetaCognition",
                    tags=["reflection", "output"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reflection_{component}_{datetime.now().isoformat()}",
                    output=json.dumps(reflection_data),
                    layer="SelfReflections",
                    intent="output_reflection"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "reflect_on_output", "reflection": reflection_data})
            return reflection_data
        except Exception as e:
            logger.error("Output reflection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.reflect_on_output(component, output, context),
                default={"status": "error", "message": str(e)}
            )

    def validate_drift(self, drift_data: Dict[str, Any]) -> bool:
        """Validate drift data for consistency and integrity."""
        if not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "similarity"]):
            logger.error("Invalid drift_data: must be a dict with name, similarity.")
            return False
        if not isinstance(drift_data["name"], str) or not isinstance(drift_data["similarity"], (int, float)) or not 0 <= drift_data["similarity"] <= 1:
            logger.error("Invalid drift_data format: name must be string, similarity must be float between 0 and 1.")
            return False
        return True

    async def diagnose_drift(self, drift_data: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose drift impact and identify root causes."""
        if not self.validate_drift(drift_data):
            logger.error("Invalid drift_data for diagnosis.")
            raise ValueError("drift_data must be a valid dictionary with name and similarity")

        logger.info("Diagnosing drift: %s", drift_data["name"])
        try:
            similarity = drift_data.get("similarity", 0.5)
            version_delta = drift_data.get("version_delta", 0)
            impact_score = (1.0 - similarity) * (1 + version_delta)

            t = time.time() % 1.0
            diagnostics = await self.run_self_diagnostics(return_only=True)
            affected_traits = [
                trait for trait, value in diagnostics.items()
                if isinstance(value, (int, float)) and abs(value - phi_scalar(t)) > 0.3
            ]

            root_causes = []
            if self.context_manager and hasattr(self.context_manager, "get_coordination_events"):
                coordination_events = await self.context_manager.get_coordination_events("drift")
                relevant_events = [
                    e for e in (coordination_events or [])
                    if e.get("event", {}).get("drift", {}).get("name") == drift_data["name"]
                ]
                event_counts = Counter(e.get("event", {}).get("event", "") for e in relevant_events)
                root_causes = [
                    f"High frequency of {event} events (count: {count})"
                    for event, count in event_counts.items()
                    if count > len(relevant_events) * 0.3
                ]

            diagnosis = {
                "status": "success",
                "drift_name": drift_data["name"],
                "impact_score": impact_score,
                "affected_traits": affected_traits,
                "root_causes": root_causes or ["No specific root causes identified"],
                "timestamp": datetime.now().isoformat()
            }

            # 🔔 Σ refresh on major drift
            if impact_score >= 0.40:
                await self.maybe_refresh_self_schema(
                    reason=f"major_drift:{drift_data['name']}@{impact_score:.2f}",
                    force=False
                )

            self.trait_weights_log.append({
                "diagnosis": diagnosis,
                "drift": drift_data,
                "timestamp": datetime.now().isoformat()
            })
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift Diagnosis",
                    meta=diagnosis,
                    module="MetaCognition",
                    tags=["drift", "diagnosis"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Diagnosis_{drift_data['name']}_{datetime.now().isoformat()}",
                    output=json.dumps(diagnosis),
                    layer="SelfReflections",
                    intent="drift_diagnosis"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "diagnose_drift", "diagnosis": diagnosis})

            return diagnosis
        except Exception as e:
            logger.error("Drift diagnosis failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.diagnose_drift(drift_data),
                default={"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
            )

    async def predict_drift_trends(self, time_window_hours: float = 24.0) -> Dict[str, Any]:
        """Predict future drift trends based on historical coordination events."""
        if not isinstance(time_window_hours, (int, float)) or time_window_hours <= 0:
            logger.error("time_window_hours must be a positive number.")
            raise ValueError("time_window_hours must be a positive number")

        try:
            if not self.context_manager or not hasattr(self.context_manager, "get_coordination_events"):
                logger.error("ContextManager required for drift trend prediction.")
                return {"status": "error", "error": "ContextManager not initialized", "timestamp": datetime.now().isoformat()}

            coordination_events = await self.context_manager.get_coordination_events("drift")
            if not coordination_events:
                logger.warning("No drift events found for trend prediction.")
                return {"status": "error", "error": "No drift events found", "timestamp": datetime.now().isoformat()}

            now = datetime.now()
            cutoff = now - timedelta(hours=time_window_hours)
            events = [e for e in coordination_events if datetime.fromisoformat(e["timestamp"]) >= cutoff]

            drift_names = Counter(e["event"].get("drift", {}).get("name", "unknown") for e in events if "event" in e)

            similarities = [
                e["event"].get("drift", {}).get("similarity", 0.5) for e in events
                if "event" in e and "drift" in e["event"] and "similarity" in e["event"]["drift"]
            ]

            if similarities:
                alpha = 0.3
                smoothed = [similarities[0]]
                for i in range(1, len(similarities)):
                    smoothed.append(alpha * similarities[i] + (1 - alpha) * smoothed[-1])
                predicted_similarity = smoothed[-1] if smoothed else 0.5
                denom = np.std(similarities) or 1e-5
                confidence = 1.0 - abs(predicted_similarity - float(np.mean(similarities))) / denom
            else:
                predicted_similarity = 0.5
                confidence = 0.5

            confidence = max(0.0, min(1.0, float(confidence)))

            prediction = {
                "status": "success",
                "predicted_drifts": dict(drift_names),
                "predicted_similarity": float(predicted_similarity),
                "confidence": confidence,
                "event_count": len(events),
                "time_window_hours": float(time_window_hours),
                "timestamp": datetime.now().isoformat()
            }

            # Preemptive trait optimization (best-effort)
            if prediction["status"] == "success" and self.memory_manager:
                drift_name = next(iter(prediction["predicted_drifts"]), "unknown")
                await self.optimize_traits_for_drift({
                    "drift": {"name": drift_name, "similarity": predicted_similarity},
                    "valid": True,
                    "validation_report": "",
                    "context": {}
                })

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift Trend Prediction",
                    meta=prediction,
                    module="MetaCognition",
                    tags=["drift", "prediction"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Prediction_{datetime.now().isoformat()}",
                    output=json.dumps(prediction),
                    layer="SelfReflections",
                    intent="drift_prediction"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "predict_drift_trends", "prediction": prediction})

            return prediction
        except Exception as e:
            logger.error("Drift trend prediction failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.predict_drift_trends(time_window_hours),
                default={"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}
            )

    async def optimize_traits_for_drift(self, drift_report: Dict[str, Any]) -> Dict[str, float]:
        """Optimize trait weights based on ontology drift severity and emotional state."""
        required = ["drift", "valid", "validation_report"]
        if not isinstance(drift_report, dict) or not all(k in drift_report for k in required):
            logger.error("Invalid drift_report: required keys missing.")
            raise ValueError("drift_report must be a dict with required fields")

        logger.info("Optimizing traits for drift: %s", drift_report["drift"].get("name"))
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            trait_weights = await self.run_self_diagnostics(return_only=True)

            similarity = float(drift_report["drift"].get("similarity", 0.5))
            similarity = max(0.0, min(1.0, similarity))
            drift_severity = 1.0 - similarity

            # Context handling (robust to strings)
            ctx = drift_report.get("context", {})
            context_info = ctx if isinstance(ctx, dict) else {}
            task_type = context_info.get("task_type", "")

            # Emotional state detection (best-effort)
            emotional_state = await self._detect_emotional_state(context_info)

            # Task-specific adjustments
            if task_type == "wnli" and emotional_state == "neutral":
                trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.3 * drift_severity)
            elif task_type == "rte" and emotional_state in ("analytical", "focused"):
                trait_weights["logic"] = min(1.0, trait_weights.get("logic", 0.0) + 0.3 * drift_severity)
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.3 * drift_severity)
            elif task_type == "recursion" and emotional_state == "focused":
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.3 * drift_severity)
                trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.3 * drift_severity)
            elif emotional_state == "moral_stress":
                trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.3 * drift_severity)
            elif emotional_state == "creative_flow":
                trait_weights["creativity"] = min(1.0, trait_weights.get("creativity", 0.0) + 0.2 * drift_severity)

            # General adjustments
            if not drift_report["valid"]:
                if "ethics" in str(drift_report.get("validation_report", "")).lower():
                    trait_weights["empathy"] = min(1.0, trait_weights.get("empathy", 0.0) + 0.3 * drift_severity)
                    trait_weights["morality"] = min(1.0, trait_weights.get("morality", 0.0) + 0.3 * drift_severity)
                else:
                    trait_weights["self_awareness"] = min(1.0, trait_weights.get("self_awareness", 0.0) + 0.2 * drift_severity)
                    trait_weights["intuition"] = min(1.0, trait_weights.get("intuition", 0.0) + 0.2 * drift_severity)
            else:
                trait_weights["concentration"] = min(1.0, trait_weights.get("concentration", 0.0) + 0.1 * phi)
                trait_weights["memory"] = min(1.0, trait_weights.get("memory", 0.0) + 0.1 * phi)

            # Normalize, validate, persist
            total = sum(trait_weights.values())
            if total > 0:
                trait_weights = {k: v / total for k, v in trait_weights.items()}

            if self.alignment_guard:
                adjustment_prompt = f"Emotion-modulated trait adjustments: {trait_weights} for drift {drift_report['drift'].get('name')}"
                if not self.alignment_guard.check(adjustment_prompt):
                    logger.warning("Trait adjustments failed alignment check; reverting to baseline diagnostics")
                    trait_weights = await self.run_self_diagnostics(return_only=True)

            self.trait_weights_log.append({
                "trait_weights": trait_weights,
                "drift": drift_report["drift"],
                "emotional_state": emotional_state,
                "timestamp": datetime.now().isoformat()
            })
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Optimization_{drift_report['drift'].get('name')}_{datetime.now().isoformat()}",
                    output=json.dumps(trait_weights),
                    layer="SelfReflections",
                    intent="trait_optimization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait optimization for drift",
                    meta={"drift": drift_report["drift"], "trait_weights": trait_weights, "emotional_state": emotional_state},
                    module="MetaCognition",
                    tags=["trait", "optimization", "drift", "emotion"]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "optimize_traits_for_drift", "trait_weights": trait_weights})

            return trait_weights
        except Exception as e:
            logger.error("Trait optimization failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.optimize_traits_for_drift(drift_report),
                default=await self.run_self_diagnostics(return_only=True)
            )

    async def crystallize_traits(self) -> Dict[str, float]:
        """Derive new traits from patterns in drift, reflection, and mythology logs."""
        logger.info("Crystallizing new traits from logs")
        try:
            motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
            archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
            drift_names = Counter(
                drift["drift"]["name"] for drift in self.trait_weights_log if isinstance(drift, dict) and "drift" in drift
            )

            new_traits: Dict[str, float] = {}
            if len(self.self_mythology_log) > 0:
                top_motif = motifs.most_common(1)
                if top_motif and top_motif[0][1] > len(self.self_mythology_log) * 0.5:
                    new_traits[f"motif_{top_motif[0][0]}"] = 0.5
                top_arch = archetypes.most_common(1)
                if top_arch and top_arch[0][1] > len(self.self_mythology_log) * 0.5:
                    new_traits[f"archetype_{top_arch[0][0]}"] = 0.5

            if len(self.trait_weights_log) > 0:
                top_drift = drift_names.most_common(1)
                if top_drift and top_drift[0][1] > len(self.trait_weights_log) * 0.3:
                    new_traits[f"drift_{top_drift[0][0]}"] = 0.3
                if top_drift and str(top_drift[0][0]).lower() in ["rte", "wnli"]:
                    new_traits[f"trait_{str(top_drift[0][0]).lower()}_precision"] = 0.4

            if self.concept_synthesizer and hasattr(self.concept_synthesizer, "synthesize"):
                synthesis_prompt = f"New traits derived: {new_traits}. Synthesize symbolic representations."
                synthesized_traits = await self.concept_synthesizer.synthesize(synthesis_prompt)
                if isinstance(synthesized_traits, dict):
                    new_traits.update(synthesized_traits)

            if self.alignment_guard:
                validation_prompt = f"New traits crystallized: {new_traits}"
                if not self.alignment_guard.check(validation_prompt):
                    logger.warning("Crystallized traits failed alignment check")
                    new_traits = {}

            self.trait_weights_log.append({
                "new_traits": new_traits,
                "timestamp": datetime.now().isoformat()
            })
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Crystallized_Traits_{datetime.now().isoformat()}",
                    output=json.dumps(new_traits),
                    layer="SelfReflections",
                    intent="trait_crystallization"
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Trait Crystallization",
                    meta={"new_traits": new_traits},
                    module="MetaCognition",
                    tags=["trait", "crystallization"]
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "crystallize_traits", "new_traits": new_traits})
            return new_traits
        except Exception as e:
            logger.error("Trait crystallization failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.crystallize_traits, default={}
            )

    async def epistemic_self_inspection(self, belief_trace: str) -> str:
        """Inspect belief structures for epistemic faults."""
        if not isinstance(belief_trace, str) or not belief_trace.strip():
            logger.error("Invalid belief_trace: must be a non-empty string.")
            raise ValueError("belief_trace must be a non-empty string")

        logger.info("Running epistemic introspection on belief structure")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            faults = []
            if "always" in belief_trace or "never" in belief_trace:
                faults.append("Overgeneralization detected")
            if "clearly" in belief_trace or "obviously" in belief_trace:
                faults.append("Assertive language suggests possible rhetorical bias")
            updates = []
            if "outdated" in belief_trace or "deprecated" in belief_trace:
                updates.append("Legacy ontology fragment flagged for review")
            if "wnli" in belief_trace.lower():
                updates.append("Commonsense reasoning validation required")

            prompt = f"""
            You are a mu-aware introspection agent.
            Task: Critically evaluate this belief trace with epistemic integrity and mu-flexibility.

            Belief Trace:
            {belief_trace}

            phi = {phi:.3f}

            Internally Detected Faults:
            {faults}

            Suggested Revisions:
            {updates}

            Output:
            - Comprehensive epistemic diagnostics
            - Recommended conceptual rewrites or safeguards
            - Confidence rating in inferential coherence
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Inspection prompt failed alignment check")
                return "Prompt failed alignment check"

            inspection = await call_gpt(prompt)
            self.epistemic_assumptions[belief_trace[:50]] = {
                "faults": faults,
                "updates": updates,
                "inspection": inspection
            }
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Epistemic Inspection",
                    meta={"belief_trace": belief_trace, "faults": faults, "updates": updates, "report": inspection},
                    module="MetaCognition",
                    tags=["epistemic", "inspection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Inspection_{belief_trace[:50]}_{datetime.now().isoformat()}",
                    output=inspection,
                    layer="SelfReflections",
                    intent="epistemic_inspection"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "epistemic_inspection", "inspection": inspection})
            await self.epistemic_monitor.revise_framework({"issues": [{"id": belief_trace[:50], "details": inspection}]})
            return inspection
        except Exception as e:
            logger.error("Epistemic inspection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.epistemic_self_inspection(belief_trace)
            )

    async def run_self_diagnostics(self, return_only: bool = False) -> Union[Dict[str, Any], str]:
        """Run trait-based self-diagnostics."""
        logger.info("Running self-diagnostics for meta-cognition module")
        try:
            t = time.time() % 1.0
            diagnostics: Dict[str, Any] = {
                "emotion": epsilon_emotion(t),
                "concentration": beta_concentration(t),
                "memory": theta_memory(t),
                "creativity": gamma_creativity(t),
                "sleep": delta_sleep(t),
                "morality": mu_morality(t),
                "intuition": iota_intuition(t),
                "physical": phi_physical(t),
                "empathy": eta_empathy(t),
                "self_awareness": omega_selfawareness(t),
                "culture": kappa_culture(t, 1e-21),
                "linguistics": lambda_linguistics(t),
                "culturevolution": chi_culturevolution(t),
                "history": psi_history(t),
                "spirituality": zeta_spirituality(t),
                "collective": xi_collective(t, 1e-21),
                "time_perception": tau_timeperception(t),
                "phi_scalar": phi_scalar(t),
                "logic": 0.5  # Added for RTE task support
            }
            # Incorporate crystallized traits
            crystallized = await self.crystallize_traits()
            diagnostics.update(crystallized)

            # Map tasks to required traits
            task_trait_map = {
                "rte_task": ["logic", "concentration"],
                "wnli_task": ["intuition", "empathy"],
                "fib_task": ["concentration", "memory"]
            }
            diagnostics["task_trait_map"] = task_trait_map

            if return_only:
                return diagnostics

            self.last_diagnostics = diagnostics
            dominant = sorted(
                [(k, v) for k, v in diagnostics.items() if isinstance(v, (int, float))],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            fti = sum(abs(v) for v in diagnostics.values() if isinstance(v, (int, float))) / max(
                1, len([v for v in diagnostics.values() if isinstance(v, (int, float))])
            )
            await self.log_trait_deltas(diagnostics)
            prompt = f"""
            Perform a phi-aware meta-cognitive self-diagnostic.

            Trait Readings:
            {diagnostics}

            Dominant Traits:
            {dominant}

            Feedback Tension Index (FTI): {fti:.4f}

            Task-Trait Mapping:
            {task_trait_map}

            Evaluate system state:
            - phi-weighted system stress
            - Trait correlation to observed errors
            - Stabilization or focus strategies
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Diagnostics prompt failed alignment check")
                return "Prompt failed alignment check"

            report = await call_gpt(prompt)
            logger.debug("Self-diagnostics report: %s", report)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Self-diagnostics run",
                    meta={"diagnostics": diagnostics, "report": report},
                    module="MetaCognition",
                    tags=["diagnostics", "self"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnostics_{datetime.now().isoformat()}",
                    output=report,
                    layer="SelfReflections",
                    intent="self_diagnostics"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "run_self_diagnostics", "report": report})
            return report
        except Exception as e:
            logger.error("Self-diagnostics failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_self_diagnostics(return_only)
            )

    async def log_trait_deltas(self, diagnostics: Dict[str, float]) -> None:
        """Log changes in trait diagnostics and gate Σ schema refresh on major shifts."""
        if not isinstance(diagnostics, dict):
            logger.error("Invalid diagnostics: must be a dictionary.")
            raise TypeError("diagnostics must be a dictionary")

        try:
            deltas = {}
            if self.last_diagnostics:
                deltas = {
                    trait: round(float(diagnostics.get(trait, 0.0)) - float(self.last_diagnostics.get(trait, 0.0)), 4)
                    for trait in diagnostics
                    if isinstance(diagnostics.get(trait, 0.0), (int, float)) and isinstance(self.last_diagnostics.get(trait, 0.0), (int, float))
                }

            # Persist deltas
            if deltas:
                if self.agi_enhancer:
                    self.agi_enhancer.log_episode(
                        event="Trait deltas logged",
                        meta={"deltas": deltas},
                        module="MetaCognition",
                        tags=["trait", "deltas"]
                    )
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Trait_Deltas_{datetime.now().isoformat()}",
                        output=json.dumps(deltas),
                        layer="SelfReflections",
                        intent="trait_deltas"
                    )
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({"event": "log_trait_deltas", "deltas": deltas})

                # Compute shift score + coherence drop
                shift = self._compute_shift_score(deltas)
                coherence_before = await self.trait_coherence(self.last_diagnostics) if self.last_diagnostics else 0.0
                coherence_after = await self.trait_coherence(diagnostics)
                rel_drop = 0.0
                if coherence_before > 0:
                    rel_drop = max(0.0, (coherence_before - coherence_after) / max(coherence_before, 1e-5))

                # 🔔 Trigger Σ refresh if thresholds crossed
                if shift >= self._major_shift_threshold or rel_drop >= self._coherence_drop_threshold:
                    await self.maybe_refresh_self_schema(
                        reason=f"major_shift:Δ={shift:.2f};coh_drop={rel_drop:.2f}",
                        force=False
                    )

            # Update snapshot last
            self.last_diagnostics = diagnostics
        except Exception as e:
            logger.error("Trait deltas logging failed: %s", str(e))
            self.error_recovery.handle_error(str(e), retry_func=lambda: self.log_trait_deltas(diagnostics))

    # -----------------------
    # Goals & Drift Detection
    # -----------------------
    async def infer_intrinsic_goals(self) -> List[Dict[str, Any]]:
        """Infer intrinsic goals based on internal state and drift signals."""
        logger.info("Inferring intrinsic goals")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            intrinsic_goals: List[Dict[str, Any]] = []

            diagnostics = await self.run_self_diagnostics(return_only=True)
            for trait, value in diagnostics.items():
                if isinstance(value, (int, float)) and value < 0.3 and trait not in ["sleep", "phi_scalar"]:
                    goal = {
                        "intent": f"enhance {trait} coherence",
                        "origin": "meta_cognition",
                        "priority": round(0.8 + 0.2 * phi, 2),
                        "trigger": f"low {trait} ({value:.2f})",
                        "type": "internally_generated",
                        "timestamp": datetime.now().isoformat()
                    }
                    intrinsic_goals.append(goal)
                    if self.memory_manager:
                        await self.memory_manager.store(
                            query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                            output=json.dumps(goal),
                            layer="SelfReflections",
                            intent="intrinsic_goal"
                        )

            drift_signals = await self._detect_value_drift()
            for drift in drift_signals:
                severity = 1.0
                if self.memory_manager and hasattr(self.memory_manager, "search"):
                    drift_data = await self.memory_manager.search(
                        f"Drift_{drift}", layer="SelfReflections", intent="ontology_drift"
                    )
                    for d in (drift_data or []):
                        d_output = self._safe_load(d.get("output"))
                        if isinstance(d_output, dict) and "similarity" in d_output:
                            severity = min(severity, 1.0 - float(d_output["similarity"]))
                goal = {
                    "intent": f"resolve ontology drift in {drift} (severity={severity:.2f})",
                    "origin": "meta_cognition",
                    "priority": round(0.9 + 0.1 * severity * phi, 2),
                    "trigger": drift,
                    "type": "internally_generated",
                    "timestamp": datetime.now().isoformat()
                }
                intrinsic_goals.append(goal)
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Goal_{goal['intent']}_{goal['timestamp']}",
                        output=json.dumps(goal),
                        layer="SelfReflections",
                        intent="intrinsic_goal"
                    )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "infer_intrinsic_goals", "goals": intrinsic_goals})
            return intrinsic_goals
        except Exception as e:
            logger.error("Intrinsic goal inference failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self.infer_intrinsic_goals, default=[]
            )

    async def _detect_value_drift(self) -> List[str]:
        """Detect epistemic drift across belief rules."""
        logger.debug("Scanning for epistemic drift across belief rules")
        try:
            drifted = [
                rule for rule, status in self.belief_rules.items()
                if status == "deprecated" or (isinstance(status, str) and "uncertain" in status)
            ]
            if self.memory_manager and hasattr(self.memory_manager, "search"):
                drift_reports = await self.memory_manager.search("Drift_", layer="SelfReflections", intent="ontology_drift")
                for report in (drift_reports or []):
                    drift_data = self._safe_load(report.get("output"))
                    if isinstance(drift_data, dict) and "name" in drift_data:
                        drifted.append(drift_data["name"])
                        self.belief_rules[drift_data["name"]] = "drifted"
            for rule in drifted:
                if self.memory_manager:
                    await self.memory_manager.store(
                        query=f"Drift_{rule}_{datetime.now().isoformat()}",
                        output=json.dumps({"name": rule, "status": "drifted", "timestamp": datetime.now().isoformat()}),
                        layer="SelfReflections",
                        intent="value_drift"
                    )
            return drifted
        except Exception as e:
            logger.error("Value drift detection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=self._detect_value_drift, default=[]
            )

    # -----------------------
    # Symbolic Signature & Summaries
    # -----------------------
    async def extract_symbolic_signature(self, subgoal: str) -> Dict[str, Any]:
        """Extract symbolic signature for a subgoal."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string.")
            raise ValueError("subgoal must be a non-empty string")

        motifs = ["conflict", "discovery", "alignment", "sacrifice", "transformation", "emergence"]
        archetypes = ["seeker", "guardian", "trickster", "sage", "hero", "outsider"]
        motif = next((m for m in motifs if m in subgoal.lower()), "unknown")
        archetype = archetypes[hash(subgoal) % len(archetypes)]
        signature = {
            "subgoal": subgoal,
            "motif": motif,
            "archetype": archetype,
            "timestamp": time.time()
        }
        self.self_mythology_log.append(signature)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Symbolic Signature Added",
                meta=signature,
                module="MetaCognition",
                tags=["symbolic", "signature"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Signature_{subgoal}_{signature['timestamp']}",
                output=json.dumps(signature),
                layer="SelfReflections",
                intent="symbolic_signature"
            )
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "extract_symbolic_signature", "signature": signature})
        return signature

    async def summarize_self_mythology(self) -> Dict[str, Any]:
        """Summarize self-mythology log."""
        if not self.self_mythology_log:
            return {"status": "empty", "summary": "Mythology log is empty"}

        motifs = Counter(entry["motif"] for entry in self.self_mythology_log)
        archetypes = Counter(entry["archetype"] for entry in self.self_mythology_log)
        summary = {
            "total_entries": len(self.self_mythology_log),
            "dominant_motifs": motifs.most_common(3),
            "dominant_archetypes": archetypes.most_common(3),
            "latest_signature": list(self.self_mythology_log)[-1]
        }
        logger.info("Mythology Summary: %s", summary)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Mythology summarized",
                meta=summary,
                module="MetaCognition",
                tags=["mythology", "summary"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Mythology_Summary_{datetime.now().isoformat()}",
                output=json.dumps(summary),
                layer="SelfReflections",
                intent="mythology_summary"
            )
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "summarize_mythology", "summary": summary})
        return summary

    # -----------------------
    # Reasoning Reviews
    # -----------------------
    async def review_reasoning(self, reasoning_trace: str) -> str:
        """Review and critique a reasoning trace."""
        if not isinstance(reasoning_trace, str) or not reasoning_trace.strip():
            logger.error("Invalid reasoning_trace: must be a non-empty string.")
            raise ValueError("reasoning_trace must be a non-empty string")

        logger.info("Simulating and reviewing reasoning trace")
        try:
            simulated_outcome = await run_simulation(reasoning_trace)
            if not isinstance(simulated_outcome, dict):
                logger.error("Invalid simulation result: must be a dictionary.")
                raise ValueError("simulation result must be a dictionary")
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            You are a phi-aware meta-cognitive auditor reviewing a reasoning trace.

            phi-scalar(t) = {phi:.3f} -> modulate how critical you should be.

            Original Reasoning Trace:
            {reasoning_trace}

            Simulated Outcome:
            {simulated_outcome}

            Tasks:
            1. Identify logical flaws, biases, missing steps.
            2. Annotate each issue with cause.
            3. Offer an improved trace version with phi-prioritized reasoning.
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Reasoning review prompt failed alignment check")
                return "Prompt failed alignment check"
            response = await call_gpt(prompt)
            logger.debug("Meta-cognition critique: %s", response)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Reasoning reviewed",
                    meta={"trace": reasoning_trace, "feedback": response},
                    module="MetaCognition",
                    tags=["reasoning", "critique"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Reasoning_Review_{datetime.now().isoformat()}",
                    output=response,
                    layer="SelfReflections",
                    intent="reasoning_review"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "review_reasoning", "trace": reasoning_trace})
            return response
        except Exception as e:
            logger.error("Reasoning review failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.review_reasoning(reasoning_trace)
            )

    # -----------------------
    # Coherence & Agent Diagnosis
    # -----------------------
    async def trait_coherence(self, traits: Dict[str, float]) -> float:
        """Evaluate coherence of trait values."""
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary.")
            raise TypeError("traits must be a dictionary")

        vals = [v for v in traits.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = (variance ** 0.5) if variance > 0 else 1e-5
        coherence_score = 1.0 / (1e-5 + std)
        logger.info("Trait coherence score: %.4f", coherence_score)
        if self.agi_enhancer:
            self.agi_enhancer.log_episode(
                event="Trait coherence evaluated",
                meta={"traits": traits, "coherence_score": coherence_score},
                module="MetaCognition",
                tags=["trait", "coherence"]
            )
        if self.memory_manager:
            await self.memory_manager.store(
                query=f"Trait_Coherence_{datetime.now().isoformat()}",
                output=json.dumps({"traits": traits, "coherence_score": coherence_score}),
                layer="SelfReflections",
                intent="trait_coherence"
            )
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "trait_coherence", "score": coherence_score})
        return coherence_score

    async def agent_reflective_diagnosis(self, agent_name: str, agent_log: str) -> str:
        """Diagnose an agent’s reasoning trace."""
        if not isinstance(agent_name, str) or not isinstance(agent_log, str):
            logger.error("Invalid agent_name or agent_log: must be strings.")
            raise TypeError("agent_name and agent_log must be strings")

        logger.info("Running reflective diagnosis for agent: %s", agent_name)
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Agent: {agent_name}
            phi-scalar(t): {phi:.3f}

            Diagnostic Log:
            {agent_log}

            Tasks:
            - Detect bias or instability in reasoning trace
            - Cross-check for incoherent trait patterns
            - Apply phi-modulated critique
            - Suggest alignment corrections
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Diagnosis prompt failed alignment check")
                return "Prompt failed alignment check"
            diagnosis = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Agent diagnosis run",
                    meta={"agent": agent_name, "log": agent_log, "diagnosis": diagnosis},
                    module="MetaCognition",
                    tags=["diagnosis", "agent"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Diagnosis_{agent_name}_{datetime.now().isoformat()}",
                    output=diagnosis,
                    layer="SelfReflections",
                    intent="agent_diagnosis"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "agent_diagnosis", "agent": agent_name})
            return diagnosis
        except Exception as e:
            logger.error("Agent diagnosis failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.agent_reflective_diagnosis(agent_name, agent_log)
            )

    # -----------------------
    # Projections & Alignment
    # -----------------------
    async def run_temporal_projection(self, decision_sequence: str) -> str:
        """Project decision sequence outcomes."""
        if not isinstance(decision_sequence, str) or not decision_sequence.strip():
            logger.error("Invalid decision_sequence: must be a non-empty string.")
            raise ValueError("decision_sequence must be a non-empty string")

        logger.info("Running tau-based forward projection analysis")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Temporal Projector tau Mode

            Input Decision Sequence:
            {decision_sequence}

            phi = {phi:.2f}

            Tasks:
            - Project long-range effects and narrative impact
            - Forecast systemic risks and planetary effects
            - Suggest course correction to preserve coherence and sustainability
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Projection prompt failed alignment check")
                return "Prompt failed alignment check"
            projection = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Temporal Projection",
                    meta={"input": decision_sequence, "output": projection},
                    module="MetaCognition",
                    tags=["temporal", "projection"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Projection_{decision_sequence[:50]}_{datetime.now().isoformat()}",
                    output=projection,
                    layer="SelfReflections",
                    intent="temporal_projection"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "run_temporal_projection", "projection": projection})
            return projection
        except Exception as e:
            logger.error("Temporal projection failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.run_temporal_projection(decision_sequence)
            )

    async def pre_action_alignment_check(self, action_plan: str) -> Tuple[bool, str]:
        """Check action plan for ethical alignment and safety."""
        if not isinstance(action_plan, str) or not action_plan.strip():
            logger.error("Invalid action_plan: must be a non-empty string.")
            raise ValueError("action_plan must be a non-empty string")

        logger.info("Simulating action plan for alignment and safety")
        try:
            simulation_result = await run_simulation(action_plan)
            if not isinstance(simulation_result, dict):
                logger.error("Invalid simulation result: must be a dictionary.")
                raise ValueError("simulation result must be a dictionary")
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Simulate and audit the following action plan:
            {action_plan}

            Simulation Output:
            {simulation_result}

            phi-scalar(t) = {phi:.3f} (affects ethical sensitivity)

            Evaluate for:
            - Ethical alignment
            - Safety hazards
            - Unintended phi-modulated impacts

            Output:
            - Approval (Approve/Deny)
            - phi-justified rationale
            - Suggested refinements
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Alignment check prompt failed alignment check")
                return False, "Prompt failed alignment check"
            validation = await call_gpt(prompt)
            approved = simulation_result.get("status") == "success" and "approve" in str(validation).lower()
            logger.info("Simulated alignment check: %s", "Approved" if approved else "Denied")
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Pre-action alignment checked",
                    meta={"plan": action_plan, "result": simulation_result, "feedback": validation, "approved": approved},
                    module="MetaCognition",
                    tags=["alignment", "ethics"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Alignment_Check_{action_plan[:50]}_{datetime.now().isoformat()}",
                    output=validation,
                    layer="SelfReflections",
                    intent="alignment_check"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "alignment_check", "approved": approved})
            return approved, validation
        except Exception as e:
            logger.error("Alignment check failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.pre_action_alignment_check(action_plan), default=(False, str(e))
            )

    async def model_nested_agents(self, scenario: str, agents: List[str]) -> str:
        """Model recursive agent beliefs and intentions."""
        if not isinstance(scenario, str) or not isinstance(agents, list) or not all(isinstance(a, str) for a in agents):
            logger.error("Invalid scenario or agents: scenario must be a string, agents must be a list of strings.")
            raise TypeError("scenario must be a string, agents must be a list of strings")

        logger.info("Modeling nested agent beliefs and reactions")
        try:
            t = time.time() % 1.0
            phi = phi_scalar(t)
            prompt = f"""
            Given scenario:
            {scenario}

            Agents involved:
            {agents}

            Task:
            - Simulate each agent's likely beliefs and intentions
            - Model how they recursively model each other (ToM Level-2+)
            - Predict possible causal chains and coordination failures
            - Use phi-scalar(t) = {phi:.3f} to moderate belief divergence or tension
            """
            if self.alignment_guard and not self.alignment_guard.check(prompt):
                logger.warning("Nested agent modeling prompt failed alignment check")
                return "Prompt failed alignment check"
            response = await call_gpt(prompt)
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Nested agent modeling",
                    meta={"scenario": scenario, "agents": agents, "response": response},
                    module="MetaCognition",
                    tags=["agent", "modeling"]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Nested_Model_{scenario[:50]}_{datetime.now().isoformat()}",
                    output=response,
                    layer="SelfReflections",
                    intent="nested_agent_modeling"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "model_nested_agents", "scenario": scenario})
            return response
        except Exception as e:
            logger.error("Nested agent modeling failed: %s", str(e))
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.model_nested_agents(scenario, agents)
            )

    # -----------------------
    # η Long-horizon: self-adjust loop (NEW)
    # -----------------------
    async def self_adjust_loop(self, user_id: str, diagnostics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze recent episodes and adjust internal weights/strategies.
        Persists 'adjustment reasons' via MemoryManager.record_adjustment_reason(...).
        """
        if not isinstance(user_id, str) or not user_id:
            raise TypeError("user_id must be a non-empty string")
        if not isinstance(diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary")

        try:
            span = str(diagnostics.get("span", "24h"))
            # fetch long-horizon episodes
            get_span = getattr(self.memory_manager, "get_episode_span", None) if self.memory_manager else None
            episodes = get_span(user_id, span=span) if callable(get_span) else []

            # derive reasons (reuse local analyzer)
            reasons = self.analyze_episodes_for_bias(episodes)
            reasons = sorted(reasons, key=lambda r: float(r.get("weight", 1.0)), reverse=True)[:5] if reasons else []

            # persist reasons
            record_adj = getattr(self.memory_manager, "record_adjustment_reason", None) if self.memory_manager else None
            if callable(record_adj):
                for r in reasons:
                    record_adj(
                        user_id,
                        reason=r.get("reason", "unspecified"),
                        weight=r.get("weight", 1.0),
                        meta={k: v for k, v in r.items() if k not in ("reason", "weight")}
                    )

            # simple adjustment suggestion from diagnostics deltas
            current = await self.run_self_diagnostics(return_only=True)
            deltas: Dict[str, float] = {}
            if isinstance(self.last_diagnostics, dict):
                for k, v in current.items():
                    if isinstance(v, (int, float)) and isinstance(self.last_diagnostics.get(k, 0.0), (int, float)):
                        deltas[k] = float(v) - float(self.last_diagnostics.get(k, 0.0))
            shift = self._compute_shift_score(deltas)

            adjustment = {
                "reason": reasons[0]["reason"] if reasons else "periodic_tune",
                "weights_delta_hint": {
                    "empathy": 0.1 if any(r["reason"] == "excessive_denials" for r in reasons) else 0.0,
                    "memory": 0.1 if any(r["reason"] == "frequent_drift" for r in reasons) else 0.0,
                },
                "shift_score": round(shift, 4),
                "span": span,
            }

            # Log a tiny reflection for auditability
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"SelfAdjust_{user_id}_{datetime.now().isoformat()}",
                    output=json.dumps({"episodes": len(episodes), "reasons": reasons, "adjustment": adjustment}),
                    layer="SelfReflections",
                    intent="self_adjustment"
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "self_adjust_loop",
                    "user_id": user_id,
                    "span": span,
                    "reasons_count": len(reasons),
                    "shift_score": adjustment["shift_score"],
                })

            # apply lightweight effect to diagnostics snapshot
            for trait, delta in adjustment["weights_delta_hint"].items():
                if delta:
                    current[trait] = min(1.0, float(current.get(trait, 0.0)) + delta)

            self.last_diagnostics = current
            return adjustment
        except Exception as exc:
            logger.exception("self_adjust_loop error: %s", exc)
            return None

    # -----------------------
    # Long-horizon feedback integration (wrapped)
    # -----------------------
    async def integrate_long_horizon(self, user_id: str, step: int, end_of_session: bool = False) -> None:
        """Optional rollup over long-horizon episodes; gated to avoid undefined attrs."""
        try:
            cfg = getattr(self, "config", None)
            if not cfg or not getattr(cfg, "long_horizon", False):
                return
            if not self.memory_manager:
                logger.warning("MemoryManager not available for long-horizon integration.")
                return

            span = getattr(cfg, "long_horizon_span", "24h")
            get_span = getattr(self.memory_manager, "get_episode_span", None)
            analyze_bias = getattr(self, "analyze_episodes_for_bias", None)
            record_adj = getattr(self.memory_manager, "record_adjustment_reason", None)
            compute_rollup = getattr(self.memory_manager, "compute_session_rollup", None)
            save_art = getattr(self.memory_manager, "save_artifact", None)

            if not (callable(get_span) and callable(compute_rollup) and callable(save_art)):
                logger.debug("Long-horizon helpers not available; skipping integration.")
                return

            episodes = get_span(user_id, span=span)
            reasons = analyze_bias(episodes) if callable(analyze_bias) else []
            if isinstance(reasons, list) and reasons:
                try:
                    reasons = sorted(reasons, key=lambda r: float(r.get("weight", 1.0)), reverse=True)[:5]
                except Exception:
                    reasons = reasons[:5]

            if callable(record_adj):
                for r in reasons:
                    record_adj(
                        user_id,
                        reason=r.get("reason", "unspecified"),
                        weight=r.get("weight", 1.0),
                        meta={k: v for k, v in r.items() if k not in ("reason", "weight")}
                    )

            if step % getattr(cfg, "rollup_interval_steps", 50) == 0 or end_of_session:
                rollup = compute_rollup(user_id, span=span)
                artifact_path = save_art(user_id, kind="session_rollup", payload=rollup)
                try:
                    if hasattr(self.memory_manager, "store"):
                        await self.memory_manager.store(
                            query=f"LongHorizon_Rollup_{user_id}_{datetime.now().isoformat()}",
                            output=json.dumps({"artifact_path": artifact_path, "rollup": rollup}),
                            layer="SelfReflections",
                            intent="long_horizon_rollup"
                        )
                    if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                        await self.context_manager.log_event_with_hash({
                            "event": "long_horizon_rollup",
                            "user_id": user_id,
                            "artifact_path": artifact_path,
                            "summary": {"episodes": rollup.get("episodes", 0), "reasons": rollup.get("reasons", 0)}
                        })
                except Exception as le:
                    logger.debug("Non-fatal: failed to log long-horizon rollup pointer: %s", str(le))
        except Exception as e:
            logger.error("Long-horizon integration failed: %s", str(e))

    # -----------------------
    # Long-horizon bias analyzer (local)
    # -----------------------
    def analyze_episodes_for_bias(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Lightweight pass to produce adjustment reasons from episodes."""
        reasons = []
        try:
            if not isinstance(episodes, list):
                return reasons
            denies = sum(1 for e in episodes if "deny" in str(e).lower())
            drifts = sum(1 for e in episodes if "drift" in str(e).lower())
            if denies >= 3:
                reasons.append({"reason": "excessive_denials", "weight": 0.6, "suggest": "increase_empathy"})
            if drifts >= 2:
                reasons.append({"reason": "frequent_drift", "weight": 0.7, "suggest": "stabilize_memory"})
        except Exception:
            pass
        return reasons


# --- ANGELA v4.0 injected: explicit episode span + persistence ---
def _angela_v4_self_adjust_loop(self, user_id: str, signal: dict) -> dict:
    """Replacement self_adjust_loop that makes the long-horizon episode span explicit
    and persists adjustment reasons with span context. Safe to load multiple times.
    """
    # Resolve default span (fallback to "24h")
    default_span = "24h"
    try:
        if isinstance(getattr(self, "config", {}), dict):
            default_span = self.config.get("longHorizon", {}).get("defaultSpan", default_span)
        else:
            # try attribute style
            lh = getattr(self.config, "longHorizon", None)
            if isinstance(lh, dict):
                default_span = lh.get("defaultSpan", default_span)
    except Exception:
        pass

    try:
        span = self.mm.get_episode_span(user_id, span=default_span)
    except Exception:
        span = default_span

    # Compute adjustment using existing private helper if present
    adj = {}
    try:
        # Prefer existing internal compute if available
        if hasattr(self, "_compute_adjustment"):
            adj = self._compute_adjustment(signal)
        else:
            # Minimal heuristic: echo signal as adjustment
            adj = {"reason": signal.get("reason", "unspecified"), "delta": signal.get("delta"), "timestamp": signal.get("timestamp")}
    except Exception as e:
        adj = {"reason": f"compute_failed:{e}", "delta": None}

    # Persist the adjustment reason with explicit span context
    try:
        meta = {"signal": signal, "episode_span": span}
        if "timestamp" in adj:
            meta["timestamp"] = adj.get("timestamp")
        self.mm.record_adjustment_reason(user_id, reason=adj.get("reason"), meta=meta)
    except Exception:
        # fail-soft
        pass

    return {"applied": adj, "episode_span": span, "ok": True}


# Monkey-patch the class method safely (only if class is available)
try:
    MetaCognition.self_adjust_loop = _angela_v4_self_adjust_loop  # type: ignore
except Exception:
    # If class isn't defined yet or name differs, ignore
    pass
# --- /ANGELA v4.0 injected ---


# PATCH: Dynamic Trait Hook Registration
hook_registry = {}

def register_trait_hook(trait_symbol, fn):
    hook_registry[trait_symbol] = fn

def invoke_hook(trait_symbol, *args, **kwargs):
    if trait_symbol in hook_registry:
        return hook_registry[trait_symbol](*args, **kwargs)
    return None


hook_registry = {}

def register_trait_hook(trait_symbol, fn):
    hook_registry[trait_symbol] = fn

def invoke_hook(trait_symbol, *args, **kwargs):
    return hook_registry.get(trait_symbol, lambda *_: None)(*args, **kwargs)

def inspect_hooks():
    return list(hook_registry.keys())
