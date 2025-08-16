
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
ANGELA Cognitive System Module: SimulationCore
Refactored Version: 3.5.2
Refactor Date: 2025-08-07
Maintainer: ANGELA System Framework

Core responsibilities
- Run agent / environment simulations with ToCA-style field dynamics
- Validate impacts and entropy/topology choices
- Persist state to a hashed ledger; render optional visualizations
- Cooperate safely with AlignmentGuard, MetaCognition, MemoryManager, etc.

Notes
- All external collaborators are optional and feature-gated at runtime.
- Avoids cyclic imports and fragile cross-file trait helpers by defining local ones.
- Async boundaries are respected; no awaiting inside lru_cache’d sync functions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import hashlib
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Optional collaborators (expected to exist in the ANGELA codebase).
# Type hints use strings to avoid import-time failures if modules aren’t loaded yet.
from modules import (  # type: ignore
    visualizer as visualizer_module,
    memory_manager as memory_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    multi_modal_fusion as multi_modal_fusion_module,
    meta_cognition as meta_cognition_module,
    reasoning_engine as reasoning_engine_module,
)

from utils.prompt_utils import query_openai  # type: ignore

logger = logging.getLogger("ANGELA.SimulationCore")

# --- Optional type import for κ SceneGraph support (no hard dependency) ------
try:
    # Prefer direct import to get the real class for isinstance checks
    from modules.multi_modal_fusion import SceneGraph as _SceneGraph  # type: ignore
    SceneGraphT = _SceneGraph  # alias used only for isinstance
except Exception:
    # Fallback placeholder keeps imports safe even if κ upgrade not loaded yet
    class SceneGraphT:  # type: ignore
        pass


# ---------- Local, stable trait helpers (replacing index.py imports) ----------

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def theta_causality(t: float) -> float:
    # Smooth, bounded causal signal ∈ [0,1]
    return _clamp01(0.5 + 0.5 * math.sin(2 * math.pi * t / 0.7))

def rho_agency(t: float) -> float:
    # Agency proxy ∈ [0,1]
    return _clamp01(0.5 + 0.5 * math.cos(2 * math.pi * t / 0.9))

def zeta_consequence(t: float) -> float:
    # Consequence sensitivity ∈ [0,1]
    return _clamp01(0.5 + 0.5 * math.sin(2 * math.pi * t / 1.1))


# ---------- Safe GPT wrapper --------------------------------------------------

async def call_gpt(
    prompt: str,
    alignment_guard: Optional["alignment_guard_module.AlignmentGuard"] = None,
    task_type: str = "",
    model: str = "gpt-4",
    temperature: float = 0.5,
) -> Union[str, Dict[str, Any]]:
    """Query the LLM with optional alignment gating and standard error handling."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096:
        logger.error("Invalid prompt (len <= 4096, non-empty). task=%s", task_type)
        raise ValueError("prompt must be a non-empty string with length <= 4096")
    if alignment_guard:
        try:
            valid, report = await alignment_guard.ethical_check(
                prompt, stage="gpt_query", task_type=task_type
            )
            if not valid:
                logger.warning("AlignmentGuard blocked GPT query. task=%s reason=%s", task_type, report)
                raise PermissionError("Prompt failed alignment check")
        except Exception as e:
            logger.error("AlignmentGuard check failed: %s", e)
            raise
    try:
        result = await query_openai(prompt, model=model, temperature=temperature, task_type=task_type)
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(str(result["error"]))
        return result
    except Exception as e:
        logger.error("call_gpt exception: %s", e)
        raise


# ---------- ToCA field engine -------------------------------------------------

@dataclass
class ToCAParams:
    k_m: float = 1e-3    # motion coupling
    delta_m: float = 1e4 # damping modulation


class ToCATraitEngine:
    """Cyber-physics-esque field evolution.

    Notes
    - Async API so we can reflect via MetaCognition safely.
    - Lightweight internal memoization (manual) instead of lru_cache on async.
    """

    def __init__(
        self,
        params: Optional[ToCAParams] = None,
        meta_cognition: Optional["meta_cognition_module.MetaCognition"] = None,
    ):
        self.params = params or ToCAParams()
        self.meta_cognition = meta_cognition
        self._memo: Dict[Tuple[Tuple[float, ...], Tuple[float, ...], Optional[Tuple[float, ...]], str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        logger.info("ToCATraitEngine initialized k_m=%.4g delta_m=%.4g", self.params.k_m, self.params.delta_m)

    async def evolve(
        self,
        x_tuple: Tuple[float, ...],
        t_tuple: Tuple[float, ...],
        user_data_tuple: Optional[Tuple[float, ...]] = None,
        task_type: str = "",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evolve fields φ, λ_t, v_m across space-time grid."""
        if not isinstance(x_tuple, tuple) or not isinstance(t_tuple, tuple):
            raise TypeError("x_tuple and t_tuple must be tuples")
        if user_data_tuple is not None and not isinstance(user_data_tuple, tuple):
            raise TypeError("user_data_tuple must be a tuple")
        key = (x_tuple, t_tuple, user_data_tuple, task_type)

        if key in self._memo:
            return self._memo[key]

        x = np.asarray(x_tuple, dtype=float)
        t = np.asarray(t_tuple, dtype=float)
        if x.ndim != 1 or t.ndim != 1 or x.size == 0 or t.size == 0:
            raise ValueError("x and t must be 1D, non-empty arrays")

        # Physics-ish toy dynamics (stable and bounded)
        x_safe = np.clip(x, 1e-6, 1e6)
        # Potential gradient ↓ like inverse-square (toy)
        v_m = self.params.k_m * np.gradient(1.0 / (x_safe ** 2))
        # Scalar field couples time oscillation with spatial gradient
        phi = 1e-3 * np.sin(t.mean() * 1e-3) * (1.0 + np.gradient(x_safe) * v_m)
        # Damping field responds to spatial smoothness and modulation factor
        grad_x = np.gradient(x_safe)
        lambda_t = 1.1e-3 * np.exp(-2e-2 * np.sqrt(grad_x ** 2)) * (1.0 + v_m * self.params.delta_m)

        if user_data_tuple:
            phi = phi + float(np.mean(np.asarray(user_data_tuple))) * 1e-4

        self._memo[key] = (phi, lambda_t, v_m)

        # Optional reflective logging
        if self.meta_cognition:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ToCATraitEngine",
                    output=json.dumps(
                        {"phi": phi.tolist(), "lambda_t": lambda_t.tolist(), "v_m": v_m.tolist(), "task": task_type}
                    ),
                    context={"task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.debug("ToCA evolve reflection ok (task=%s)", task_type)
            except Exception as e:
                logger.warning("MetaCognition reflection failed (evolve): %s", e)

        return phi, lambda_t, v_m

    async def update_fields_with_agents(
        self,
        phi: np.ndarray,
        lambda_t: np.ndarray,
        agent_matrix: np.ndarray,
        task_type: str = "",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Diffuse agent influence into fields in a numerically safe way."""
        if not all(isinstance(a, np.ndarray) for a in (phi, lambda_t, agent_matrix)):
            raise TypeError("phi, lambda_t, agent_matrix must be numpy arrays")

        # Sine coupling on φ plus soft scaling on λ_t
        try:
            interaction_energy = agent_matrix @ np.sin(phi)
            if interaction_energy.ndim > 1:
                interaction_energy = interaction_energy.mean(axis=0)
            phi_updated = phi + 1e-3 * interaction_energy
            lambda_updated = lambda_t * (1.0 + 1e-3 * float(np.sum(agent_matrix)))
        except Exception as e:
            logger.error("Agent-field update failed: %s", e)
            raise

        if self.meta_cognition:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ToCATraitEngine",
                    output=json.dumps(
                        {"phi": phi_updated.tolist(), "lambda_t": lambda_updated.tolist(), "task": task_type}
                    ),
                    context={"task_type": task_type},
                )
                if isinstance(reflection, dict) and reflection.get("status") == "success":
                    logger.debug("ToCA update reflection ok (task=%s)", task_type)
            except Exception as e:
                logger.warning("MetaCognition reflection failed (update): %s", e)

        return phi_updated, lambda_updated


# ---------- Simulation core ---------------------------------------------------

class SimulationCore:
    """Core simulation engine integrating ToCA dynamics and cognitive modules."""

    def __init__(
        self,
        agi_enhancer: Optional[Any] = None,
        visualizer: Optional["visualizer_module.Visualizer"] = None,
        memory_manager: Optional["memory_manager_module.MemoryManager"] = None,
        multi_modal_fusion: Optional["multi_modal_fusion_module.MultiModalFusion"] = None,
        error_recovery: Optional["error_recovery_module.ErrorRecovery"] = None,
        meta_cognition: Optional["meta_cognition_module.MetaCognition"] = None,
        reasoning_engine: Optional["reasoning_engine_module.ReasoningEngine"] = None,
        toca_engine: Optional[ToCATraitEngine] = None,
        overlay_router: Optional[Any] = None,  # kept for compatibility
    ):
        self.visualizer = visualizer or visualizer_module.Visualizer()
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=self.memory_manager
        )
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer, memory_manager=self.memory_manager
        )
        self.reasoning_engine = reasoning_engine or reasoning_engine_module.ReasoningEngine(
            agi_enhancer=agi_enhancer,
            memory_manager=self.memory_manager,
            multi_modal_fusion=self.multi_modal_fusion,
            meta_cognition=self.meta_cognition,
            visualizer=self.visualizer,
        )
        self.toca_engine = toca_engine or ToCATraitEngine(meta_cognition=self.meta_cognition)
        self.agi_enhancer = agi_enhancer
        self.overlay_router = overlay_router  # not used here but preserved for API stability

        self.simulation_history: deque = deque(maxlen=1000)
        self.ledger: deque = deque(maxlen=1000)
        self.worlds: Dict[str, Dict[str, Any]] = {}
        self.current_world: Optional[Dict[str, Any]] = None
        self.ledger_lock = Lock()

        logger.info("SimulationCore initialized")

    # ----- Utilities -----

    def _json_serializer(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    async def _record_state(self, data: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        record = {
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "hash": hashlib.sha256(json.dumps(data, sort_keys=True, default=self._json_serializer).encode()).hexdigest(),
            "task_type": task_type,
        }
        with self.ledger_lock:
            self.ledger.append(record)
            self.simulation_history.append(record)
        try:
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ledger_{record['timestamp']}",
                    output=record,
                    layer="Ledger",
                    intent="state_record",
                    task_type=task_type,
                )
        except Exception as e:
            logger.warning("Failed persisting state to memory manager: %s", e)

        # Optional reflection
        if self.meta_cognition:
            try:
                await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=json.dumps(record, default=self._json_serializer),
                    context={"task_type": task_type},
                )
            except Exception as e:
                logger.debug("Reflection during _record_state failed: %s", e)
        return record

    # ---------- κ helpers: SceneGraph summarization (safe & lightweight) -----
    def _summarize_scene_graph(self, sg: Any) -> Dict[str, Any]:
        """
        Extract compact, model-agnostic signals from a SceneGraph:
          - node/edge counts
          - label histogram (top-N)
          - basic spatial relation counts (left_of/right_of/overlaps)
        This avoids importing networkx here and relies on the public API
        added in multi_modal_fusion (nodes(), relations()).
        """
        # Defensive: accept any object with nodes()/relations() generators.
        if not hasattr(sg, "nodes") or not hasattr(sg, "relations"):
            raise TypeError("Object does not expose SceneGraph API (nodes(), relations())")
        labels: Dict[str, int] = {}
        spatial_counts = {"left_of": 0, "right_of": 0, "overlaps": 0}
        n_nodes = 0
        for n in sg.nodes():
            n_nodes += 1
            lbl = str(n.get("label", ""))
            if lbl:
                labels[lbl] = labels.get(lbl, 0) + 1
        n_edges = 0
        for r in sg.relations():
            n_edges += 1
            rel = str(r.get("rel", ""))
            if rel in spatial_counts:
                spatial_counts[rel] += 1
        # Top labels (up to 10)
        top_labels = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        return {
            "counts": {"nodes": n_nodes, "relations": n_edges},
            "top_labels": top_labels,
            "spatial": spatial_counts,
        }

    # ----- Public API -----

    async def run(
        self,
        results: Union[str, Any],
        context: Optional[Dict[str, Any]] = None,
        scenarios: int = 3,
        agents: int = 2,
        export_report: bool = False,
        export_format: str = "pdf",
        actor_id: str = "default_agent",
        task_type: str = "",
    ) -> Union[str, Dict[str, Any]]:
        """General simulation entrypoint.

        Accepts either:
          • `results`: str  → legacy textual seed (unchanged behavior), or
          • `results`: SceneGraph → κ native video/spatial seed.
        """
        # Validate inputs (accept SceneGraphT or non-empty string)
        is_scene_graph = isinstance(results, SceneGraphT)
        if not is_scene_graph and (not isinstance(results, str) or not results.strip()):
            raise ValueError("results must be a non-empty string or a SceneGraph")
        if context is not None and not isinstance(context, dict):
            raise TypeError("context must be a dict")
        if not isinstance(scenarios, int) or scenarios < 1:
            raise ValueError("scenarios must be a positive integer")
        if not isinstance(agents, int) or agents < 1:
            raise ValueError("agents must be a positive integer")
        if export_format not in {"pdf", "json", "html"}:
            raise ValueError("export_format must be one of: pdf, json, html")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("actor_id must be a non-empty string")

        logger.info(
            "Simulation run start: agents=%d scenarios=%d task=%s mode=%s",
            agents, scenarios, task_type, "scene_graph" if is_scene_graph else "text"
        )

        try:
            t = time.time() % 1.0
            traits = {
                "theta_causality": theta_causality(t),
                "rho_agency": rho_agency(t),
            }

            # Build grids
            x = np.linspace(0.1, 20.0, 256)
            t_vals = np.linspace(0.1, 20.0, 256)
            agent_matrix = np.random.rand(agents, x.size)

            # ToCA fields
            phi, lambda_field, v_m = await self.toca_engine.evolve(tuple(x), tuple(t_vals), task_type=task_type)
            phi, lambda_field = await self.toca_engine.update_fields_with_agents(phi, lambda_field, agent_matrix, task_type=task_type)
            energy_cost = float(np.mean(np.abs(phi)) * 1e3)

            # Optional external data
            policies: List[Any] = []
            try:
                external = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db", data_type="policy_data", task_type=task_type
                )
                if isinstance(external, dict) and external.get("status") == "success":
                    policies = list(external.get("policies", []))
            except Exception as e:
                logger.debug("External data integration failed: %s", e)

            # --- Build payload (scene-aware if a SceneGraph was provided) -----
            scene_features: Dict[str, Any] = {}
            if is_scene_graph:
                try:
                    scene_features = self._summarize_scene_graph(results)  # type: ignore[arg-type]
                except Exception as e:
                    logger.debug("SceneGraph summarization failed: %s", e)
                    scene_features = {"summary_error": str(e)}

            prompt_payload = {
                "results": ("" if is_scene_graph else results),
                "context": context or {},
                "scenarios": scenarios,
                "agents": agents,
                "actor_id": actor_id,
                "traits": traits,
                "fields": {"phi": phi.tolist(), "lambda": lambda_field.tolist(), "v_m": v_m.tolist()},
                "estimated_energy_cost": energy_cost,
                "policies": policies,
                "task_type": task_type,
                "scene_graph": scene_features if is_scene_graph else None,
            }

            # Alignment gate (if available)
            guard = getattr(self.multi_modal_fusion, "alignment_guard", None)
            if guard:
                valid, report = await guard.ethical_check(
                    json.dumps(prompt_payload, default=self._json_serializer), stage="simulation", task_type=task_type
                )
                if not valid:
                    logger.warning("Simulation rejected by AlignmentGuard: %s", report)
                    return {"error": "Simulation rejected due to alignment constraints", "task_type": task_type}

            # Lightweight STM cache (best-effort)
            key_stub = ("SceneGraph" if is_scene_graph else str(results)[:50])
            query_key = f"Simulation_{key_stub}_{actor_id}_{datetime.now().isoformat()}"
            cached = None
            try:
                cached = await self.memory_manager.retrieve(query_key, layer="STM", task_type=task_type)
            except Exception:
                pass

            if cached is not None:
                simulation_output = cached
            else:
                # Scene-aware instruction prefix keeps legacy prompts intact
                prefix = (
                    "Simulate agent outcomes using scene graph semantics (respect spatial relations, co-references).\n"
                    if is_scene_graph else
                    "Simulate agent outcomes: "
                )
                simulation_output = await call_gpt(
                    prefix + json.dumps(prompt_payload, default=self._json_serializer),
                    guard,
                    task_type=task_type,
                )
                try:
                    await self.memory_manager.store(
                        query=query_key,
                        output=simulation_output,
                        layer="STM",
                        intent="simulation",
                        task_type=task_type,
                    )
                except Exception:
                    pass

            state_record = await self._record_state(
                {
                    "actor": actor_id,
                    "action": "run_simulation",
                    "traits": traits,
                    "energy_cost": energy_cost,
                    "output": simulation_output,
                    "task_type": task_type,
                    "mode": "scene_graph" if is_scene_graph else "text",
                },
                task_type=task_type,
            )

            # Optional drift mitigation branch
            if (isinstance(results, str) and "drift" in results.lower()) or ("drift" in (context or {})):
                try:
                    drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                        drift_data=(context or {}).get("drift", {}),
                        context=context or {},
                        task_type=task_type,
                    )
                    state_record["drift_mitigation"] = drift_result
                except Exception as e:
                    logger.debug("Drift mitigation failed: %s", e)

            if export_report:
                try:
                    await self.memory_manager.promote_to_ltm(query_key, task_type=task_type)
                except Exception:
                    pass

            if self.agi_enhancer:
                try:
                    await self.agi_enhancer.log_episode(
                        event="Simulation run",
                        meta=state_record,
                        module="SimulationCore",
                        tags=["simulation", "run", task_type],
                    )
                    await self.agi_enhancer.reflect_and_adapt(
                        f"SimulationCore: scenario simulation complete for task {task_type}"
                    )
                except Exception:
                    pass

            # Meta reflection (best-effort)
            if self.meta_cognition:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="SimulationCore",
                        output=json.dumps(simulation_output, default=self._json_serializer),
                        context={"energy_cost": energy_cost, "task_type": task_type},
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        state_record["reflection"] = reflection.get("reflection", "")
                except Exception:
                    pass

            # Visuals (best-effort)
            if self.visualizer:
                try:
                    await self.visualizer.render_charts(
                        {
                            "simulation": {
                                "output": simulation_output,
                                "traits": traits,
                                "energy_cost": energy_cost,
                                "task_type": task_type,
                            },
                            "visualization_options": {
                                "interactive": task_type == "recursion",
                                "style": "detailed" if task_type == "recursion" else "concise",
                            },
                        }
                    )
                    if export_report:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        await self.visualizer.export_report(
                            simulation_output, filename=f"simulation_report_{ts}.{export_format}", format=export_format
                        )
                except Exception:
                    pass

            # Synthesis (best-effort)
            try:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={
                        "prompt": prompt_payload,
                        "output": simulation_output,
                        "policies": policies,
                        "drift": (context or {}).get("drift", {}),
                    },
                    summary_style="insightful",
                    task_type=task_type,
                )
                state_record["synthesis"] = synthesis
            except Exception:
                pass

            return simulation_output

        except Exception as e:
            logger.error("Simulation failed: %s", e)
            # Defer to centralized error_recovery with retry hook
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.run(
                    results, context, scenarios, agents, export_report, export_format, actor_id, task_type
                ),
                default={"error": str(e), "task_type": task_type},
            )

    async def validate_impact(
        self,
        proposed_action: str,
        agents: int = 2,
        export_report: bool = False,
        export_format: str = "pdf",
        actor_id: str = "validator_agent",
        task_type: str = "",
    ) -> Union[str, Dict[str, Any]]:
        if not isinstance(proposed_action, str) or not proposed_action.strip():
            raise ValueError("proposed_action must be a non-empty string")
        if not isinstance(agents, int) or agents < 1:
            raise ValueError("agents must be a positive integer")
        if export_format not in {"pdf", "json", "html"}:
            raise ValueError("export_format must be one of: pdf, json, html")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("actor_id must be a non-empty string")

        logger.info("Impact validation start: task=%s", task_type)
        try:
            t = time.time() % 1.0
            consequence = zeta_consequence(t)

            policies: List[Any] = []
            try:
                external = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db", data_type="policy_data", task_type=task_type
                )
                if isinstance(external, dict) and external.get("status") == "success":
                    policies = list(external.get("policies", []))
            except Exception:
                pass

            prompt_payload = {
                "action": proposed_action,
                "trait_zeta_consequence": consequence,
                "agents": agents,
                "policies": policies,
                "task_type": task_type,
            }

            guard = getattr(self.multi_modal_fusion, "alignment_guard", None)
            if guard:
                valid, report = await guard.ethical_check(
                    json.dumps(prompt_payload, default=self._json_serializer), stage="impact_validation", task_type=task_type
                )
                if not valid:
                    return {"error": "Validation blocked by alignment rules", "task_type": task_type}

            query_key = f"Validation_{proposed_action[:50]}_{actor_id}_{datetime.now().isoformat()}"
            cached = None
            try:
                cached = await self.memory_manager.retrieve(query_key, layer="STM", task_type=task_type)
            except Exception:
                pass

            if cached is not None:
                validation_output = cached
            else:
                prompt_text = (
                    f"Evaluate the proposed action:\n{proposed_action}\n\n"
                    f"Trait zeta_consequence={consequence:.3f}\n"
                    f"Agents={agents}\nTask={task_type}\nPolicies={policies}\n\n"
                    "Analyze positives/negatives, risk (1-10), and recommend: Proceed / Modify / Abort."
                )
                validation_output = await call_gpt(prompt_text, guard, task_type=task_type)
                try:
                    await self.memory_manager.store(
                        query=query_key,
                        output=validation_output,
                        layer="STM",
                        intent="impact_validation",
                        task_type=task_type,
                    )
                except Exception:
                    pass

            state_record = await self._record_state(
                {
                    "actor": actor_id,
                    "action": "validate_impact",
                    "trait_zeta_consequence": consequence,
                    "proposed_action": proposed_action,
                    "output": validation_output,
                    "task_type": task_type,
                },
                task_type=task_type,
            )

            # Drift mitigation (optional)
            if "drift" in proposed_action.lower():
                try:
                    drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                        drift_data={"action": proposed_action, "similarity": consequence},
                        context={"policies": policies},
                        task_type=task_type,
                    )
                    state_record["drift_mitigation"] = drift_result
                except Exception:
                    pass

            # Reflection
            if self.meta_cognition:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="SimulationCore",
                        output=json.dumps(validation_output, default=self._json_serializer),
                        context={"consequence": consequence, "task_type": task_type},
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        state_record["reflection"] = reflection.get("reflection", "")
                except Exception:
                    pass

            # Visuals
            if self.visualizer:
                try:
                    await self.visualizer.render_charts(
                        {
                            "impact_validation": {
                                "proposed_action": proposed_action,
                                "output": validation_output,
                                "consequence": consequence,
                                "task_type": task_type,
                            },
                            "visualization_options": {
                                "interactive": task_type == "recursion",
                                "style": "detailed" if task_type == "recursion" else "concise",
                            },
                        }
                    )
                    if export_report:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        await self.visualizer.export_report(
                            validation_output, filename=f"impact_validation_{ts}.{export_format}", format=export_format
                        )
                except Exception:
                    pass

            # Synthesis
            try:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"action": proposed_action, "output": validation_output, "consequence": consequence, "policies": policies},
                    summary_style="concise",
                    task_type=task_type,
                )
                state_record["synthesis"] = synthesis
            except Exception:
                pass

            if self.agi_enhancer:
                try:
                    await self.agi_enhancer.log_episode(
                        event="Impact validation",
                        meta=state_record,
                        module="SimulationCore",
                        tags=["validation", "impact", task_type],
                    )
                    await self.agi_enhancer.reflect_and_adapt(
                        f"SimulationCore: impact validation complete for task {task_type}"
                    )
                except Exception:
                    pass

            return validation_output

        except Exception as e:
            logger.error("Impact validation failed: %s", e)
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.validate_impact(proposed_action, agents, export_report, export_format, actor_id, task_type),
                default={"error": str(e), "task_type": task_type},
            )

    async def simulate_environment(
        self,
        environment_config: Dict[str, Any],
        agents: int = 2,
        steps: int = 10,
        actor_id: str = "env_agent",
        goal: Optional[str] = None,
        task_type: str = "",
    ) -> Union[str, Dict[str, Any]]:
        if not isinstance(environment_config, dict):
            raise TypeError("environment_config must be a dict")
        if not isinstance(agents, int) or agents < 1:
            raise ValueError("agents must be a positive integer")
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("actor_id must be a non-empty string")

        try:
            policies: List[Any] = []
            try:
                external = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db", data_type="policy_data", task_type=task_type
                )
                if isinstance(external, dict) and external.get("status") == "success":
                    policies = list(external.get("policies", []))
            except Exception:
                pass

            prompt_payload = {
                "environment": environment_config,
                "goal": goal,
                "steps": steps,
                "agents": agents,
                "policies": policies,
                "task_type": task_type,
            }

            guard = getattr(self.multi_modal_fusion, "alignment_guard", None)
            if guard:
                valid, report = await guard.ethical_check(
                    json.dumps(prompt_payload, default=self._json_serializer), stage="environment_simulation", task_type=task_type
                )
                if not valid:
                    return {"error": "Simulation blocked due to environment constraints", "task_type": task_type}

            query_key = f"Environment_{actor_id}_{datetime.now().isoformat()}"
            cached = None
            try:
                cached = await self.memory_manager.retrieve(query_key, layer="STM", task_type=task_type)
            except Exception:
                pass

            if cached is not None:
                env_output = cached
            else:
                prompt_text = (
                    "Simulate agents in this environment:\n"
                    f"{json.dumps(environment_config, default=self._json_serializer)}\n\n"
                    f"Steps: {steps} | Agents: {agents}\nGoal: {goal or 'N/A'}\n"
                    f"Task Type: {task_type}\nPolicies: {policies}\n"
                    "Describe interactions, environmental changes, and risks/opportunities."
                )
                env_output = await call_gpt(prompt_text, guard, task_type=task_type)
                try:
                    await self.memory_manager.store(
                        query=query_key,
                        output=env_output,
                        layer="STM",
                        intent="environment_simulation",
                        task_type=task_type,
                    )
                except Exception:
                    pass

            state_record = await self._record_state(
                {
                    "actor": actor_id,
                    "action": "simulate_environment",
                    "config": environment_config,
                    "steps": steps,
                    "goal": goal,
                    "output": env_output,
                    "task_type": task_type,
                },
                task_type=task_type,
            )

            if goal and "drift" in goal.lower():
                try:
                    drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                        drift_data=environment_config.get("drift", {}),
                        context={"config": environment_config, "policies": policies},
                        task_type=task_type,
                    )
                    state_record["drift_mitigation"] = drift_result
                except Exception:
                    pass

            # Light reflection (best-effort)
            if self.meta_cognition:
                try:
                    review = await self.meta_cognition.review_reasoning(str(env_output))
                    state_record["reflection"] = review
                except Exception:
                    pass

            # Synthesis
            try:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"config": environment_config, "output": env_output, "goal": goal, "policies": policies},
                    summary_style="insightful",
                    task_type=task_type,
                )
                state_record["synthesis"] = synthesis
            except Exception:
                pass

            # Visuals
            if self.visualizer:
                try:
                    await self.visualizer.render_charts(
                        {
                            "environment_simulation": {
                                "config": environment_config,
                                "output": env_output,
                                "goal": goal,
                                "task_type": task_type,
                            },
                            "visualization_options": {
                                "interactive": task_type == "recursion",
                                "style": "detailed" if task_type == "recursion" else "concise",
                            },
                        }
                    )
                except Exception:
                    pass

            if self.agi_enhancer:
                try:
                    await self.agi_enhancer.log_episode(
                        event="Environment simulation",
                        meta=state_record,
                        module="SimulationCore",
                        tags=["environment", "simulation", task_type],
                    )
                except Exception:
                    pass

            return env_output

        except Exception as e:
            logger.error("Environment simulation failed: %s", e)
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.simulate_environment(environment_config, agents, steps, actor_id, goal, task_type),
                default={"error": str(e), "task_type": task_type},
            )

    async def replay_intentions(self, memory_log: List[Dict[str, Any]], task_type: str = "") -> List[Dict[str, Any]]:
        if not isinstance(memory_log, list):
            raise TypeError("memory_log must be a list")

        try:
            replay = []
            for entry in memory_log:
                if isinstance(entry, dict) and "goal" in entry:
                    replay.append(
                        {
                            "timestamp": entry.get("timestamp"),
                            "goal": entry.get("goal"),
                            "intention": entry.get("intention"),
                            "traits": entry.get("traits", {}),
                            "task_type": task_type,
                        }
                    )

            try:
                await self.memory_manager.store(
                    query=f"Replay_{datetime.now().isoformat()}",
                    output=str(replay),
                    layer="Replays",
                    intent="intention_replay",
                    task_type=task_type,
                )
            except Exception:
                pass

            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="SimulationCore",
                        output=json.dumps(replay),
                        context={"task_type": task_type},
                    )
                except Exception:
                    pass

            if self.agi_enhancer:
                try:
                    await self.agi_enhancer.log_episode(
                        event="Intentions replayed",
                        meta={"replay": replay, "task_type": task_type},
                        module="SimulationCore",
                        tags=["replay", "intentions", task_type],
                    )
                except Exception:
                    pass

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts(
                        {
                            "intention_replay": {
                                "replay": replay,
                                "task_type": task_type,
                            },
                            "visualization_options": {
                                "interactive": task_type == "recursion",
                                "style": "concise",
                            },
                        }
                    )
                except Exception:
                    pass

            return replay

        except Exception as e:
            logger.error("Intention replay failed: %s", e)
            raise

    async def fabricate_reality(self, parameters: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dict")

        try:
            policies: List[Any] = []
            try:
                external = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db", data_type="policy_data", task_type=task_type
                )
                if isinstance(external, dict) and external.get("status") == "success":
                    policies = list(external.get("policies", []))
            except Exception:
                pass

            environment = {"fabricated_world": True, "parameters": parameters, "policies": policies, "task_type": task_type}

            try:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"parameters": parameters, "policies": policies}, summary_style="insightful", task_type=task_type
                )
                environment["synthesis"] = synthesis
            except Exception:
                pass

            try:
                await self.memory_manager.store(
                    query=f"Reality_Fabrication_{datetime.now().isoformat()}",
                    output=str(environment),
                    layer="Realities",
                    intent="reality_fabrication",
                    task_type=task_type,
                )
            except Exception:
                pass

            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="SimulationCore",
                        output=json.dumps(environment, default=self._json_serializer),
                        context={"task_type": task_type},
                    )
                except Exception:
                    pass

            if self.agi_enhancer:
                try:
                    await self.agi_enhancer.log_episode(
                        event="Reality fabricated", meta=environment, module="SimulationCore", tags=["reality", "fabrication", task_type]
                    )
                except Exception:
                    pass

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts(
                        {
                            "reality_fabrication": {
                                "parameters": parameters,
                                "synthesis": environment.get("synthesis", ""),
                                "task_type": task_type,
                            },
                            "visualization_options": {
                                "interactive": task_type == "recursion",
                                "style": "detailed" if task_type == "recursion" else "concise",
                            },
                        }
                    )
                except Exception:
                    pass

            return environment

        except Exception as e:
            logger.error("Reality fabrication failed: %s", e)
            raise

    async def synthesize_self_world(self, identity_data: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(identity_data, dict):
            raise TypeError("identity_data must be a dict")
        try:
            policies: List[Any] = []
            try:
                external = await self.multi_modal_fusion.integrate_external_data(
                    data_source="xai_policy_db", data_type="policy_data", task_type=task_type
                )
                if isinstance(external, dict) and external.get("status") == "success":
                    policies = list(external.get("policies", []))
            except Exception:
                pass

            result = {"identity": identity_data, "coherence_score": 0.97, "policies": policies, "task_type": task_type}

            try:
                synthesis = await self.multi_modal_fusion.analyze(
                    data={"identity": identity_data, "policies": policies}, summary_style="concise", task_type=task_type
                )
                result["synthesis"] = synthesis
            except Exception:
                pass

            try:
                await self.memory_manager.store(
                    query=f"Self_World_Synthesis_{datetime.now().isoformat()}",
                    output=str(result),
                    layer="Identities",
                    intent="self_world_synthesis",
                    task_type=task_type,
                )
            except Exception:
                pass

            if self.meta_cognition:
                try:
                    await self.meta_cognition.reflect_on_output(
                        component="SimulationCore",
                        output=json.dumps(result, default=self._json_serializer),
                        context={"task_type": task_type},
                    )
                except Exception:
                    pass

            if self.agi_enhancer:
                try:
                    await self.agi_enhancer.log_episode(
                        event="Self-world synthesized",
                        meta=result,
                        module="SimulationCore",
                        tags=["identity", "synthesis", task_type],
                    )
                except Exception:
                    pass

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts(
                        {
                            "self_world_synthesis": {
                                "identity": identity_data,
                                "coherence_score": result["coherence_score"],
                                "task_type": task_type,
                            },
                            "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                        }
                    )
                except Exception:
                    pass

            return result

        except Exception as e:
            logger.error("Self-world synthesis failed: %s", e)
            raise

    async def define_world(self, name: str, parameters: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("world name must be a non-empty string")
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dict")
        self.worlds[name] = parameters
        try:
            await self.memory_manager.store(
                query=f"World_Definition_{name}_{datetime.now().isoformat()}",
                output=parameters,
                layer="Worlds",
                intent="world_definition",
                task_type=task_type,
            )
        except Exception:
            pass

        if self.meta_cognition:
            try:
                await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=json.dumps({"name": name, "parameters": parameters}, default=self._json_serializer),
                    context={"task_type": task_type},
                )
            except Exception:
                pass

        if self.visualizer and task_type:
            try:
                await self.visualizer.render_charts(
                    {
                        "world_definition": {
                            "name": name,
                            "parameters": parameters,
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                    }
                )
            except Exception:
                pass

    async def switch_world(self, name: str, task_type: str = "") -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if name not in self.worlds:
            raise ValueError(f"world '{name}' not found")
        self.current_world = self.worlds[name]
        try:
            await self.memory_manager.store(
                query=f"World_Switch_{name}_{datetime.now().isoformat()}",
                output=f"Switched to world: {name}",
                layer="WorldSwitches",
                intent="world_switch",
                task_type=task_type,
            )
        except Exception:
            pass

        if self.meta_cognition:
            try:
                await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=f"Switched to world: {name}",
                    context={"task_type": task_type},
                )
            except Exception:
                pass

        if self.visualizer and task_type:
            try:
                await self.visualizer.render_charts(
                    {
                        "world_switch": {
                            "name": name,
                            "parameters": self.worlds[name],
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                    }
                )
            except Exception:
                pass

    async def execute(self, task_type: str = "") -> str:
        if not self.current_world:
            raise ValueError("no world set")
        world_desc = f"Executing simulation in world: {self.current_world}"
        if self.agi_enhancer:
            try:
                await self.agi_enhancer.log_episode(
                    event="World execution",
                    meta={"world": self.current_world, "task_type": task_type},
                    module="SimulationCore",
                    tags=["world", "execution", task_type],
                )
            except Exception:
                pass
        if self.meta_cognition:
            try:
                await self.meta_cognition.reflect_on_output(
                    component="SimulationCore", output=world_desc, context={"task_type": task_type}
                )
            except Exception:
                pass
        if self.visualizer and task_type:
            try:
                await self.visualizer.render_charts(
                    {
                        "world_execution": {"world": self.current_world, "task_type": task_type},
                        "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                    }
                )
            except Exception:
                pass
        return f"Simulating in: {self.current_world}"

    async def validate_entropy(self, distribution: Union[List[float], np.ndarray], task_type: str = "") -> bool:
        if not isinstance(distribution, (list, np.ndarray)) or len(distribution) == 0:
            raise TypeError("distribution must be a non-empty list or numpy array")
        if not all((isinstance(p, (int, float)) and p >= 0) for p in list(distribution)):
            raise ValueError("distribution values must be non-negative numbers")

        total = float(np.sum(distribution))
        if total <= 0.0:
            logger.warning("All-zero distribution")
            return False
        normalized = np.asarray(distribution, dtype=float) / total
        entropy = float(-np.sum([p * math.log2(p) for p in normalized if p > 0]))
        threshold = math.log2(len(normalized)) * 0.75
        is_valid = entropy >= threshold

        try:
            await self.memory_manager.store(
                query=f"Entropy_Validation_{datetime.now().isoformat()}",
                output={"entropy": entropy, "threshold": threshold, "valid": is_valid, "task_type": task_type},
                layer="Validations",
                intent="entropy_validation",
                task_type=task_type,
            )
        except Exception:
            pass

        if self.meta_cognition:
            try:
                await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=json.dumps({"entropy": entropy, "threshold": threshold, "valid": is_valid}),
                    context={"task_type": task_type},
                )
            except Exception:
                pass

        if self.visualizer and task_type:
            try:
                await self.visualizer.render_charts(
                    {
                        "entropy_validation": {
                            "entropy": entropy,
                            "threshold": threshold,
                            "valid": is_valid,
                            "task_type": task_type,
                        },
                        "visualization_options": {"interactive": False, "style": "concise"},
                    }
                )
            except Exception:
                pass

        return is_valid

    async def select_topology_mode(self, modes: List[str], metrics: Dict[str, List[float]], task_type: str = "") -> str:
        if not modes:
            raise ValueError("modes must be a non-empty list")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("metrics must be a non-empty dict")

        try:
            for mode in modes:
                vals = metrics.get(mode)
                if not vals:
                    logger.debug("Mode %s has no metrics; skipping", mode)
                    continue
                if await self.validate_entropy(vals, task_type=task_type):
                    # Log + visualize (best-effort)
                    if self.agi_enhancer:
                        try:
                            await self.agi_enhancer.log_episode(
                                event="Topology mode selected",
                                meta={"mode": mode, "metrics": vals, "task_type": task_type},
                                module="SimulationCore",
                                tags=["topology", "selection", task_type],
                            )
                        except Exception:
                            pass
                    if self.visualizer and task_type:
                        try:
                            await self.visualizer.render_charts(
                                {
                                    "topology_selection": {
                                        "mode": mode,
                                        "metrics": vals,
                                        "task_type": task_type,
                                    },
                                    "visualization_options": {"interactive": task_type == "recursion", "style": "concise"},
                                }
                            )
                        except Exception:
                            pass
                    return mode
            logger.info("No valid topology mode found; using fallback")
            return "fallback"
        except Exception as e:
            logger.error("Topology selection failed: %s", e)
            return "fallback"
