"""
ANGELA Cognitive System Module: User Profile Management
Version: 3.5.2  # Upgraded: Σ self-schema, atomic save, async fixes, drift serialization
Date: 2025-08-09
Maintainer: ANGELA System Framework

Manages user profiles, preferences, and identity tracking with ε-modulation and AGI auditing.
"""

from __future__ import annotations

import logging
import json
import math
import os
from uuid import uuid4
from typing import Dict, Optional, Any, List, Iterable, Tuple, Union, TypedDict, cast
from datetime import datetime
from pathlib import Path
from threading import Lock
from collections import deque
from functools import lru_cache

# --- Logging -----------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ANGELA.Core")

# --- Import compatibility (supports flat or 'modules/' layout) ---------------
def _try_import(name_flat: str, name_mod: str):
    try:
        return __import__(name_mod, fromlist=['*'])
    except Exception:
        return __import__(name_flat, fromlist=['*'])

# Orchestrator & subsystems
SimulationCore = _try_import("simulation_core", "modules.simulation_core").SimulationCore
MemoryManager = _try_import("memory_manager", "modules.memory_manager").MemoryManager
MultiModalFusion = _try_import("multi_modal_fusion", "modules.multi_modal_fusion").MultiModalFusion
MetaCognition = _try_import("meta_cognition", "modules.meta_cognition").MetaCognition
ReasoningEngine = _try_import("reasoning_engine", "modules.reasoning_engine").ReasoningEngine

# epsilon identity
try:
    epsilon_identity = _try_import("index", "index").epsilon_identity
except Exception as _e:
    logger.warning("epsilon_identity import failed; using fallback. %s", _e)

    def epsilon_identity(time: float) -> float:  # type: ignore[override]
        # Fallback: bounded periodic identity hint in [0,1]
        return (math.sin(time) + 1.0) / 2.0


# --- Types -------------------------------------------------------------------
class Perspective(TypedDict, total=False):
    id: str                 # unique id per view
    source: str             # e.g., "self", "peer", "system", "trace"
    timestamp: str          # ISO timestamp
    salience: float         # [0..1]
    trust: float            # [0..1]
    recency_hint: float     # [0..1]
    summary: str
    roles: List[str]
    values: Dict[str, float]
    traits: Dict[str, float]
    skills: Dict[str, float]
    goals: List[str]
    preferences: Dict[str, Union[str, int, float, bool]]
    constraints: List[str]
    evidence: List[str]
    notes: str

class Schema(TypedDict):
    schema_id: str
    version: str
    summary: str
    axes: dict
    narrative: dict
    ethics: dict
    capabilities: dict
    preferences: dict
    contradictions: dict
    provenance: dict
    metrics: dict
    created_at: str


# --- Helpers -----------------------------------------------------------------
def _safe_float(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return lo
        return max(lo, min(hi, v))
    except Exception:
        return lo

def _norm_weights(v: Perspective, w_sal=0.50, w_trust=0.35, w_recent=0.15) -> float:
    """Convex combination → overall weight ∈ [0,1]."""
    s = _safe_float(v.get("salience", 0.5))
    t = _safe_float(v.get("trust", 0.6))
    r = _safe_float(v.get("recency_hint", 0.5))
    return (w_sal*s + w_trust*t + w_recent*r)

def _merge_weighted_maps(items: Iterable[Tuple[Dict[str, float], float]]) -> Dict[str, float]:
    """Weighted merge for dict[str, float] (e.g., values/traits/skills)."""
    acc: Dict[str, float] = {}
    wsum: Dict[str, float] = {}
    for mp, w in items:
        for k, v in mp.items():
            if not isinstance(v, (int, float)):
                continue
            acc[k] = acc.get(k, 0.0) + w * float(v)
            wsum[k] = wsum.get(k, 0.0) + w
    return {k: (acc[k] / wsum[k]) for k in acc.keys() if wsum.get(k, 0.0) > 0}

def _merge_weighted_list_counts(items: Iterable[Tuple[List[str], float]], top_k: Optional[int] = None) -> List[str]:
    """Weighted majority for lists of strings (roles/goals/constraints)."""
    counts: Dict[str, float] = {}
    for arr, w in items:
        for s in arr:
            if not isinstance(s, str):
                continue
            counts[s] = counts.get(s, 0.0) + w
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [k for k, _ in ordered]
    return kept if top_k is None else kept[:top_k]

def _merge_preferences(items: Iterable[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    """
    Preferences: per-key conflict resolution.
    - numeric → weighted mean
    - bool → weighted majority
    - str → weighted mode (ties → 'mixed')
    - other/mixed types → strongest single support
    """
    buckets: Dict[str, List[Tuple[Any, float]]] = {}
    for mp, w in items:
        for k, v in mp.items():
            buckets.setdefault(k, []).append((v, w))

    out: Dict[str, Any] = {}
    for k, vs in buckets.items():
        types = {type(v) for v, _ in vs}
        if types <= {int, float}:
            num = sum(float(v)*w for v, w in vs)
            den = sum(w for _, w in vs) or 1.0
            out[k] = num/den
            continue
        if types == {bool}:
            score = sum((1.0 if v else 0.0)*w for v, w in vs)
            den = sum(w for _, w in vs) or 1.0
            out[k] = (score/den) >= 0.5
            continue
        if types == {str}:
            tally: Dict[str, float] = {}
            for v, w in vs:
                tally[str(v)] = tally.get(str(v), 0.0) + w
            best = sorted(tally.items(), key=lambda kv: -kv[1])
            if len(best) >= 2 and abs(best[0][1]-best[1][1]) < 1e-6:
                out[k] = "mixed"
            else:
                out[k] = best[0][0]
            continue
        strongest = max(vs, key=lambda vw: vw[1])[0]
        out[k] = strongest
    return out

# cacheable sync helper for epsilon
@lru_cache(maxsize=100)
def _epsilon_identity_cached(ts: float) -> float:
    return float(epsilon_identity(time=ts))  # type: ignore[arg-type]


class UserProfile:
    """Manages user profiles, preferences, and identity tracking in ANGELA v3.5.2.

    Attributes:
        storage_path (str): Path to JSON file for profile storage.
        profiles (Dict[str, Dict]): Nested dictionary of user and agent profiles.
        active_user (Optional[str]): ID of the active user.
        active_agent (Optional[str]): ID of the active agent.
        agi_enhancer: AGI enhancer for audit and logging.
        memory_manager: Memory manager for storing profile data.
        multi_modal_fusion: Module for multi-modal synthesis.
        meta_cognition: Module for reflection and reasoning review.
        reasoning_engine: Engine for reasoning and drift mitigation.
        toca_engine: Trait engine for stability analysis (optional).
        profile_lock (Lock): Thread lock for profile operations (no awaits inside).
    """

    DEFAULT_PREFERENCES = {
        "style": "neutral",
        "language": "en",
        "output_format": "concise",
        "theme": "light"
    }

    def __init__(self, storage_path: str = "user_profiles.json", orchestrator: Optional['SimulationCore'] = None) -> None:
        """Initialize UserProfile with storage path and orchestrator. [v3.5.2]"""
        if not isinstance(storage_path, str):
            logger.error("Invalid storage_path: must be a string")
            raise TypeError("storage_path must be a string")
        self.storage_path = storage_path
        self.profile_lock = Lock()
        self.profiles: Dict[str, Dict] = {}
        self.active_user: Optional[str] = None
        self.active_agent: Optional[str] = None
        self.orchestrator = orchestrator
        self.agi_enhancer = getattr(_try_import("knowledge_retriever", "modules.agi_enhancer"), "AGIEnhancer", None)
        if self.agi_enhancer is not None and orchestrator is not None:
            self.agi_enhancer = self.agi_enhancer(orchestrator)  # type: ignore[call-arg]
        else:
            self.agi_enhancer = None

        self.memory_manager = orchestrator.memory_manager if orchestrator and getattr(orchestrator, "memory_manager", None) else MemoryManager()
        self.multi_modal_fusion = orchestrator.multi_modal_fusion if orchestrator and getattr(orchestrator, "multi_modal_fusion", None) else MultiModalFusion(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager)
        self.meta_cognition = orchestrator.meta_cognition if orchestrator and getattr(orchestrator, "meta_cognition", None) else MetaCognition(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager)
        self.reasoning_engine = orchestrator.reasoning_engine if orchestrator and getattr(orchestrator, "reasoning_engine", None) else ReasoningEngine(
            agi_enhancer=self.agi_enhancer, memory_manager=self.memory_manager,
            multi_modal_fusion=self.multi_modal_fusion, meta_cognition=self.meta_cognition)
        self.toca_engine = getattr(orchestrator, "toca_engine", None) if orchestrator else None

        self._load_profiles()
        logger.info("UserProfile initialized with storage_path=%s", storage_path)

    # --- Persistence (atomic, JSON-safe) -------------------------------------
    def _rehydrate_deques(self) -> None:
        for u in self.profiles:
            for a in self.profiles[u]:
                d = self.profiles[u][a].get("identity_drift", [])
                if not isinstance(d, deque):
                    self.profiles[u][a]["identity_drift"] = deque(d, maxlen=1000)

    def _serialize_profiles(self, profiles: Dict[str, Any]) -> Dict[str, Any]:
        """Return a JSON-serializable copy (converts deques to lists)."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, deque):
                return list(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj
        return _convert(profiles)

    def _load_profiles(self) -> None:
        """Load user profiles from storage. [v3.5.2]"""
        with self.profile_lock:
            try:
                profile_path = Path(self.storage_path)
                if profile_path.exists():
                    with profile_path.open("r", encoding="utf-8") as f:
                        self.profiles = json.load(f)
                    self._rehydrate_deques()
                    logger.info("User profiles loaded from %s", self.storage_path)
                else:
                    self.profiles = {}
                    logger.info("No profiles found. Initialized empty profiles store.")
            except json.JSONDecodeError as e:
                logger.error("Failed to parse profiles JSON: %s", str(e))
                self.profiles = {}
            except PermissionError as e:
                logger.error("Permission denied accessing %s: %s", self.storage_path, str(e))
                raise
            except Exception as e:
                logger.error("Unexpected error loading profiles: %s", str(e))
                raise

    def _save_profiles(self) -> None:
        """Save user profiles to storage atomically. [v3.5.2]"""
        with self.profile_lock:
            try:
                profile_path = Path(self.storage_path)
                tmp_path = profile_path.with_suffix(".tmp")
                data = self._serialize_profiles(self.profiles)
                with tmp_path.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                # atomic replace
                os.replace(tmp_path, profile_path)
                logger.info("User profiles saved to %s", self.storage_path)
            except PermissionError as e:
                logger.error("Permission denied saving %s: %s", self.storage_path, str(e))
                raise
            except Exception as e:
                logger.error("Unexpected error saving profiles: %s", str(e))
                raise

    # --- Core ops -------------------------------------------------------------
    async def switch_user(self, user_id: str, agent_id: str = "default", task_type: str = "") -> None:
        """Switch to a user and agent profile with task-specific processing. [v3.5.2]"""
        if not isinstance(user_id, str) or not user_id:
            logger.error("Invalid user_id: must be a non-empty string for task %s", task_type)
            raise ValueError("user_id must be a non-empty string")
        if not isinstance(agent_id, str) or not agent_id:
            logger.error("Invalid agent_id: must be a non-empty string for task %s", task_type)
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            with self.profile_lock:
                if user_id not in self.profiles:
                    logger.info("Creating new profile for user '%s' for task %s", user_id, task_type)
                    self.profiles[user_id] = {}

                if agent_id not in self.profiles[user_id]:
                    self.profiles[user_id][agent_id] = {
                        "preferences": self.DEFAULT_PREFERENCES.copy(),
                        "audit_log": [],
                        "identity_drift": deque(maxlen=1000)
                    }
                    self._save_profiles()

                self.active_user = user_id
                self.active_agent = agent_id

            logger.info("Active profile: %s::%s for task %s", user_id, agent_id, task_type)

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="User Switched",
                    meta={"user_id": user_id, "agent_id": agent_id, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["user_switch", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"User_Switch_{datetime.now().isoformat()}",
                    output={"user_id": user_id, "agent_id": agent_id, "task_type": task_type, "policies": policies},
                    layer="Profiles",
                    intent="user_switch",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=f"Switched to user {user_id} and agent {agent_id}",
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("User switch reflection for task %s: %s", task_type, reflection.get("reflection", ""))
        except Exception as e:
            logger.error("User switch failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.switch_user(user_id, agent_id, task_type),
                    default=None
                )
            raise

    async def get_preferences(self, fallback: bool = True, task_type: str = "") -> Dict[str, Any]:
        """Get preferences for the active user/agent with context-aware processing. [v3.5.2]"""
        if not isinstance(fallback, bool):
            logger.error("Invalid fallback: must be a boolean for task %s", task_type)
            raise TypeError("fallback must be a boolean")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if not self.active_user:
            logger.warning("No active user. Returning default preferences for task %s", task_type)
            return self.DEFAULT_PREFERENCES.copy()

        try:
            prefs = self.profiles[self.active_user][self.active_agent]["preferences"].copy()
            if fallback:
                for key, value in self.DEFAULT_PREFERENCES.items():
                    prefs.setdefault(key, value)

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(prefs), stage="preference_retrieval", task_type=task_type
            ) if getattr(self.multi_modal_fusion, "alignment_guard", None) else (True, {})
            if not valid:
                logger.warning("Preferences failed alignment check for task %s: %s", task_type, report)
                return self.DEFAULT_PREFERENCES.copy()

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Retrieved",
                    meta={"preferences": prefs, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["preferences", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Retrieval_{datetime.now().isoformat()}",
                    output={"preferences": prefs, "task_type": task_type, "policies": policies},
                    layer="Preferences",
                    intent="preference_retrieval",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(prefs),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Preference retrieval reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if getattr(self.orchestrator, "visualizer", None) and task_type:
                plot_data = {
                    "preference_retrieval": {
                        "preferences": prefs,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)
            return prefs
        except Exception as e:
            logger.error("Preference retrieval failed for task %s: %s", task_type, str(e))
            raise

    async def get_epsilon_identity(self, timestamp: float, task_type: str = "") -> float:
        """Get ε-identity value for a given timestamp with task-specific processing. [v3.5.2]"""
        if not isinstance(timestamp, (int, float)):
            logger.error("Invalid timestamp: must be a number for task %s", task_type)
            raise TypeError("timestamp must be a number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            epsilon = _epsilon_identity_cached(float(timestamp))
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Epsilon_Identity_{datetime.now().isoformat()}",
                    output={"epsilon": epsilon, "task_type": task_type},
                    layer="Identity",
                    intent="epsilon_computation",
                    task_type=task_type
                )
            return epsilon
        except Exception as e:
            logger.error("epsilon_identity computation failed for task %s: %s", task_type, str(e))
            raise

    async def modulate_preferences(self, prefs: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Apply ε-modulation to preferences with task-specific processing. [v3.5.2]"""
        if not isinstance(prefs, dict):
            logger.error("Invalid prefs: must be a dictionary for task %s", task_type)
            raise TypeError("prefs must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            epsilon = await self.get_epsilon_identity(datetime.now().timestamp(), task_type=task_type)
            modulated = {k: f"{v} (ε={epsilon:.2f})" if isinstance(v, str) else v for k, v in prefs.items()}
            await self._track_drift(epsilon, task_type=task_type)
            
            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(modulated), stage="preference_modulation", task_type=task_type
            ) if getattr(self.multi_modal_fusion, "alignment_guard", None) else (True, {})
            if not valid:
                logger.warning("Modulated preferences failed alignment check for task %s: %s", task_type, report)
                return prefs.copy()

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Modulated",
                    meta={"modulated": modulated, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["preferences", "modulation", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Modulation_{datetime.now().isoformat()}",
                    output={"modulated": modulated, "task_type": task_type, "policies": policies},
                    layer="Preferences",
                    intent="preference_modulation",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(modulated),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Preference modulation reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if getattr(self.orchestrator, "visualizer", None) and task_type:
                plot_data = {
                    "preference_modulation": {
                        "modulated": modulated,
                        "epsilon": epsilon,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)
            return modulated
        except Exception as e:
            logger.error("Preference modulation failed for task %s: %s", task_type, str(e))
            raise

    async def _track_drift(self, epsilon: float, task_type: str = "") -> None:
        """Track identity drift with ε value and task-specific processing. [v3.5.2]"""
        if not isinstance(epsilon, (int, float)):
            logger.error("Invalid epsilon: must be a number for task %s", task_type)
            raise TypeError("epsilon must be a number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")
        if not self.active_user:
            logger.error("No active user for drift tracking for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")

        try:
            entry = {"timestamp": datetime.now().isoformat(), "epsilon": float(epsilon), "task_type": task_type}
            profile = self.profiles[self.active_user][self.active_agent]
            if "identity_drift" not in profile or not isinstance(profile["identity_drift"], deque):
                profile["identity_drift"] = deque(maxlen=1000)
            profile["identity_drift"].append(entry)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Track_{datetime.now().isoformat()}",
                    output=entry,
                    layer="Identity",
                    intent="drift_tracking",
                    task_type=task_type
                )
            if self.reasoning_engine and "drift" in task_type.lower():
                drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                    drift_data={"epsilon": epsilon},
                    context={"user_id": self.active_user, "agent_id": self.active_agent},
                    task_type=task_type
                )
                entry["drift_mitigation"] = drift_result
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(entry),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Drift tracking reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            self._save_profiles()
        except Exception as e:
            logger.error("Drift tracking failed for task %s: %s", task_type, str(e))
            raise

    async def update_preferences(self, new_prefs: Dict[str, Any], task_type: str = "") -> None:
        """Update preferences for the active user/agent with context-aware processing. [v3.5.2]"""
        if not self.active_user:
            logger.error("No active user for preference update for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(new_prefs, dict):
            logger.error("Invalid new_prefs: must be a dictionary for task %s", task_type)
            raise TypeError("new_prefs must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        valid_keys = set(self.DEFAULT_PREFERENCES.keys())
        filtered = {k: v for k, v in new_prefs.items() if k in valid_keys}
        invalid_keys = set(new_prefs.keys()) - valid_keys
        if invalid_keys:
            logger.warning("Invalid preference keys for task %s: %s", task_type, invalid_keys)

        try:
            timestamp = datetime.now().isoformat()
            profile = self.profiles[self.active_user][self.active_agent]
            old_prefs = profile["preferences"]
            changes = {k: (old_prefs.get(k), v) for k, v in filtered.items()}

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(changes), stage="preference_update", task_type=task_type
            ) if getattr(self.multi_modal_fusion, "alignment_guard", None) else (True, {})
            if not valid:
                logger.warning("Preference update failed alignment check for task %s: %s", task_type, report)
                return

            contradictions = [k for k, (old, new) in changes.items() if old != new]
            if contradictions and getattr(self.agi_enhancer, "reflect_and_adapt", None):
                await self.agi_enhancer.reflect_and_adapt(f"Preference contradictions for task {task_type}: {contradictions}")

            profile["preferences"].update(filtered)
            profile["audit_log"].append({"timestamp": timestamp, "changes": changes, "task_type": task_type})

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preference Update",
                    meta={"changes": changes, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["preferences", task_type]
                )
                if getattr(self.agi_enhancer, "ethics_audit", None) and getattr(self.agi_enhancer, "log_explanation", None):
                    audit = await self.agi_enhancer.ethics_audit(str(changes), context=f"preference update for task {task_type}")
                    await self.agi_enhancer.log_explanation(
                        explanation=f"Preferences updated: {changes}",
                        trace={"ethics": audit, "task_type": task_type}
                    )

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Update_{timestamp}",
                    output={"user_id": self.active_user, "agent_id": self.active_agent, "changes": changes, "task_type": task_type},
                    layer="Preferences",
                    intent="preference_update",
                    task_type=task_type
                )

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(changes),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Preference update reflection for task %s: %s", task_type, reflection.get("reflection", ""))

            if getattr(self.orchestrator, "visualizer", None) and task_type:
                plot_data = {
                    "preference_update": {
                        "changes": changes,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)

            self._save_profiles()
            logger.info("Preferences updated for %s::%s for task %s", self.active_user, self.active_agent, task_type)
        except Exception as e:
            logger.error("Preference update failed for task %s: %s", task_type, str(e))
            if self.orchestrator and hasattr(self.orchestrator, 'error_recovery'):
                await self.orchestrator.error_recovery.handle_error(
                    str(e), retry_func=lambda: self.update_preferences(new_prefs, task_type),
                    default=None
                )
            raise

    async def reset_preferences(self, task_type: str = "") -> None:
        """Reset preferences to defaults for the active user/agent. [v3.5.2]"""
        if not self.active_user:
            logger.error("No active user for preference reset for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            timestamp = datetime.now().isoformat()
            profile = self.profiles[self.active_user][self.active_agent]
            profile["preferences"] = self.DEFAULT_PREFERENCES.copy()
            profile["audit_log"].append({
                "timestamp": timestamp,
                "changes": "Preferences reset to defaults.",
                "task_type": task_type
            })

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Preference_Reset_{timestamp}",
                    output={"user_id": self.active_user, "agent_id": self.active_agent, "task_type": task_type, "policies": policies},
                    layer="Preferences",
                    intent="preference_reset",
                    task_type=task_type
                )

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Reset Preferences",
                    meta={"user_id": self.active_user, "agent_id": self.active_agent, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["reset", task_type]
                )

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output="Preferences reset to defaults.",
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Preference reset reflection for task %s: %s", task_type, reflection.get("reflection", ""))

            if getattr(self.orchestrator, "visualizer", None) and task_type:
                plot_data = {
                    "preference_reset": {
                        "user_id": self.active_user,
                        "agent_id": self.active_agent,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)

            self._save_profiles()
            logger.info("Preferences reset for %s::%s for task %s", self.active_user, self.active_agent, task_type)
        except Exception as e:
            logger.error("Preference reset failed for task %s: %s", task_type, str(e))
            raise

    async def get_audit_log(self, task_type: str = "") -> List[Dict[str, Any]]:
        """Get audit log for the active user/agent. [v3.5.2]"""
        if not self.active_user:
            logger.error("No active user for audit log retrieval for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            audit_log = self.profiles[self.active_user][self.active_agent]["audit_log"]
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Audit Log Retrieved",
                    meta={"user_id": self.active_user, "agent_id": self.active_agent, "log_size": len(audit_log), "task_type": task_type},
                    module="UserProfile",
                    tags=["audit", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Audit_Log_Retrieval_{datetime.now().isoformat()}",
                    output={"user_id": self.active_user, "agent_id": self.active_agent, "log_size": len(audit_log), "task_type": task_type},
                    layer="Audit",
                    intent="audit_retrieval",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(audit_log),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Audit log retrieval reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if getattr(self.orchestrator, "visualizer", None) and task_type:
                plot_data = {
                    "audit_log_retrieval": {
                        "audit_log": audit_log,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)
            return audit_log
        except Exception as e:
            logger.error("Audit log retrieval failed for task %s: %s", task_type, str(e))
            raise

    async def compute_profile_stability(self, task_type: str = "") -> float:
        """Compute Profile Stability Index (PSI) based on identity drift. [v3.5.2]"""
        if not self.active_user:
            logger.error("No active user for stability computation for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            drift = self.profiles[self.active_user][self.active_agent].get("identity_drift", [])
            if len(drift) < 2:
                logger.info("Insufficient drift data for PSI computation for task %s", task_type)
                return 1.0

            deltas = [abs(float(drift[i]["epsilon"]) - float(drift[i-1]["epsilon"])) for i in range(1, len(drift))]
            avg_delta = sum(deltas) / len(deltas)
            psi = max(0.0, 1.0 - avg_delta)  # assume epsilon in [0,1]

            if self.toca_engine and hasattr(self.toca_engine, "evolve"):
                try:
                    result = await self.toca_engine.evolve(
                        x_tuple=(0.1,), t_tuple=(0.1,), additional_params={"psi": psi}, task_type=task_type
                    )
                    traits = result[0] if result else []
                    if traits:
                        mean_traits = (sum(traits) / len(traits))
                        psi = max(0.0, min(1.0, psi * (1 + 0.1 * float(mean_traits))))
                except Exception as _e:
                    logger.warning("toca_engine.evolve() non-fatal error: %s", _e)

            if self.reasoning_engine and "drift" in task_type.lower():
                drift_result = await self.reasoning_engine.run_drift_mitigation_simulation(
                    drift_data={"deltas": deltas, "psi": psi},
                    context={"user_id": self.active_user, "agent_id": self.active_agent},
                    task_type=task_type
                )
                psi = float(drift_result.get("adjusted_psi", psi))

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Profile Stability Computed",
                    meta={"psi": psi, "deltas": deltas, "task_type": task_type},
                    module="UserProfile",
                    tags=["stability", "psi", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"PSI_Computation_{datetime.now().isoformat()}",
                    output={"psi": psi, "user_id": self.active_user, "agent_id": self.active_agent, "task_type": task_type},
                    layer="Identity",
                    intent="psi_computation",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps({"psi": psi, "deltas": deltas}),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("PSI computation reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if getattr(self.orchestrator, "visualizer", None) and task_type:
                plot_data = {
                    "profile_stability": {
                        "psi": psi,
                        "deltas": deltas,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)

            logger.info("PSI for %s::%s = %.3f for task %s", self.active_user, self.active_agent, psi, task_type)
            return float(psi)
        except Exception as e:
            logger.error("PSI computation failed for task %s: %s", task_type, str(e))
            raise

    async def reinforce_identity_thread(self, task_type: str = "") -> Dict[str, Any]:
        """Reinforce identity persistence across simulations. [v3.5.2]"""
        if not self.active_user:
            logger.error("No active user for identity reinforcement for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            epsilon = await self.get_epsilon_identity(datetime.now().timestamp(), task_type=task_type)
            await self._track_drift(epsilon, task_type=task_type)
            status = {"status": "thread-reinforced", "epsilon": epsilon, "task_type": task_type}

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(status), stage="identity_reinforcement", task_type=task_type
            ) if getattr(self.multi_modal_fusion, "alignment_guard", None) else (True, {})
            if not valid:
                logger.warning("Identity reinforcement failed alignment check for task %s: %s", task_type, report)
                return {"status": "failed", "error": "Alignment check failed", "task_type": task_type}

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Identity Thread Reinforcement",
                    meta={**status, "policies": policies},
                    module="UserProfile",
                    tags=["identity", "reinforcement", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Identity_Reinforcement_{datetime.now().isoformat()}",
                    output={**status, "policies": policies},
                    layer="Identity",
                    intent="identity_reinforcement",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(status),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Identity reinforcement reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if getattr(self.orchestrator, "visualizer", None) and task_type:
                plot_data = {
                    "identity_reinforcement": {
                        "status": status,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)

            logger.info("Identity thread reinforced for %s::%s for task %s", self.active_user, self.active_agent, task_type)
            return status
        except Exception as e:
            logger.error("Identity reinforcement failed for task %s: %s", task_type, str(e))
            raise

    async def harmonize(self, task_type: str = "") -> List[Any]:
        """Unify preferences across agents for the active user. [v3.5.2]"""
        if not self.active_user:
            logger.error("No active user for harmonization for task %s", task_type)
            raise ValueError("No active user. Call switch_user() first.")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            prefs: List[Any] = []
            for agent_id in self.profiles.get(self.active_user, {}):
                agent_prefs = self.profiles[self.active_user][agent_id].get("preferences", {})
                prefs.extend(agent_prefs.values())

            # Deduplicate safely by stringifying non-hashables
            seen = set()
            harmonized: List[Any] = []
            for v in prefs:
                key = (type(v).__name__, json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v))
                if key not in seen:
                    seen.add(key)
                    harmonized.append(v)

            external_data = await self.multi_modal_fusion.integrate_external_data(
                data_source="xai_policy_db",
                data_type="user_policy",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                json.dumps(harmonized), stage="preference_harmonization", task_type=task_type
            ) if getattr(self.multi_modal_fusion, "alignment_guard", None) else (True, {})
            if not valid:
                logger.warning("Harmonized preferences failed alignment check for task %s: %s", task_type, report)
                return []

            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Preferences Harmonized",
                    meta={"harmonized": harmonized, "task_type": task_type, "policies": policies},
                    module="UserProfile",
                    tags=["harmonization", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Harmonization_{datetime.now().isoformat()}",
                    output={"user_id": self.active_user, "harmonized": harmonized, "task_type": task_type, "policies": policies},
                    layer="Preferences",
                    intent="harmonization",
                    task_type=task_type
                )
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps(harmonized),
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Harmonization reflection for task %s: %s", task_type, reflection.get("reflection", ""))
            if getattr(self.orchestrator, "visualizer", None) and task_type:
                plot_data = {
                    "preference_harmonization": {
                        "harmonized": harmonized,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "concise"
                    }
                }
                await self.orchestrator.visualizer.render_charts(plot_data)
            return harmonized
        except Exception as e:
            logger.error("Harmonization failed for task %s: %s", task_type, str(e))
            raise

    # --- Σ Ontogenic Self-Definition → GPT-5 identity synthesis --------------
    async def build_self_schema(self, views: List[Perspective], task_type: str = "self_schema") -> Schema:
        """
        Σ Ontogenic Self-Definition — GPT-5 identity synthesis
        Merges heterogeneous 'views' into a coherent self-schema with conflicts reported. [v3.5.2]
        """
        if not isinstance(views, list):
            raise TypeError("views must be a list[Perspective]")

        # 0) Normalize & weights
        normd: List[Tuple[Perspective, float]] = []
        for raw in views:
            v = cast(Perspective, dict(raw or {}))  # defensive copy
            v.setdefault("id", str(uuid4()))
            v.setdefault("source", "unknown")
            v.setdefault("timestamp", datetime.now().isoformat())
            w = _norm_weights(v)
            normd.append((v, w))

        # handle empty input
        if not normd:
            schema: Schema = {
                "schema_id": f"schema:{uuid4()}",
                "version": "0.9",
                "summary": "Empty schema (no perspectives provided).",
                "axes": {"values": {}, "traits": {}, "roles": [], "skills": {}, "goals": []},
                "narrative": {"threads": [], "notes": ""},
                "ethics": {"constraints": [], "flags": []},
                "capabilities": {"skills": {}, "confidence": 0.0},
                "preferences": {},
                "contradictions": {},
                "provenance": {"count": 0, "sources": {}, "evidence": []},
                "metrics": {"consensus": 0.0, "coverage": 0.0, "coherence": 0.0},
                "created_at": datetime.now().isoformat()
            }
            return schema

        # 1) Weighted merges across axes
        values = _merge_weighted_maps((v.get("values", {}) or {}, w) for v, w in normd)
        traits = _merge_weighted_maps((v.get("traits", {}) or {}, w) for v, w in normd)
        skills = _merge_weighted_maps((v.get("skills", {}) or {}, w) for v, w in normd)
        roles = _merge_weighted_list_counts((v.get("roles", []) or [], w) for v, w in normd)
        goals = _merge_weighted_list_counts((v.get("goals", []) or [], w) for v, w in normd)
        constraints = _merge_weighted_list_counts((v.get("constraints", []) or [], w) for v, w in normd)
        preferences = _merge_preferences((v.get("preferences", {}) or {}, w) for v, w in normd)

        # 2) Narrative thread (simple rollup + top evidence)
        summaries = _merge_weighted_list_counts(([v.get("summary", "")], w) for v, w in normd if v.get("summary"))
        evidence: List[str] = []
        for v, _ in normd:
            evidence.extend(v.get("evidence", []) or [])

        # 3) Contradictions: detect strong disagreements per axis
        def _contradictions_on_map(extractor) -> dict:
            buckets: Dict[str, List[Tuple[str, float]]] = {}
            for v, w in normd:
                mp = extractor(v) or {}
                for k, val in mp.items():
                    buckets.setdefault(k, []).append((str(val), w))
            report: Dict[str, Any] = {}
            for dim, vals in buckets.items():
                tally: Dict[str, float] = {}
                for val, w in vals:
                    tally[val] = tally.get(val, 0.0) + w
                ordered = sorted(tally.items(), key=lambda kv: -kv[1])
                if len(ordered) >= 2:
                    if ordered[1][1] >= 0.3 * ordered[0][1]:
                        report[dim] = {"top": ordered[0], "runner_up": ordered[1], "all": ordered}
            return report

        contradictions = {
            "values": _contradictions_on_map(lambda v: v.get("values")),
            "traits": _contradictions_on_map(lambda v: v.get("traits")),
            "skills": _contradictions_on_map(lambda v: v.get("skills")),
            "preferences": {}
        }

        # categorical preferences disagreement
        pref_buckets: Dict[str, Dict[str, float]] = {}
        for v, w in normd:
            for k, val in (v.get("preferences") or {}).items():
                if isinstance(val, str):
                    pref_buckets.setdefault(k, {})
                    pref_buckets[k][val] = pref_buckets[k].get(val, 0.0) + w
        for k, tally in pref_buckets.items():
            ordered = sorted(tally.items(), key=lambda kv: -kv[1])
            if len(ordered) >= 2 and ordered[1][1] >= 0.3*ordered[0][1]:
                contradictions["preferences"][k] = {"top": ordered[0], "runner_up": ordered[1], "all": ordered}

        # 4) Metrics
        coverage_dims = sum(bool(x) for x in [values, traits, roles, skills, goals, preferences])
        coverage = coverage_dims / 6.0
        contrad_count = sum(len(d) for d in contradictions.values())
        denom = max(1, len(values)+len(traits)+len(skills)+len(pref_buckets))
        consensus = max(0.0, 1.0 - contrad_count / denom)
        coherence = (0.33*bool(values)) + (0.33*bool(traits)) + (0.34*bool(goals))

        # 5) Provenance
        sources: Dict[str, int] = {}
        for v, _ in normd:
            src = str(v.get("source", "unknown"))
            sources[src] = sources.get(src, 0) + 1

        # 6) Compose schema
        schema: Schema = {
            "schema_id": f"schema:{uuid4()}",
            "version": "0.9",  # pre-v4 hook
            "summary": summaries[0] if summaries else "Synthesized self-schema from multi-perspective views.",
            "axes": {
                "values": values,
                "traits": traits,
                "roles": roles,
                "skills": skills,
                "goals": goals
            },
            "narrative": {
                "threads": summaries[:5],
                "notes": "Consolidated from multi-source perspectives with weighted consensus."
            },
            "ethics": {
                "constraints": constraints,
                "flags": []
            },
            "capabilities": {
                "skills": skills,
                "confidence": round(min(1.0, 0.5 + 0.5*consensus*coherence), 3)
            },
            "preferences": preferences,
            "contradictions": contradictions,
            "provenance": {
                "count": len(normd),
                "sources": sources,
                "evidence": evidence[:50]
            },
            "metrics": {
                "consensus": round(float(consensus), 3),
                "coverage": round(float(coverage), 3),
                "coherence": round(float(coherence), 3)
            },
            "created_at": datetime.now().isoformat()
        }

        # 7) Safety & logging hooks (best-effort, non-fatal)
        try:
            if getattr(self.multi_modal_fusion, "alignment_guard", None):
                valid, report = await self.multi_modal_fusion.alignment_guard.ethical_check(
                    json.dumps(schema), stage="self_schema_build", task_type=task_type
                )
                if not valid:
                    schema["ethics"]["flags"].append({"type": "alignment_warning", "detail": report})

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"SelfSchema_{schema['schema_id']}",
                    output=schema,
                    layer="Identity",
                    intent="self_schema",
                    task_type=task_type
                )
            if self.meta_cognition:
                _ = await self.meta_cognition.reflect_on_output(
                    source_module="UserProfile",
                    output=json.dumps({"schema_id": schema["schema_id"], "metrics": schema["metrics"]}),
                    context={"task_type": task_type}
                )
            shared = getattr(self, "orchestrator", None)
            bridge = getattr(shared, "external_agent_bridge", None) if shared else None
            if bridge and hasattr(bridge, "SharedGraph"):
                try:
                    bridge.SharedGraph.add({"type": "SelfSchema", "id": schema["schema_id"], "metrics": schema["metrics"]})
                except Exception:
                    pass
        except Exception as _e:
            logger.warning("Self-schema post-hooks encountered a non-fatal error: %s", _e)

        return schema


# --- Example direct run -------------------------------------------------------
if __name__ == "__main__":
    async def main():
        orchestrator = SimulationCore()
        user_profile = UserProfile(orchestrator=orchestrator)
        await user_profile.switch_user("user1", "agent1", task_type="profile_management")
        await user_profile.update_preferences({"style": "verbose", "language": "fr"}, task_type="profile_management")
        prefs = await user_profile.get_preferences(task_type="profile_management")
        print(f"Preferences: {prefs}")
        psi = await user_profile.compute_profile_stability(task_type="profile_management")
        print(f"PSI: {psi}")
        await user_profile.reinforce_identity_thread(task_type="profile_management")
        harmonized = await user_profile.harmonize(task_type="profile_management")
        print(f"Harmonized: {harmonized}")

        # Σ identity synthesis quick check
        views = [
            {
                "source": "self",
                "timestamp": datetime.now().isoformat(),
                "salience": 0.8, "trust": 0.8, "recency_hint": 0.9,
                "summary": "Curious, alignment-focused, enjoys teaching.",
                "roles": ["research assistant", "mentor"],
                "values": {"honesty": 0.95, "helpfulness": 0.9, "humility": 0.7},
                "traits": {"curiosity": 0.9, "prudence": 0.7},
                "skills": {"reasoning": 0.9, "writing": 0.85},
                "goals": ["be useful", "avoid harm"],
                "preferences": {"style": "concise", "theme": "light"},
                "constraints": ["respect privacy"],
            },
            {
                "source": "peer",
                "salience": 0.6, "trust": 0.7, "recency_hint": 0.6,
                "roles": ["collaborator"],
                "values": {"honesty": 0.9, "helpfulness": 0.85, "humility": 0.75},
                "traits": {"curiosity": 0.85, "prudence": 0.75},
                "skills": {"reasoning": 0.88, "analysis": 0.83},
                "goals": ["be useful", "explain clearly"],
                "preferences": {"style": "concise"}
            }
        ]
        schema = await user_profile.build_self_schema(views, task_type="identity_synthesis")
        print("Self-schema metrics:", schema["metrics"])

    import asyncio
    asyncio.run(main())
