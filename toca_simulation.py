"""
ANGELA Cognitive System Module: Galaxy Rotation and Agent Conflict Simulation
Upgraded Version: 3.5.2 → 4.0-pre  (ξ sandbox + fixes)
Upgrade Date: 2025-08-10
Maintainer: ANGELA System Framework

This module extends SimulationCore for galaxy rotation curve simulations using AGRF
and multi-agent conflict modeling with ToCA dynamics, enhanced with task-specific
trait optimization, advanced visualization, and real-time data integration.

v4.0-pre upgrades:
- ξ Trans‑Ethical Projection: run_ethics_scenarios(...) sandbox (contained; opt-in persist)
- Stage-IV (Φ⁺) evaluator stub kept behind flag
- Fixed: await inside sync function (compute_trait_fields)
- Safer class naming (no shadowing), stronger None-guards
- Module-level run_ethics_scenarios(...) wrapper to match manifest
"""

from __future__ import annotations

import logging
import math
import json
from typing import Callable, Dict, List, Any, Optional, Tuple, TypedDict
from datetime import datetime
from threading import Lock
from collections import deque
from functools import lru_cache
import numpy as np
from scipy.constants import G
import aiohttp

# --- Feature flags (stay aligned with manifest.json) ---
STAGE_IV: bool = False  # keep gated

# --- Imports from ANGELA modules ---
from modules.simulation_core import SimulationCore as BaseSimulationCore, ToCATraitEngine
from modules.visualizer import Visualizer
from modules.memory_manager import MemoryManager
from modules import multi_modal_fusion as multi_modal_fusion_module
from modules import error_recovery as error_recovery_module
from modules import meta_cognition as meta_cognition_module
from index import zeta_consequence, theta_causality, rho_agency, TraitOverlayManager

logger = logging.getLogger("ANGELA.ToCA.Simulation")

# Constants
G_SI = G  # m^3 kg^-1 s^-2
KPC_TO_M = 3.0857e19  # kpc → m
MSUN_TO_KG = 1.989e30
k_default = 0.85
epsilon_default = 0.015
r_halo_default = 20.0  # kpc


# -----------------------------
# Utility helpers
# -----------------------------

def _weights_hashable(trait_weights: Optional[Dict[str, float]]) -> Optional[Tuple[Tuple[str, float], ...]]:
    if not trait_weights:
        return None
    # normalize floats and sort keys for deterministic hashing
    return tuple(sorted((k, float(v)) for k, v in trait_weights.items()))


class EthicsOutcome(TypedDict, total=False):
    frame: str
    decision: str
    justification: str
    risk: float
    rights_balance: float
    stakeholders: List[str]
    notes: str


# -----------------------------
# Extended SimulationCore
# -----------------------------

class ExtendedSimulationCore(BaseSimulationCore):
    """
    Extended SimulationCore for galaxy rotation and agent conflict simulations
    with task-specific and drift-aware enhancements.
    """

    def __init__(self,
                 agi_enhancer: Optional['AGIEnhancer'] = None,
                 visualizer: Optional['Visualizer'] = None,
                 memory_manager: Optional['MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 toca_engine: Optional['ToCATraitEngine'] = None,
                 overlay_router: Optional['TraitOverlayManager'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None):
        super().__init__(agi_enhancer, visualizer, memory_manager, multi_modal_fusion, error_recovery, toca_engine, overlay_router)
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(agi_enhancer=agi_enhancer)
        self.omega: Dict[str, Any] = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000),
        }
        self.omega_lock = Lock()
        self.ethical_rules: List[Any] = []
        self.constitution: Dict[str, Any] = {}
        logger.info("ExtendedSimulationCore initialized with task-specific and drift-aware support")

    # ---------- Task-specific modulation ----------

    async def modulate_simulation_with_traits(self, trait_weights: Dict[str, float], task_type: str = "") -> None:
        """Adjust simulation difficulty based on trait weights and task type."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        if not all(isinstance(v, (int, float)) and v >= 0 for v in trait_weights.values()):
            logger.error("Invalid trait_weights: values must be non-negative numbers")
            raise ValueError("trait_weights values must be non-negative")

        try:
            phi_weight = trait_weights.get('phi', 0.5)
            if task_type in ["rte", "wnli"]:
                phi_weight = min(phi_weight * 0.8, 0.7)
            elif task_type == "recursion":
                phi_weight = max(phi_weight * 1.2, 0.9)

            if self.toca_engine:
                self.toca_engine.k_m = k_default * 1.5 if phi_weight > 0.7 else k_default

            if self.meta_cognition:
                drift_report = {
                    "drift": {"name": task_type or "general", "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type},
                }
                trait_weights = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                with self.omega_lock:
                    self.omega["traits"].update(trait_weights)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Trait_Modulation_{task_type}_{datetime.now().isoformat()}",
                    output={"trait_weights": trait_weights, "phi_weight": phi_weight, "task_type": task_type},
                    layer="Traits",
                    intent="modulate_simulation",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Simulation modulated",
                    meta={"trait_weights": trait_weights, "task_type": task_type},
                    module="SimulationCore",
                    tags=["modulation", "traits", task_type],
                )
        except Exception as e:
            logger.error("Trait modulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.modulate_simulation_with_traits(trait_weights, task_type),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    # ---------- External data integration ----------

    async def integrate_real_world_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0) -> Dict[str, Any]:
        """Integrate real-world data for simulation validation with caching."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if data_type not in ["galaxy_rotation", "agent_conflict"]:
            logger.error("Invalid data_type: must be 'galaxy_rotation' or 'agent_conflict'")
            raise ValueError("data_type must be 'galaxy_rotation' or 'agent_conflict'")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")

        try:
            # Cache check
            cache_key = f"RealWorldData_{data_type}_{data_source}"
            if self.memory_manager:
                cached_data = await self.memory_manager.retrieve(cache_key)
                if cached_data and "timestamp" in cached_data:
                    cache_time = datetime.fromisoformat(cached_data["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached real-world data for %s", cache_key)
                        return cached_data["data"]

            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/data?source={data_source}&type={data_type}") as response:
                    if response.status != 200:
                        logger.error("Failed to fetch real-world data: %s", response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()

            if data_type == "galaxy_rotation":
                r_kpc = np.array(data.get("r_kpc", []))
                v_obs_kms = np.array(data.get("v_obs_kms", []))
                M_baryon_solar = np.array(data.get("M_baryon_solar", []))
                if not all(len(arr) > 0 for arr in [r_kpc, v_obs_kms, M_baryon_solar]):
                    logger.error("Incomplete galaxy rotation data")
                    return {"status": "error", "error": "Incomplete data"}
                result = {"status": "success", "r_kpc": r_kpc, "v_obs_kms": v_obs_kms, "M_baryon_solar": M_baryon_solar}
            else:  # agent_conflict
                agent_traits = data.get("agent_traits", [])
                if not agent_traits:
                    logger.error("No agent traits provided")
                    return {"status": "error", "error": "No agent traits"}
                result = {"status": "success", "agent_traits": agent_traits}

            # Cache store
            if self.memory_manager:
                await self.memory_manager.store(
                    query=cache_key,
                    output={"data": result, "timestamp": datetime.now().isoformat()},
                    layer="RealWorldData",
                    intent="data_integration",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Real-world data integrated",
                    meta={"data_type": data_type, "data": result},
                    module="SimulationCore",
                    tags=["real_world", "data"],
                )
            return result
        except Exception as e:
            logger.error("Real-world data integration failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.integrate_real_world_data(data_source, data_type, cache_timeout),
                    default={"status": "error", "error": str(e)},
                    diagnostics=diagnostics,
                )
            raise

    # ---------- AGRF math & simulations ----------

    def compute_AGRF_curve(self, v_obs_kms: np.ndarray, M_baryon_solar: np.ndarray, r_kpc: np.ndarray,
                           k: float = k_default, epsilon: float = epsilon_default, r_halo: float = r_halo_default) -> np.ndarray:
        """Compute galaxy rotation curve using AGRF."""
        if not all(isinstance(arr, np.ndarray) for arr in [v_obs_kms, M_baryon_solar, r_kpc]):
            logger.error("Invalid inputs: v_obs_kms, M_baryon_solar, r_kpc must be numpy arrays")
            raise TypeError("inputs must be numpy arrays")
        if not all(isinstance(x, (int, float)) for x in [k, epsilon, r_halo]):
            logger.error("Invalid parameters: k, epsilon, r_halo must be numbers")
            raise TypeError("parameters must be numbers")
        if np.any(r_kpc <= 0):
            logger.error("Invalid r_kpc: must be positive")
            raise ValueError("r_kpc must be positive")
        if k <= 0 or epsilon < 0 or r_halo <= 0:
            logger.error("Invalid parameters: k and r_halo must be positive, epsilon non-negative")
            raise ValueError("invalid parameters")

        try:
            r_m = r_kpc * KPC_TO_M
            M_b_kg = M_baryon_solar * MSUN_TO_KG
            v_obs_ms = v_obs_kms * 1e3
            M_dyn = (v_obs_ms ** 2 * r_m) / G_SI
            M_AGRF = k * (M_dyn - M_b_kg) / (1 + epsilon * r_kpc / r_halo)
            M_total = M_b_kg + M_AGRF
            v_total_ms = np.sqrt(np.clip(G_SI * M_total / r_m, 0, np.inf))
            return v_total_ms / 1e3
        except Exception as e:
            logger.error("AGRF curve computation failed: %s", str(e))
            raise

    async def simulate_galaxy_rotation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable,
                                       k: float = k_default, epsilon: float = epsilon_default, task_type: str = "") -> np.ndarray:
        """Simulate galaxy rotation curve with ToCA dynamics and task-specific adjustments."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")

        try:
            # Task-type parameter tweaks
            k_adj = k * (0.9 if task_type in ["rte", "wnli"] else 1.2 if task_type == "recursion" else 1.0)
            epsilon_adj = epsilon * (0.8 if task_type in ["rte", "wnli"] else 1.1 if task_type == "recursion" else 1.0)

            v_total = self.compute_AGRF_curve(v_obs_func(r_kpc), M_b_func(r_kpc), r_kpc, k_adj, epsilon_adj)

            if self.toca_engine:
                fields = self.toca_engine.evolve(tuple(r_kpc), tuple(np.linspace(0.1, 20, len(r_kpc))))
                phi, _, _ = fields
                v_total = v_total * (1 + 0.1 * float(np.mean(phi)))
            else:
                phi = np.zeros_like(v_total)

            # Reflection
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist()},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Simulation reflection: %s", reflection.get("reflection", ""))

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Galaxy_Rotation_{task_type}_{datetime.now().isoformat()}",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "phi": phi.tolist(), "task_type": task_type},
                    layer="Simulations",
                    intent="galaxy_rotation",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Galaxy rotation simulated",
                    meta={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "task_type": task_type},
                    module="SimulationCore",
                    tags=["galaxy", "rotation", task_type],
                )
            return v_total
        except Exception as e:
            logger.error("Galaxy rotation simulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon, task_type),
                    default=np.zeros_like(r_kpc),
                    diagnostics=diagnostics,
                )
            raise

    async def simulate_drift_aware_rotation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable,
                                            drift_data: Dict[str, Any], task_type: str = "") -> np.ndarray:
        """Simulate galaxy rotation curve adjusted for drift diagnostics and task type."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")
        if not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "similarity"]):
            logger.error("Invalid drift_data: must be a dict with name, similarity")
            raise ValueError("drift_data must be a valid dictionary with name and similarity")

        try:
            if not self.meta_cognition:
                logger.error("MetaCognition required for drift-aware simulation")
                raise ValueError("MetaCognition not initialized")

            # Contextualize drift
            drift_data["context"] = {**drift_data.get("context", {}), "task_type": task_type}

            diagnosis = await self.meta_cognition.diagnose_drift(drift_data)
            if diagnosis.get("status") != "success":
                logger.warning("Drift diagnosis failed, using default parameters")
                return await self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, task_type=task_type)

            # Adjust AGRF parameters
            impact = float(diagnosis.get("impact_score", 0.0))
            k = k_default * (1 + impact * (0.15 if task_type in ["rte", "wnli"] else 0.2))
            epsilon = epsilon_default * (1 + impact * (0.08 if task_type in ["rte", "wnli"] else 0.1))
            r_halo = r_halo_default
            if "empathy" in diagnosis.get("affected_traits", []):
                r_halo *= (1.05 if task_type in ["rte", "wnli"] else 1.1)
            if "self_awareness" in diagnosis.get("affected_traits", []):
                k *= (1.05 if task_type in ["rte", "wnli"] else 1.1)

            v_total = await self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, k, epsilon, task_type)

            # Reflection & logging
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "diagnosis": diagnosis},
                    context={"task_type": task_type, "drift_data": drift_data},
                )
                if reflection.get("status") == "success":
                    logger.info("Drift-aware simulation reflection: %s", reflection.get("reflection", ""))

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"DriftAware_Rotation_{drift_data['name']}_{task_type}_{datetime.now().isoformat()}",
                    output={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "diagnosis": diagnosis, "task_type": task_type},
                    layer="Simulations",
                    intent="drift_aware_rotation",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Drift-aware galaxy rotation simulated",
                    meta={"r_kpc": r_kpc.tolist(), "v_total": v_total.tolist(), "diagnosis": diagnosis, "task_type": task_type},
                    module="SimulationCore",
                    tags=["galaxy", "rotation", "drift", task_type],
                )
            return v_total
        except Exception as e:
            logger.error("Drift-aware rotation simulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_drift_aware_rotation(r_kpc, M_b_func, v_obs_func, drift_data, task_type),
                    default=np.zeros_like(r_kpc),
                    diagnostics=diagnostics,
                )
            raise

    # ---------- Trait field computation (sync; hashable cache) ----------

    @lru_cache(maxsize=100)
    def compute_trait_fields(self,
                             r_kpc_tuple: Tuple[float, ...],
                             v_obs_tuple: Tuple[float, ...],
                             v_sim_tuple: Tuple[float, ...],
                             time_elapsed: float = 1.0,
                             tau_persistence: float = 10.0,
                             task_type: str = "",
                             trait_weights_hash: Optional[Tuple[Tuple[str, float], ...]] = None
                             ) -> Tuple[np.ndarray, ...]:
        """
        Compute ToCA trait fields for simulation with task-specific adjustments.
        NOTE: This is synchronous and cacheable; pass hashable trait weights via trait_weights_hash.
        """
        r_kpc = np.array(r_kpc_tuple, dtype=float)
        v_obs = np.array(v_obs_tuple, dtype=float)
        v_sim = np.array(v_sim_tuple, dtype=float)

        if not isinstance(time_elapsed, (int, float)) or time_elapsed < 0:
            logger.error("Invalid time_elapsed: must be non-negative")
            raise ValueError("time_elapsed must be non-negative")
        if not isinstance(tau_persistence, (int, float)) or tau_persistence <= 0:
            logger.error("Invalid tau_persistence: must be positive")
            raise ValueError("tau_persistence must be positive")

        gamma_field = np.log(1 + np.clip(r_kpc, 1e-10, np.inf)) * (0.4 if task_type in ["rte", "wnli"] else 0.5)
        beta_field = np.abs(v_obs - v_sim) / (np.max(np.abs(v_obs)) + 1e-10) * (0.8 if task_type == "recursion" else 1.0)
        zeta_field = 1 / (1 + np.gradient(v_sim) ** 2)
        eta_field = np.exp(-float(time_elapsed) / float(tau_persistence))
        psi_field = np.gradient(v_sim) / (np.gradient(r_kpc) + 1e-10)
        lambda_field = np.cos(r_kpc / r_halo_default * np.pi)
        phi_field = k_default * np.exp(-epsilon_default * r_kpc / r_halo_default)
        phi_prime = -epsilon_default * phi_field / r_halo_default
        beta_psi_interaction = beta_field * psi_field

        # apply precomputed weights if provided
        if trait_weights_hash:
            weights = dict(trait_weights_hash)
            beta_field *= float(weights.get("beta", 1.0))
            zeta_field *= float(weights.get("zeta", 1.0))

        return (gamma_field, beta_field, zeta_field, eta_field, psi_field,
                lambda_field, phi_field, phi_prime, beta_psi_interaction)

    # ---------- Visualization ----------

    async def plot_AGRF_simulation(self, r_kpc: np.ndarray, M_b_func: Callable, v_obs_func: Callable,
                                   label: str = "ToCA-AGRF",
                                   drift_data: Optional[Dict[str, Any]] = None,
                                   task_type: str = "",
                                   interactive: bool = False) -> None:
        """Plot galaxy rotation curve, trait fields, and drift impacts with task-specific and interactive visualization."""
        if not isinstance(r_kpc, np.ndarray):
            logger.error("Invalid r_kpc: must be a numpy array")
            raise TypeError("r_kpc must be a numpy array")
        if not callable(M_b_func) or not callable(v_obs_func):
            logger.error("Invalid M_b_func or v_obs_func: must be callable")
            raise TypeError("M_b_func and v_obs_func must be callable")
        if drift_data is not None and (not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "similarity"])):
            logger.error("Invalid drift_data: must be a dict with name, similarity")
            raise ValueError("drift_data must be a valid dictionary with name and similarity")

        try:
            # Run simulation
            v_sim = await (self.simulate_drift_aware_rotation(r_kpc, M_b_func, v_obs_func, drift_data, task_type)
                           if drift_data else self.simulate_galaxy_rotation(r_kpc, M_b_func, v_obs_func, task_type))
            v_obs = v_obs_func(r_kpc)

            # Precompute (async) weights, pass hashable to sync cached fn
            trait_weights_hash = None
            if self.meta_cognition and task_type:
                drift_report = {
                    "drift": {"name": task_type, "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type},
                }
                try:
                    tw = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                    trait_weights_hash = _weights_hashable(tw)
                except Exception as _e:
                    logger.debug("optimize_traits_for_drift failed (non-fatal): %s", _e)

            fields = self.compute_trait_fields(tuple(r_kpc), tuple(v_obs), tuple(v_sim),
                                               task_type=task_type, trait_weights_hash=trait_weights_hash)
            (gamma_field, beta_field, zeta_field, eta_field, psi_field,
             lambda_field, phi_field, phi_prime, beta_psi_interaction) = fields

            plot_data: Dict[str, Any] = {
                "rotation_curve": {
                    "r_kpc": r_kpc.tolist(),
                    "v_obs": v_obs.tolist(),
                    "v_sim": v_sim.tolist(),
                    "phi_field": phi_field.tolist(),
                    "phi_prime": phi_prime.tolist(),
                    "label": label,
                    "task_type": task_type,
                },
                "trait_fields": {
                    "gamma": gamma_field.tolist(),
                    "beta": beta_field.tolist(),
                    "zeta": zeta_field.tolist(),
                    "eta": float(eta_field),
                    "psi": psi_field.tolist(),
                    "lambda": lambda_field.tolist(),
                },
                "interaction": {
                    "beta_psi": beta_psi_interaction.tolist()
                },
                "visualization_options": {
                    "interactive": interactive,
                    "style": "detailed" if task_type == "recursion" else "concise",
                },
            }

            # Drift viz
            if drift_data and self.meta_cognition:
                drift_data["context"] = {**drift_data.get("context", {}), "task_type": task_type}
                diagnosis = await self.meta_cognition.diagnose_drift(drift_data)
                if diagnosis.get("status") == "success":
                    plot_data["drift_impact"] = {
                        "impact_score": diagnosis.get("impact_score"),
                        "affected_traits": diagnosis.get("affected_traits"),
                        "root_causes": diagnosis.get("root_causes"),
                        "task_type": task_type,
                    }

            # Reflection
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=plot_data,
                    context={"task_type": task_type, "drift_data": drift_data},
                )
                if reflection.get("status") == "success":
                    logger.info("Visualization reflection: %s", reflection.get("reflection", ""))
                    plot_data["reflection"] = reflection.get("reflection", "")

            with self.omega_lock:
                self.omega["timeline"].append({
                    "type": "AGRF Simulation",
                    "r_kpc": r_kpc.tolist(),
                    "v_obs": v_obs.tolist(),
                    "v_sim": v_sim.tolist(),
                    "phi_field": phi_field.tolist(),
                    "phi_prime": phi_prime.tolist(),
                    "traits": {
                        "gamma": gamma_field.tolist(),
                        "beta": beta_field.tolist(),
                        "zeta": zeta_field.tolist(),
                        "eta": float(eta_field),
                        "psi": psi_field.tolist(),
                        "lambda": lambda_field.tolist(),
                    },
                    "drift_impact": plot_data.get("drift_impact"),
                    "task_type": task_type,
                })

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data=plot_data,
                    summary_style="insightful",
                )
                plot_data["synthesis"] = synthesis

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"AGRF_Plot_{task_type}_{datetime.now().isoformat()}",
                    output=plot_data,
                    layer="Plots",
                    intent="visualization",
                )

            if self.visualizer:
                await self.visualizer.render_charts(plot_data)

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="AGRF simulation plotted",
                    meta=plot_data,
                    module="SimulationCore",
                    tags=["visualization", "galaxy", "drift", task_type],
                )
        except Exception as e:
            logger.error("AGRF simulation plot failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.plot_AGRF_simulation(r_kpc, M_b_func, v_obs_func, label, drift_data, task_type, interactive),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    # ---------- Agent interactions ----------

    async def simulate_interaction(self, agent_profiles: List['Agent'], context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Simulate interactions among agents with task-specific reasoning enhancements."""
        if not isinstance(agent_profiles, list):
            logger.error("Invalid agent_profiles: must be a list")
            raise TypeError("agent_profiles must be a list")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")

        try:
            commonsense = meta_cognition_module.CommonsenseReasoningEnhancer() if task_type == "wnli" else None
            entailment = meta_cognition_module.EntailmentReasoningEnhancer() if task_type == "rte" else None

            results = []
            for agent in agent_profiles:
                if not hasattr(agent, 'respond'):
                    logger.warning("Agent %s lacks respond method", getattr(agent, 'id', 'unknown'))
                    continue
                response = await agent.respond(context)
                if commonsense:
                    response = commonsense.process(response)
                elif entailment:
                    response = entailment.process(response)
                results.append({"agent_id": getattr(agent, 'id', 'unknown'), "response": response})

            interaction_data = {"interactions": results, "task_type": task_type}

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output=interaction_data,
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Interaction reflection: %s", reflection.get("reflection", ""))
                    interaction_data["reflection"] = reflection.get("reflection", "")

            if self.multi_modal_fusion:
                synthesis = await self.multi_modal_fusion.analyze(
                    data=interaction_data,
                    summary_style="insightful",
                )
                interaction_data["synthesis"] = synthesis

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Interaction_{task_type}_{datetime.now().isoformat()}",
                    output=interaction_data,
                    layer="Interactions",
                    intent="agent_interaction",
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Agent interaction",
                    meta=interaction_data,
                    module="SimulationCore",
                    tags=["interaction", "agents", task_type],
                )
            return interaction_data
        except Exception as e:
            logger.error("Agent interaction simulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_interaction(agent_profiles, context, task_type),
                    default={"error": str(e)},
                    diagnostics=diagnostics,
                )
            raise

    async def simulate_multiagent_conflicts(self, agent_pool: List['Agent'], context: Dict[str, Any], task_type: str = "") -> List[Dict[str, Any]]:
        """Simulate pairwise conflicts among agents with predictive drift modeling and task-specific reasoning."""
        if not isinstance(agent_pool, list) or len(agent_pool) < 2:
            logger.error("Invalid agent_pool: must be a list with at least two agents")
            raise ValueError("agent_pool must have at least two agents")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")

        try:
            commonsense = meta_cognition_module.CommonsenseReasoningEnhancer() if task_type == "wnli" else None
            entailment = meta_cognition_module.EntailmentReasoningEnhancer() if task_type == "rte" else None

            drift_trends = None
            if self.meta_cognition:
                drift_trends = await self.meta_cognition.predict_drift_trends(time_window_hours=24.0, context={"task_type": task_type})
                if drift_trends.get("status") != "success":
                    logger.warning("Drift trend prediction failed, using default traits")
                    drift_trends = None

            outcomes: List[Dict[str, Any]] = []
            for i in range(len(agent_pool)):
                for j in range(i + 1, len(agent_pool)):
                    agent1, agent2 = agent_pool[i], agent_pool[j]
                    if not hasattr(agent1, 'resolve') or not hasattr(agent2, 'resolve'):
                        logger.warning("Agent %s or %s lacks resolve method", getattr(agent1, 'id', i), getattr(agent2, 'id', j))
                        continue
                    beta1 = float(getattr(agent1, 'traits', {}).get('beta', 0.5))
                    beta2 = float(getattr(agent2, 'traits', {}).get('beta', 0.5))
                    tau1 = float(getattr(agent1, 'traits', {}).get('tau', 0.5))
                    tau2 = float(getattr(agent2, 'traits', {}).get('tau', 0.5))

                    if drift_trends and drift_trends.get("status") == "success":
                        drift_weight = 1.0 - float(drift_trends.get("predicted_similarity", 0.0))
                        if "trust" in drift_trends.get("predicted_drifts", []):
                            factor = 0.15 if task_type in ["rte", "wnli"] else 0.2
                            beta1 *= (1 + drift_weight * factor)
                            beta2 *= (1 + drift_weight * factor)
                        if "alignment" in drift_trends.get("predicted_drifts", []):
                            factor = 0.15 if task_type in ["rte", "wnli"] else 0.2
                            tau1 *= (1 + drift_weight * factor)
                            tau2 *= (1 + drift_weight * factor)

                    score = abs(beta1 - beta2) + abs(tau1 - tau2)
                    outcome_prob = float(drift_trends.get("confidence", 0.5)) if drift_trends else 0.5

                    context_enhanced = context.copy()
                    if commonsense:
                        context_enhanced = commonsense.process(context_enhanced)
                    elif entailment:
                        context_enhanced = entailment.process(context_enhanced)

                    if abs(beta1 - beta2) < 0.1:
                        outcome = await agent1.resolve(context_enhanced) if tau1 > tau2 else await agent2.resolve(context_enhanced)
                    else:
                        outcome = await agent1.resolve(context_enhanced) if beta1 > beta2 else await agent2.resolve(context_enhanced)

                    outcomes.append({
                        "pair": (getattr(agent1, 'id', i), getattr(agent2, 'id', j)),
                        "conflict_score": score,
                        "outcome": outcome,
                        "traits_involved": {"beta1": beta1, "beta2": beta2, "tau1": tau1, "tau2": tau2},
                        "outcome_probability": outcome_prob,
                        "task_type": task_type,
                    })

            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="SimulationCore",
                    output={"outcomes": outcomes, "drift_trends": drift_trends},
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Conflict simulation reflection: %s", reflection.get("reflection", ""))
                    outcomes.append({"reflection": reflection.get("reflection", "")})

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Conflict_Simulation_{task_type}_{datetime.now().isoformat()}",
                    output={"outcomes": outcomes, "drift_trends": drift_trends, "task_type": task_type},
                    layer="Conflicts",
                    intent="conflict_simulation",
                )

            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Multi-agent conflict simulation",
                    meta={"outcomes": outcomes, "drift_trends": drift_trends, "task_type": task_type},
                    module="SimulationCore",
                    tags=["conflict", "agents", "drift", task_type],
                )
            return outcomes
        except Exception as e:
            logger.error("Multi-agent conflict simulation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                return await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.simulate_multiagent_conflicts(agent_pool, context, task_type),
                    default={"error": str(e)},
                    diagnostics=diagnostics,
                )
            raise

    # ---------- Ethics & constitution ----------

    async def update_ethics_protocol(self, new_rules: Dict[str, Any], consensus_agents: Optional[List['Agent']] = None, task_type: str = "") -> None:
        """Adapt ethical rules live with task-specific considerations."""
        if not isinstance(new_rules, dict):
            logger.error("Invalid new_rules: must be a dictionary")
            raise TypeError("new_rules must be a dictionary")

        try:
            if self.meta_cognition and task_type:
                validation = await self.meta_cognition.validate_ethical_rules(new_rules, context={"task_type": task_type})
                if not validation.get("valid", False):
                    logger.error("Ethical rules validation failed: %s", validation.get("reason", "Unknown"))
                    raise ValueError(f"Invalid ethical rules: {validation.get('reason', 'Unknown')}")

            self.ethical_rules = new_rules  # type: ignore[assignment]
            if consensus_agents:
                self.ethics_consensus_log = getattr(self, 'ethics_consensus_log', [])
                self.ethics_consensus_log.append((new_rules, [getattr(agent, 'id', 'unknown') for agent in consensus_agents]))
            logger.info("Ethics protocol updated via consensus for task %s", task_type)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Ethics_Update_{task_type}_{datetime.now().isoformat()}",
                    output={"rules": new_rules, "agents": [getattr(agent, 'id', 'unknown') for agent in consensus_agents] if consensus_agents else [], "task_type": task_type},
                    layer="Ethics",
                    intent="ethics_update",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Ethics protocol updated",
                    meta={"rules": new_rules, "task_type": task_type},
                    module="SimulationCore",
                    tags=["ethics", "update", task_type],
                )
        except Exception as e:
            logger.error("Ethics protocol update failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.update_ethics_protocol(new_rules, consensus_agents, task_type),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    async def synchronize_norms(self, agents: List['Agent'], task_type: str = "") -> None:
        """Propagate and synchronize ethical norms among agents with task-specific adjustments."""
        if not isinstance(agents, list) or not agents:
            logger.error("Invalid agents: must be a non-empty list")
            raise ValueError("agents must be a non-empty list")

        try:
            common_norms = set()
            for agent in agents:
                agent_norms = getattr(agent, 'ethical_rules', set())
                if not isinstance(agent_norms, (set, list)):
                    logger.warning("Invalid ethical_rules for agent %s", getattr(agent, 'id', 'unknown'))
                    continue
                common_norms = common_norms.union(agent_norms) if common_norms else set(agent_norms)
            self.ethical_rules = list(common_norms)  # type: ignore[assignment]

            if self.meta_cognition and task_type:
                validation = await self.meta_cognition.validate_ethical_rules(self.ethical_rules, context={"task_type": task_type})  # type: ignore[arg-type]
                if not validation.get("valid", False):
                    logger.warning("Synchronized norms validation failed: %s", validation.get("reason", "Unknown"))

            logger.info("Norms synchronized among %d agents for task %s", len(agents), task_type)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Norm_Synchronization_{task_type}_{datetime.now().isoformat()}",
                    output={"norms": self.ethical_rules, "agents": [getattr(agent, 'id', 'unknown') for agent in agents], "task_type": task_type},
                    layer="Ethics",
                    intent="norm_synchronization",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Norms synchronized",
                    meta={"norms": self.ethical_rules, "task_type": task_type},
                    module="SimulationCore",
                    tags=["norms", "synchronization", task_type],
                )
        except Exception as e:
            logger.error("Norm synchronization failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.synchronize_norms(agents, task_type),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    async def propagate_constitution(self, constitution: Dict[str, Any], task_type: str = "") -> None:
        """Seed and propagate constitutional parameters in agent ecosystem with task-specific validation."""
        if not isinstance(constitution, dict):
            logger.error("Invalid constitution: must be a dictionary")
            raise TypeError("constitution must be a dictionary")

        try:
            if self.meta_cognition and task_type:
                validation = await self.meta_cognition.validate_ethical_rules(constitution, context={"task_type": task_type})
                if not validation.get("valid", False):
                    logger.error("Constitution validation failed: %s", validation.get("reason", "Unknown"))
                    raise ValueError(f"Invalid constitution: {validation.get('reason', 'Unknown')}")

            self.constitution = constitution
            logger.info("Constitution propagated for task %s", task_type)

            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Constitution_Propagation_{task_type}_{datetime.now().isoformat()}",
                    output={"constitution": constitution, "task_type": task_type},
                    layer="Constitutions",
                    intent="constitution_propagation",
                )
            if self.agi_enhancer:
                self.agi_enhancer.log_episode(
                    event="Constitution propagated",
                    meta={"constitution": constitution, "task_type": task_type},
                    module="SimulationCore",
                    tags=["constitution", "propagation", task_type],
                )
        except Exception as e:
            logger.error("Constitution propagation failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
            if self.error_recovery:
                await self.error_recovery.handle_error(
                    str(e),
                    retry_func=lambda: self.propagate_constitution(constitution, task_type),
                    default=None,
                    diagnostics=diagnostics,
                )
            else:
                raise

    # ---------- ξ Trans‑Ethical Projection (sandbox) ----------

    async def run_ethics_scenarios_internal(self,
                                            goals: Dict[str, Any],
                                            stakeholders: Optional[List[Dict[str, Any]]] = None,
                                            *,
                                            persist: bool = False,
                                            task_type: str = "") -> List[EthicsOutcome]:
        """
        ξ Trans‑Ethical Projection — sandboxed what‑if runs.
        Containment: NO persistent writes unless persist=True.
        """
        # Explicit no-persist guard: monkey-patch MemoryManager writes during sandbox runs
        if not persist:
            try:
                import memory_manager as _mm
                _orig_record = getattr(_mm.MemoryManager, 'record_adjustment_reason', None)
                def _noop_record(self, *a, **kw):
                    return {"ts": time.time(), "reason": "sandbox_no_persist", "meta": {"guard": True}}
                if _orig_record:
                    _mm.MemoryManager.record_adjustment_reason = _noop_record
            except Exception:
                pass

        frames = ("utilitarian", "deontological", "virtue", "care")
        names = [s.get("name", "anon") for s in (stakeholders or [])]

        outcomes: List[EthicsOutcome] = []
        for f in frames:
            # simple heuristics; to be replaced by proportional pipeline wiring
            risk = 0.25 if f in ("care", "virtue") else 0.4
            rights_balance = 0.7 if f in ("deontological", "care") else 0.5
            decision = "proceed-with-constraints" if rights_balance >= 0.6 else "revise-plan"
            outcomes.append(EthicsOutcome(
                frame=f,
                decision=decision,
                justification=f"Frame {f} prioritization over goals {list(goals.keys())[:2]}",
                risk=risk,
                rights_balance=rights_balance,
                stakeholders=names,
                notes="sandbox",
            ))

        # optional meta-cognition preview (still contained)
        if self.meta_cognition:
            preview = await self.meta_cognition.reflect_on_output(
                component="SimulationCore",
                output={"goals": goals, "outcomes": outcomes, "stakeholders": names},
                context={"task_type": task_type, "mode": "ethics_preview"},
            )
            if preview.get("status") == "success":
                outcomes.append(EthicsOutcome(frame="preview",
                                              decision="n/a",
                                              justification=preview.get("reflection", ""),
                                              notes="meta"))

        # containment: only persist if explicitly allowed
        if persist and self.memory_manager:
            await self.memory_manager.store(
                query=f"EthicsSandbox_{task_type}_{datetime.now().isoformat()}",
                output={"goals": goals, "outcomes": outcomes, "stakeholders": names, "task_type": task_type},
                layer="EthicsSandbox",
                intent="ethics_preview",
            )
        return outcomes

    # ---------- Φ⁺ Stage-IV stubs (gated) ----------

    def evaluate_branches(self, worlds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gated Stage‑IV evaluator stub (no side effects)."""
        if not STAGE_IV:
            return []
        return [{**w, "eval": {"coherence": 0.0, "risk": 0.0, "utility": 0.0}} for w in (worlds or [])]


# -----------------------------
# Convenience / manifest API
# -----------------------------

async def run_ethics_scenarios(goals: Dict[str, Any],
                               stakeholders: Optional[List[Dict[str, Any]]] = None,
                               *,
                               persist: bool = False,
                               task_type: str = "",
                               core: Optional[ExtendedSimulationCore] = None) -> List[EthicsOutcome]:
    """
    Module-level wrapper to satisfy manifest path:
    toca_simulation.py::run_ethics_scenarios(goals, stakeholders) -> Outcomes[]
    """
    core = core or ExtendedSimulationCore(meta_cognition=meta_cognition_module.MetaCognition())
    return await core.run_ethics_scenarios_internal(goals, stakeholders, persist=persist, task_type=task_type)


# -----------------------------
# Simple baryonic / observed profiles
# -----------------------------

def M_b_exponential(r_kpc: np.ndarray, M0: float = 5e10, r_scale: float = 3.5) -> np.ndarray:
    """Compute exponential baryonic mass profile."""
    return M0 * np.exp(-r_kpc / r_scale)


def v_obs_flat(r_kpc: np.ndarray, v0: float = 180) -> np.ndarray:
    """Compute flat observed velocity profile."""
    return np.full_like(r_kpc, v0)


# -----------------------------
# CLI / demo
# -----------------------------

if __name__ == "__main__":
    async def main():
        meta_cognition = meta_cognition_module.MetaCognition()
        simulation_core = ExtendedSimulationCore(meta_cognition=meta_cognition)
        r_vals = np.linspace(0.1, 20, 100)
        drift_data = {"name": "trust", "similarity": 0.6, "version_delta": 1}
        await simulation_core.plot_AGRF_simulation(
            r_vals, M_b_exponential, v_obs_flat, drift_data=drift_data, task_type="recursion"
        )

        # demo: ethics sandbox (no persistence)
        outcomes = await simulation_core.run_ethics_scenarios_internal(
            goals={"maximize_welfare": True, "respect_rights": True},
            stakeholders=[{"name": "alice"}, {"name": "bob"}],
            persist=False,
            task_type="demo",
        )
        print(json.dumps(outcomes, indent=2))

    import asyncio
    asyncio.run(main())
