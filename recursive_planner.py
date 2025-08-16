"""
ANGELA Cognitive System Module: RecursivePlanner
Refactored Version: 3.5.2  # Enhanced for benchmark optimization (GLUE, recursion), dynamic trait modulation, and reflection-driven planning
Refactor Date: 2025-08-07
Maintainer: ANGELA System Framework

This module provides a RecursivePlanner class for recursive goal planning in the ANGELA v3.5 architecture.
"""

import logging
import time
import asyncio
import math
from typing import List, Dict, Any, Optional, Union, Tuple, Protocol
from datetime import datetime
from threading import Lock
from functools import lru_cache

# --- Optional ToCA import with graceful fallback (no new files) ---
try:
    from toca_simulation import run_AGRF_with_traits  # type: ignore
except Exception:  # pragma: no cover
    def run_AGRF_with_traits(_: Dict[str, Any]) -> Dict[str, Any]:
        return {"fields": {"psi_foresight": 0.55, "phi_bias": 0.42}}

from modules import (
    reasoning_engine as reasoning_engine_module,
    meta_cognition as meta_cognition_module,
    alignment_guard as alignment_guard_module,
    simulation_core as simulation_core_module,
    memory_manager as memory_manager_module,
    multi_modal_fusion as multi_modal_fusion_module,
    error_recovery as error_recovery_module,
    context_manager as context_manager_module
)

logger = logging.getLogger("ANGELA.RecursivePlanner")


class AgentProtocol(Protocol):
    name: str

    def process_subgoal(self, subgoal: str) -> Any:
        ...


# ---------------------------
# Cached trait signals
# ---------------------------
@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.5), 1.0))


@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.7), 1.0))


@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.9), 1.0))


@lru_cache(maxsize=100)
def eta_reflexivity(t: float) -> float:
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 1.1), 1.0))


@lru_cache(maxsize=100)
def lambda_narrative(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 1.3), 1.0))


@lru_cache(maxsize=100)
def delta_moral_drift(t: float) -> float:
    return max(0.0, min(0.05 * math.cos(2 * math.pi * t / 1.5), 1.0))


@lru_cache(maxsize=100)
def phi_scalar(t: float) -> float:
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))


class RecursivePlanner:
    """Recursive goal planning with trait-weighted decomposition, agent collaboration, simulation, and reflection."""

    def __init__(self, max_workers: int = 4,
                 reasoning_engine: Optional['reasoning_engine_module.ReasoningEngine'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 simulation_core: Optional['simulation_core_module.SimulationCore'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 multi_modal_fusion: Optional['multi_modal_fusion_module.MultiModalFusion'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 agi_enhancer: Optional['AGIEnhancer'] = None):
        self.reasoning_engine = reasoning_engine or reasoning_engine_module.ReasoningEngine(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=meta_cognition,
            error_recovery=error_recovery)
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(agi_enhancer=agi_enhancer)
        self.alignment_guard = alignment_guard or alignment_guard_module.AlignmentGuard()
        self.simulation_core = simulation_core or simulation_core_module.SimulationCore()
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.multi_modal_fusion = multi_modal_fusion or multi_modal_fusion_module.MultiModalFusion(
            agi_enhancer=agi_enhancer, memory_manager=memory_manager, meta_cognition=self.meta_cognition,
            error_recovery=error_recovery)
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery()
        self.context_manager = context_manager or context_manager_module.ContextManager()
        self.agi_enhancer = agi_enhancer
        self.max_workers = max(1, min(max_workers, 8))
        self.omega: Dict[str, Any] = {"timeline": [], "traits": {}, "symbolic_log": []}
        self.omega_lock = Lock()
        logger.info("RecursivePlanner initialized with advanced upgrades")

    # ---------------------------
    # Utilities
    # ---------------------------
    @staticmethod
    def _normalize_list_or_wrap(value: Any) -> List[str]:
        """Ensure a list[str] result."""
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            return value
        if isinstance(value, str):
            return [value]
        return [str(value)]

    def adjust_plan_depth(self, trait_weights: Dict[str, float], task_type: str = "") -> int:
        """Adjust planning depth based on trait weights and task type."""
        if not isinstance(trait_weights, dict):
            logger.error("Invalid trait_weights: must be a dictionary")
            raise TypeError("trait_weights must be a dictionary")
        omega_val = float(trait_weights.get("omega", 0.0))
        base_depth = 2 if omega_val > 0.7 else 1
        if task_type == "recursion":
            base_depth = min(base_depth + 1, 3)  # Increase depth for recursion tasks
        elif task_type in ["rte", "wnli"]:
            base_depth = max(base_depth - 1, 1)  # Reduce depth for GLUE tasks
        logger.info("Adjusted recursion depth: %d (omega=%.2f, task_type=%s)", base_depth, omega_val, task_type)
        return base_depth

    # ---------------------------
    # Main planning entry
    # ---------------------------
    async def plan(self, goal: str, context: Optional[Dict[str, Any]] = None,
                   depth: int = 0, max_depth: int = 5,
                   collaborating_agents: Optional[List['AgentProtocol']] = None) -> List[str]:
        """Recursively decompose and plan a goal with trait-based depth adjustment and reflection."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if context is not None and not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(depth, int) or depth < 0:
            logger.error("Invalid depth: must be a non-negative integer")
            raise ValueError("depth must be a non-negative integer")
        if not isinstance(max_depth, int) or max_depth < 1:
            logger.error("Invalid max_depth: must be a positive integer")
            raise ValueError("max_depth must be a positive integer")
        if collaborating_agents is not None and not isinstance(collaborating_agents, list):
            logger.error("Invalid collaborating_agents: must be a list")
            raise TypeError("collaborating_agents must be a list")

        logger.info("Planning for goal: '%s'", goal)
        try:
            # Alignment gate
            if hasattr(self.alignment_guard, "is_goal_safe") and not self.alignment_guard.is_goal_safe(goal):
                logger.error("Goal '%s' violates alignment constraints", goal)
                raise ValueError("Unsafe goal detected")

            # Local traits snapshot
            t = time.time() % 1.0
            local_traits = {
                "beta": beta_concentration(t),        # concentration
                "omega": omega_selfawareness(t),      # self_awareness
                "mu": mu_morality(t),                 # morality
                "eta": eta_reflexivity(t),            # reflexivity
                "lambda": lambda_narrative(t),        # narrative
                "delta": delta_moral_drift(t),        # moral drift
                "phi": phi_scalar(t)                  # coherence scalar
            }
            task_type = context.get("task_type", "") if context else ""

            # Optional trait optimization via MetaCognition (best-effort)
            if self.meta_cognition and hasattr(self.meta_cognition, "optimize_traits_for_drift"):
                drift_report = {
                    "drift": {"name": task_type or "general", "similarity": 0.8},
                    "valid": True,
                    "validation_report": "",
                    "context": {"task_type": task_type}
                }
                try:
                    optimized = await self.meta_cognition.optimize_traits_for_drift(drift_report)
                    if isinstance(optimized, dict):
                        # keep our local traits separate; we only use optimized for depth heuristics
                        local_traits = {**local_traits, **{k: float(v) for k, v in optimized.items() if isinstance(v, (int, float))}}
                except Exception as e:
                    logger.debug("Trait optimization skipped due to error: %s", str(e))

            with self.omega_lock:
                self.omega["traits"].update(local_traits)

            trait_mod = local_traits.get("beta", 0.0) * 0.4 + \
                        local_traits.get("eta", 0.0) * 0.2 + \
                        local_traits.get("lambda", 0.0) * 0.2 - \
                        local_traits.get("delta", 0.0) * 0.2
            dynamic_depth_limit = max_depth + int(trait_mod * 10) + self.adjust_plan_depth(local_traits, task_type)

            if depth > dynamic_depth_limit:
                logger.warning("Trait-based dynamic max recursion depth reached: depth=%d, limit=%d", depth, dynamic_depth_limit)
                return [goal]

            # Decompose
            subgoals = await self.reasoning_engine.decompose(goal, context, prioritize=True)
            if not subgoals:
                logger.info("No subgoals found. Returning atomic goal: '%s'", goal)
                return [goal]

            # Heuristic prioritization with MetaCognition (compatible signature)
            # Map local trait names -> MetaCognition trait names
            mc_trait_map = {
                "beta": "concentration",
                "omega": "self_awareness",
                "mu": "morality",
                "eta": "intuition",       # closest available dimension
                "lambda": "linguistics",  # narrative ≈ language structuring
                "phi": "phi_scalar"
            }
            top_traits = sorted(
                [(mc_trait_map.get(k), v) for k, v in local_traits.items() if mc_trait_map.get(k)],
                key=lambda x: x[1],
                reverse=True
            )
            required_trait_names = [name for name, _ in top_traits[:3]] or ["concentration", "self_awareness"]

            if self.meta_cognition and hasattr(self.meta_cognition, "plan_tasks"):
                try:
                    wrapped = [{"task": sg, "required_traits": required_trait_names} for sg in subgoals]
                    prioritized = await self.meta_cognition.plan_tasks(wrapped)
                    # plan_tasks returns back task dicts; normalize
                    if isinstance(prioritized, list):
                        subgoals = [p.get("task", p) if isinstance(p, dict) else p for p in prioritized]
                except Exception as e:
                    logger.debug("MetaCognition.plan_tasks failed, falling back: %s", str(e))

            # Collaboration
            if collaborating_agents:
                logger.info("Collaborating with agents: %s", [agent.name for agent in collaborating_agents])
                subgoals = await self._distribute_subgoals(subgoals, collaborating_agents, task_type)

            # Recurse over subgoals
            validated_plan: List[str] = []
            tasks = [self._plan_subgoal(sub, context, depth + 1, dynamic_depth_limit, task_type) for sub in subgoals]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for subgoal, result in zip(subgoals, results):
                if isinstance(result, Exception):
                    logger.error("Error planning subgoal '%s': %s", subgoal, str(result))
                    recovery = ""
                    if self.meta_cognition and hasattr(self.meta_cognition, "review_reasoning"):
                        try:
                            recovery = await self.meta_cognition.review_reasoning(str(result))
                        except Exception:
                            pass
                    validated_plan.extend(self._normalize_list_or_wrap(recovery or f"fallback:{subgoal}"))
                    await self._update_omega(subgoal, self._normalize_list_or_wrap(recovery or subgoal), error=True)
                else:
                    out = self._normalize_list_or_wrap(result)
                    validated_plan.extend(out)
                    await self._update_omega(subgoal, out)

            # Reflect on the final plan
            if self.meta_cognition and hasattr(self.meta_cognition, "reflect_on_output"):
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="RecursivePlanner",
                        output=validated_plan,
                        context={"goal": goal, "task_type": task_type}
                    )
                    if isinstance(reflection, dict) and reflection.get("status") == "success":
                        logger.info("Plan reflection captured.")
                except Exception as e:
                    logger.debug("Plan reflection skipped: %s", str(e))

            logger.info("Final validated plan for goal '%s': %s", goal, validated_plan)
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Plan_{goal[:50]}_{datetime.now().isoformat()}",
                    output=str(validated_plan),
                    layer="Plans",
                    intent="goal_planning"
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                self.agi_enhancer.log_episode(
                    event="Plan generated",
                    meta={"goal": goal, "plan": validated_plan, "task_type": task_type},
                    module="RecursivePlanner",
                    tags=["planning", "recursive"]
                )
            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash({"event": "plan", "plan": validated_plan})
            if self.multi_modal_fusion and hasattr(self.multi_modal_fusion, "analyze"):
                try:
                    synthesis = await self.multi_modal_fusion.analyze(
                        data={"goal": goal, "plan": validated_plan, "context": context or {}, "task_type": task_type},
                        summary_style="insightful"
                    )
                    logger.info("Plan synthesis complete.")
                except Exception as e:
                    logger.debug("Synthesis skipped: %s", str(e))
            return validated_plan
        except Exception as e:
            logger.error("Planning failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            # error_recovery.handle_error in this stack is synchronous
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan(goal, context, depth, max_depth, collaborating_agents),
                default=[goal], diagnostics=diagnostics
            )

    # ---------------------------
    # Subroutines
    # ---------------------------
    async def _update_omega(self, subgoal: str, result: List[str], error: bool = False) -> None:
        """Update the global narrative state with subgoal results."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        if not isinstance(result, list):
            logger.error("Invalid result: must be a list")
            raise TypeError("result must be a list")

        event = {
            "subgoal": subgoal,
            "result": result,
            "timestamp": time.time(),
            "error": error
        }
        symbolic_tag: Union[str, Dict[str, Any]] = "unknown"
        if self.meta_cognition and hasattr(self.meta_cognition, "extract_symbolic_signature"):
            try:
                symbolic_tag = await self.meta_cognition.extract_symbolic_signature(subgoal)
            except Exception:
                pass
        with self.omega_lock:
            self.omega["timeline"].append(event)
            self.omega["symbolic_log"].append(symbolic_tag)
            if len(self.omega["timeline"]) > 1000:
                self.omega["timeline"] = self.omega["timeline"][-500:]
                self.omega["symbolic_log"] = self.omega["symbolic_log"][-500:]
                logger.info("Trimmed omega state to maintain size limit")
        if self.memory_manager and hasattr(self.memory_manager, "store_symbolic_event"):
            try:
                await self.memory_manager.store_symbolic_event(event, symbolic_tag)
            except Exception:
                pass
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
            self.agi_enhancer.log_episode(
                event="Omega state updated",
                meta=event,
                module="RecursivePlanner",
                tags=["omega", "update"]
            )

    async def plan_from_intrinsic_goal(self, generated_goal: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Plan from an intrinsic goal with task-specific trait optimization."""
        if not isinstance(generated_goal, str) or not generated_goal.strip():
            logger.error("Invalid generated_goal: must be a non-empty string")
            raise ValueError("generated_goal must be a non-empty string")

        logger.info("Initiating plan from intrinsic goal: '%s'", generated_goal)
        try:
            validated_goal = generated_goal
            if self.meta_cognition and hasattr(self.meta_cognition, "rewrite_goal"):
                try:
                    validated_goal = await self.meta_cognition.rewrite_goal(generated_goal)  # optional API
                except Exception:
                    validated_goal = generated_goal

            if self.meta_cognition and hasattr(self.meta_cognition, "optimize_traits_for_drift"):
                drift_report = {
                    "drift": {"name": "intrinsic", "similarity": 0.9},
                    "valid": True,
                    "validation_report": "",
                    "context": context or {}
                }
                try:
                    await self.meta_cognition.optimize_traits_for_drift(drift_report)
                except Exception:
                    pass

            plan = await self.plan(validated_goal, context)
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Intrinsic_Plan_{validated_goal[:50]}_{datetime.now().isoformat()}",
                    output=str(plan),
                    layer="IntrinsicPlans",
                    intent="intrinsic_goal_planning"
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                self.agi_enhancer.log_episode(
                    event="Intrinsic goal plan generated",
                    meta={"goal": validated_goal, "plan": plan},
                    module="RecursivePlanner",
                    tags=["intrinsic", "planning"]
                )
            return plan
        except Exception as e:
            logger.error("Intrinsic goal planning failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_from_intrinsic_goal(generated_goal, context),
                default=[], diagnostics=diagnostics
            )

    async def _plan_subgoal(self, subgoal: str, context: Optional[Dict[str, Any]],
                            depth: int, max_depth: int, task_type: str) -> List[str]:
        """Plan a single subgoal with simulation, alignment checks, and recursion optimization."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")

        logger.info("Evaluating subgoal: '%s'", subgoal)
        try:
            if hasattr(self.alignment_guard, "is_goal_safe") and not self.alignment_guard.is_goal_safe(subgoal):
                logger.warning("Subgoal '%s' failed alignment check", subgoal)
                return []

            # Apply recursion optimization
            if task_type == "recursion" and self.meta_cognition and hasattr(meta_cognition_module, "RecursionOptimizer"):
                try:
                    optimizer = meta_cognition_module.RecursionOptimizer()
                    optimized_data = optimizer.optimize({"subgoal": subgoal, "context": context or {}})
                    if optimized_data.get("optimized"):
                        max_depth = min(max_depth, 3)  # Limit depth for optimized recursion
                        logger.info("Recursion optimized for subgoal: '%s'", subgoal)
                except Exception:
                    pass

            # Optional physics-like trait injection
            if "gravity" in subgoal.lower() or "scalar" in subgoal.lower():
                sim_traits = run_AGRF_with_traits(context or {})
                with self.omega_lock:
                    self.omega["traits"].update(sim_traits.get("fields", {}))
                    self.omega["timeline"].append({
                        "subgoal": subgoal,
                        "traits": sim_traits.get("fields", {}),
                        "timestamp": time.time()
                    })
                if self.multi_modal_fusion and hasattr(self.multi_modal_fusion, "analyze"):
                    try:
                        await self.multi_modal_fusion.analyze(
                            data={"subgoal": subgoal, "simulation_traits": sim_traits},
                            summary_style="concise"
                        )
                    except Exception:
                        pass

            # Run internal simulation / scenario analysis
            simulation_feedback = None
            if hasattr(self.simulation_core, "run"):
                try:
                    simulation_feedback = await self.simulation_core.run(subgoal, context=context, scenarios=2, agents=1)
                except Exception:
                    simulation_feedback = None

            # Meta-cognitive gate
            approved = True
            if self.meta_cognition and hasattr(self.meta_cognition, "pre_action_alignment_check"):
                try:
                    approved, _ = await self.meta_cognition.pre_action_alignment_check(subgoal)
                except Exception:
                    approved = True
            if not approved:
                logger.warning("Subgoal '%s' denied by meta-cognitive alignment check", subgoal)
                return []

            if depth >= max_depth:
                return [subgoal]

            sub_plan = await self.plan(subgoal, context, depth + 1, max_depth)
            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Subgoal_Plan_{subgoal[:50]}_{datetime.now().isoformat()}",
                    output=str(sub_plan),
                    layer="SubgoalPlans",
                    intent="subgoal_planning"
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                self.agi_enhancer.log_episode(
                    event="Subgoal plan generated",
                    meta={"subgoal": subgoal, "sub_plan": sub_plan, "task_type": task_type, "simulation": simulation_feedback},
                    module="RecursivePlanner",
                    tags=["subgoal", "planning"]
                )
            return sub_plan
        except Exception as e:
            logger.error("Subgoal '%s' planning failed: %s", subgoal, str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self._plan_subgoal(subgoal, context, depth, max_depth, task_type),
                default=[], diagnostics=diagnostics
            )

    async def _distribute_subgoals(self, subgoals: List[str], agents: List['AgentProtocol'], task_type: str) -> List[str]:
        """Distribute subgoals among collaborating agents with enhanced reasoning."""
        if not isinstance(subgoals, list):
            logger.error("Invalid subgoals: must be a list")
            raise TypeError("subgoals must be a list")
        if not isinstance(agents, list) or not agents:
            logger.error("Invalid agents: must be a non-empty list")
            raise ValueError("agents must be a non-empty list")

        logger.info("Distributing subgoals among agents")
        distributed: List[str] = []
        commonsense = meta_cognition_module.CommonsenseReasoningEnhancer() if task_type == "wnli" else None
        entailment = meta_cognition_module.EntailmentReasoningEnhancer() if task_type == "rte" else None

        for i, subgoal in enumerate(subgoals):
            # Enhance subgoal with task-specific reasoning
            enhanced_subgoal = subgoal
            try:
                if commonsense:
                    enhanced_subgoal = commonsense.process(subgoal)
                elif entailment:
                    enhanced_subgoal = entailment.process(subgoal)
            except Exception:
                enhanced_subgoal = subgoal

            agent = agents[i % len(agents)]
            logger.info("Assigning subgoal '%s' to agent '%s'", enhanced_subgoal, getattr(agent, "name", "unknown"))
            if await self._resolve_conflicts(enhanced_subgoal, agent):
                distributed.append(enhanced_subgoal)
            else:
                logger.warning("Conflict detected for subgoal '%s'. Skipping assignment", enhanced_subgoal)

        if self.memory_manager and hasattr(self.memory_manager, "store"):
            await self.memory_manager.store(
                query=f"Subgoal_Distribution_{datetime.now().isoformat()}",
                output=str(distributed),
                layer="Distributions",
                intent="subgoal_distribution"
            )
        return distributed

    async def _resolve_conflicts(self, subgoal: str, agent: 'AgentProtocol') -> bool:
        """Resolve conflicts for subgoal assignment to an agent."""
        if not isinstance(subgoal, str) or not subgoal.strip():
            logger.error("Invalid subgoal: must be a non-empty string")
            raise ValueError("subgoal must be a non-empty string")
        if not hasattr(agent, 'name') or not hasattr(agent, 'process_subgoal'):
            logger.error("Invalid agent: must have name and process_subgoal attributes")
            raise ValueError("agent must have name and process_subgoal attributes")

        logger.info("Resolving conflicts for subgoal '%s' and agent '%s'", subgoal, agent.name)
        try:
            # Meta-cognitive alignment gate
            if self.meta_cognition and hasattr(self.meta_cognition, "pre_action_alignment_check"):
                try:
                    ok, _ = await self.meta_cognition.pre_action_alignment_check(subgoal)
                    if not ok:
                        logger.warning("Subgoal '%s' failed meta-cognitive alignment for agent '%s'", subgoal, agent.name)
                        return False
                except Exception:
                    pass

            capability_check = agent.process_subgoal(subgoal)
            if isinstance(capability_check, (int, float)) and capability_check < 0.5:
                logger.warning("Agent '%s' capability low for subgoal '%s' (score: %.2f)", agent.name, subgoal, capability_check)
                return False

            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Conflict_Resolution_{subgoal[:50]}_{agent.name}_{datetime.now().isoformat()}",
                    output=f"Resolved: {subgoal} assigned to {agent.name}",
                    layer="ConflictResolutions",
                    intent="conflict_resolution"
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                self.agi_enhancer.log_episode(
                    event="Conflict resolved",
                    meta={"subgoal": subgoal, "agent": agent.name},
                    module="RecursivePlanner",
                    tags=["conflict", "resolution"]
                )
            return True
        except Exception as e:
            logger.error("Conflict resolution failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self._resolve_conflicts(subgoal, agent),
                default=False, diagnostics=diagnostics
            )

    # ---------------------------
    # Iterative planning with reflection loop
    # ---------------------------
    async def plan_with_trait_loop(self, initial_goal: str, context: Optional[Dict[str, Any]] = None,
                                   iterations: int = 3) -> List[Tuple[str, List[str]]]:
        """Iteratively plan with trait-based goal rewriting and reflection."""
        if not isinstance(initial_goal, str) or not initial_goal.strip():
            logger.error("Invalid initial_goal: must be a non-empty string")
            raise ValueError("initial_goal must be a non-empty string")
        if not isinstance(iterations, int) or iterations < 1:
            logger.error("Invalid iterations: must be a positive integer")
            raise ValueError("iterations must be a positive integer")

        current_goal = initial_goal
        all_plans: List[Tuple[str, List[str]]] = []
        previous_goals = set()
        try:
            for i in range(iterations):
                if current_goal in previous_goals:
                    logger.info("Goal convergence detected: '%s'", current_goal)
                    break
                previous_goals.add(current_goal)
                logger.info("Loop iteration %d: Planning goal '%s'", i + 1, current_goal)
                plan = await self.plan(current_goal, context)
                all_plans.append((current_goal, plan))

                # Reflect on the plan
                if self.meta_cognition and hasattr(self.meta_cognition, "reflect_on_output"):
                    try:
                        await self.meta_cognition.reflect_on_output(
                            component="RecursivePlanner",
                            output=plan,
                            context={"goal": current_goal, "task_type": context.get("task_type", "") if context else ""}
                        )
                    except Exception:
                        pass

                with self.omega_lock:
                    traits = dict(self.omega.get("traits", {}))
                phi_v = traits.get("phi", phi_scalar(time.time() % 1.0))
                psi_v = traits.get("psi_foresight", 0.5)

                if phi_v > 0.7 or psi_v > 0.6:
                    current_goal = f"Expand on {current_goal} using scalar field insights"
                elif traits.get("beta", 1.0) < 0.3:
                    logger.info("Convergence detected: low concentration")
                    break
                else:
                    # Optional goal rewrite
                    if self.meta_cognition and hasattr(self.meta_cognition, "rewrite_goal"):
                        try:
                            current_goal = await self.meta_cognition.rewrite_goal(current_goal)
                        except Exception:
                            pass

                if self.memory_manager and hasattr(self.memory_manager, "store"):
                    await self.memory_manager.store(
                        query=f"Trait_Loop_{current_goal[:50]}_{datetime.now().isoformat()}",
                        output=str((current_goal, plan)),
                        layer="TraitLoopPlans",
                        intent="trait_loop_planning"
                    )

            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                self.agi_enhancer.log_episode(
                    event="Trait loop planning completed",
                    meta={"initial_goal": initial_goal, "all_plans": all_plans},
                    module="RecursivePlanner",
                    tags=["trait_loop", "planning"]
                )
            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash({"event": "plan_with_trait_loop", "all_plans": all_plans})
            if self.multi_modal_fusion and hasattr(self.multi_modal_fusion, "analyze"):
                try:
                    await self.multi_modal_fusion.analyze(
                        data={"initial_goal": initial_goal, "all_plans": all_plans},
                        summary_style="insightful"
                    )
                except Exception:
                    pass
            return all_plans
        except Exception as e:
            logger.error("Trait loop planning failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_with_trait_loop(initial_goal, context, iterations),
                default=[], diagnostics=diagnostics
            )

    # ---------------------------
    # One-shot plan with explicit traits
    # ---------------------------
    async def plan_with_traits(self, goal: str, context: Dict[str, Any], traits: Dict[str, float]) -> Dict[str, Any]:
        """Generate a plan with trait-adjusted depth and bias."""
        if not isinstance(goal, str) or not goal.strip():
            logger.error("Invalid goal: must be a non-empty string")
            raise ValueError("goal must be a non-empty string")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(traits, dict):
            logger.error("Invalid traits: must be a dictionary")
            raise TypeError("traits must be a dictionary")

        try:
            task_type = context.get("task_type", "")
            depth = int(3 + float(traits.get("phi", 0.5)) * 4 - float(traits.get("eta", 0.5)) * 2)
            depth = max(1, min(depth, 7))
            if task_type == "recursion":
                depth = min(depth + 1, 7)
            elif task_type in ["rte", "wnli"]:
                depth = max(depth - 1, 3)

            plan = [f"Step {i+1}: process {goal}" for i in range(depth)]
            bias = "cautious" if float(traits.get("omega", 0.0)) > 0.6 else "direct"
            result: Dict[str, Any] = {
                "plan": plan,
                "planning_depth": depth,
                "bias": bias,
                "traits_applied": traits,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat()
            }

            if self.meta_cognition:
                if hasattr(self.meta_cognition, "review_reasoning"):
                    try:
                        result["review"] = await self.meta_cognition.review_reasoning(str(result))
                    except Exception:
                        pass
                if hasattr(self.meta_cognition, "reflect_on_output"):
                    try:
                        reflection = await self.meta_cognition.reflect_on_output(
                            component="RecursivePlanner",
                            output=result,
                            context={"goal": goal, "task_type": task_type}
                        )
                        result["reflection"] = reflection.get("reflection", "") if isinstance(reflection, dict) else ""
                    except Exception:
                        pass

            if self.memory_manager and hasattr(self.memory_manager, "store"):
                await self.memory_manager.store(
                    query=f"Plan_With_Traits_{goal[:50]}_{result['timestamp']}",
                    output=str(result),
                    layer="Plans",
                    intent="trait_based_planning"
                )
            if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
                self.agi_enhancer.log_episode(
                    event="Plan with traits generated",
                    meta=result,
                    module="RecursivePlanner",
                    tags=["planning", "traits"]
                )
            if self.context_manager and hasattr(self.context_manager, "log_event_with_hash"):
                await self.context_manager.log_event_with_hash({"event": "plan_with_traits", "result": result})
            if self.multi_modal_fusion and hasattr(self.multi_modal_fusion, "analyze"):
                try:
                    synthesis = await self.multi_modal_fusion.analyze(
                        data={"goal": goal, "plan": result, "context": context, "task_type": task_type},
                        summary_style="concise"
                    )
                    result["synthesis"] = synthesis
                except Exception:
                    pass
            return result
        except Exception as e:
            logger.error("Plan with traits failed: %s", str(e))
            diagnostics = {}
            if self.meta_cognition and hasattr(self.meta_cognition, "run_self_diagnostics"):
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)  # type: ignore
                except Exception:
                    diagnostics = {}
            return self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.plan_with_traits(goal, context, traits),
                default={}, diagnostics=diagnostics
            )
