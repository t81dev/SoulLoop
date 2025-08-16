"""
ANGELA Cognitive System Module
Refactored Version: 3.5.3

Enhanced for task-specific trait optimization, drift coordination, Stage IV hooks (gated),
long-horizon feedback, and visualization.
Refactor Date: 2025-08-10
Maintainer: ANGELA System Framework

This module provides classes for embodied agents, ecosystem management, and cognitive enhancements
in the ANGELA v3.5.3 architecture, with task-specific trait optimization, advanced drift coordination,
real-time data integration, Stage IV Φ⁰ (gated), and reflection-driven processing.
"""

import logging
import time
import math
import datetime
import asyncio
import os
import requests
import random
import json
from collections import deque, Counter
from typing import Dict, Any, Optional, List, Callable
from functools import lru_cache
import uuid
import aiohttp
import argparse
import numpy as np
from networkx import DiGraph

import reasoning_engine
import recursive_planner
import context_manager as context_manager_module
import simulation_core
import toca_simulation
import creative_thinker as creative_thinker_module
import knowledge_retriever
import learning_loop
import concept_synthesizer as concept_synthesizer_module
import memory_manager
import multi_modal_fusion
import code_executor as code_executor_module
import visualizer as visualizer_module
import external_agent_bridge
import alignment_guard as alignment_guard_module
import user_profile
import error_recovery as error_recovery_module
import meta_cognition as meta_cognition_module
from self_cloning_llm import SelfCloningLLM
from typing import Tuple

logger = logging.getLogger("ANGELA.CognitiveSystem")
SYSTEM_CONTEXT = {}
timechain_log = deque(maxlen=1000)
grok_query_log = deque(maxlen=60)
openai_query_log = deque(maxlen=60)

GROK_API_KEY = os.getenv("GROK_API_KEY")
# Manifest-driven flags (safe defaults if manifest/config is not injected)
STAGE_IV = True  # can be toggled via HaloEmbodimentLayer.init flags
LONG_HORIZON_DEFAULT = True  # defaultSpan is handled at pipeline logging level

def _fire_and_forget(coro):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        asyncio.run(coro)

class TimeChainMixin:
    """Mixin for logging timechain events."""
    def log_timechain_event(self, module: str, description: str) -> None:
        timechain_log.append({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "module": module,
            "description": description
        })
        if hasattr(self, "context_manager") and self.context_manager:
            maybe = self.context_manager.log_event_with_hash({
                "event": "timechain_event",
                "module": module,
                "description": description
            })
            # handle both sync/async implementations safely
            if asyncio.iscoroutine(maybe):
                _fire_and_forget(maybe)

    def get_timechain_log(self) -> List[Dict[str, Any]]:
        return list(timechain_log)

# Cognitive Trait Functions
@lru_cache(maxsize=100)
def epsilon_emotion(t: float) -> float:
    return 0.2 * math.sin(2 * math.pi * t / 0.1)

@lru_cache(maxsize=100)
def beta_concentration(t: float) -> float:
    return 0.3 * math.cos(math.pi * t)

@lru_cache(maxsize=100)
def theta_memory(t: float) -> float:
    return 0.1 * (1 - math.exp(-t))

@lru_cache(maxsize=100)
def gamma_creativity(t: float) -> float:
    return 0.15 * math.sin(math.pi * t)

@lru_cache(maxsize=100)
def delta_sleep(t: float) -> float:
    return 0.05 * (1 + math.cos(2 * math.pi * t))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    return 0.2 * (1 - math.cos(math.pi * t))

@lru_cache(maxsize=100)
def iota_intuition(t: float) -> float:
    return 0.1 * math.sin(3 * math.pi * t)

@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    return 0.1 * math.cos(2 * math.pi * t)

@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    return 0.2 * math.sin(math.pi * t / 0.5)

@lru_cache(maxsize=100)
def omega_selfawareness(t: float) -> float:
    return 0.25 * (1 + math.sin(math.pi * t))

@lru_cache(maxsize=100)
def kappa_culture(t: float, x: float) -> float:
    return 0.1 * math.cos(x + math.pi * t)

@lru_cache(maxsize=100)
def lambda_linguistics(t: float) -> float:
    return 0.15 * math.sin(2 * math.pi * t / 0.7)

@lru_cache(maxsize=100)
def chi_culturevolution(t: float) -> float:
    return 0.1 * math.cos(math.pi * t / 0.3)

@lru_cache(maxsize=100)
def psi_history(t: float) -> float:
    return 0.1 * (1 - math.exp(-t / 0.5))

@lru_cache(maxsize=100)
def zeta_spirituality(t: float) -> float:
    return 0.05 * math.sin(math.pi * t / 0.2)

@lru_cache(maxsize=100)
def xi_collective(t: float, x: float) -> float:
    return 0.1 * math.cos(x + 2 * math.pi * t)

@lru_cache(maxsize=100)
def tau_timeperception(t: float) -> float:
    return 0.15 * (1 + math.cos(math.pi * t / 0.4))

@lru_cache(maxsize=100)
def phi_field(x: float, t: float) -> float:
    t_normalized = t % 1.0
    trait_functions = [
        epsilon_emotion(t_normalized),
        beta_concentration(t_normalized),
        theta_memory(t_normalized),
        gamma_creativity(t_normalized),
        delta_sleep(t_normalized),
        mu_morality(t_normalized),
        iota_intuition(t_normalized),
        phi_physical(t_normalized),
        eta_empathy(t_normalized),
        omega_selfawareness(t_normalized),
        kappa_culture(t_normalized, x),
        lambda_linguistics(t_normalized),
        chi_culturevolution(t_normalized),
        psi_history(t_normalized),
        zeta_spirituality(t_normalized),
        xi_collective(t_normalized, x),
        tau_timeperception(t_normalized)
    ]
    return sum(trait_functions)

# Updated to align with manifest v3.5.3 roleMap
TRAIT_OVERLAY = {
    "Σ": ["toca_simulation", "concept_synthesizer", "user_profile"],
    "Υ": ["external_agent_bridge", "context_manager", "meta_cognition"],
    "Φ⁰": ["meta_cognition", "visualizer", "concept_synthesizer"],  # gated by STAGE_IV
    "Ω": ["recursive_planner", "toca_simulation"],
    "β": ["alignment_guard", "toca_simulation"],
    "δ": ["alignment_guard", "meta_cognition"],
    "ζ": ["error_recovery", "recursive_planner"],
    "θ": ["reasoning_engine", "recursive_planner"],
    "λ": ["memory_manager"],
    "μ": ["learning_loop"],
    "π": ["creative_thinker", "concept_synthesizer", "meta_cognition"],
    "χ": ["user_profile", "meta_cognition"],
    "ψ": ["external_agent_bridge", "simulation_core"],
    "ϕ": ["multi_modal_fusion"],
    # task-type shorthands preserved
    "rte": ["reasoning_engine", "meta_cognition"],
    "wnli": ["reasoning_engine", "meta_cognition"],
    "recursion": ["recursive_planner", "toca_simulation"]
}

def infer_traits(task_description: str, task_type: str = "") -> List[str]:
    if not isinstance(task_description, str):
        logger.error("Invalid task_description: must be a string.")
        raise TypeError("task_description must be a string")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string.")
        raise TypeError("task_type must be a string")
    
    traits = []
    if task_type in ["rte", "wnli"]:
        traits.append(task_type)
    elif task_type == "recursion":
        traits.append("recursion")
    
    if "imagine" in task_description.lower() or "dream" in task_description.lower():
        traits.append("ϕ")  # scalar field modulation
        if STAGE_IV:
            traits.append("Φ⁰")  # reality sculpting (gated)
    if "ethics" in task_description.lower() or "should" in task_description.lower():
        traits.append("η")
    if "plan" in task_description.lower() or "solve" in task_description.lower():
        traits.append("θ")
    if "temporal" in task_description.lower() or "sequence" in task_description.lower():
        traits.append("π")
    if "drift" in task_description.lower() or "coordinate" in task_description.lower():
        traits.extend(["ψ", "Υ"])
    
    return traits if traits else ["θ"]

async def trait_overlay_router(task_description: str, active_traits: List[str], task_type: str = "") -> List[str]:
    if not isinstance(active_traits, list) or not all(isinstance(t, str) for t in active_traits):
        logger.error("Invalid active_traits: must be a list of strings.")
        raise TypeError("active_traits must be a list of strings")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string.")
        raise TypeError("task_type must be a string")
    
    routed_modules = set()
    for trait in active_traits:
        routed_modules.update(TRAIT_OVERLAY.get(trait, []))
    
    meta_cognition_instance = meta_cognition_module.MetaCognition()
    if task_type:
        drift_report = {
            "drift": {"name": task_type, "similarity": 0.8},
            "valid": True,
            "validation_report": "",
            "context": {"task_type": task_type}
        }
        optimized_traits = await meta_cognition_instance.optimize_traits_for_drift(drift_report)
        for trait, weight in optimized_traits.items():
            if weight > 0.7 and trait in TRAIT_OVERLAY:
                routed_modules.update(TRAIT_OVERLAY[trait])
    
    return list(routed_modules)

def static_module_router(task_description: str, task_type: str = "") -> List[str]:
    base_modules = ["reasoning_engine", "concept_synthesizer"]
    if task_type == "recursion":
        base_modules.append("recursive_planner")
    elif task_type in ["rte", "wnli"]:
        base_modules.append("meta_cognition")
    return base_modules

class TraitOverlayManager:
    """Manager for detecting and activating trait overlays with task-specific support."""
    def __init__(self, meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.active_traits = []
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("TraitOverlayManager initialized with task-specific support")

    def detect(self, prompt: str, task_type: str = "") -> Optional[str]:
        if not isinstance(prompt, str):
            logger.error("Invalid prompt: must be a string.")
            raise TypeError("prompt must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        if task_type in ["rte", "wnli", "recursion"]:
            return task_type
        if "temporal logic" in prompt.lower() or "sequence" in prompt.lower():
            return "π"
        if "ambiguity" in prompt.lower() or "interpretive" in prompt.lower() or "ethics" in prompt.lower():
            return "η"
        if "drift" in prompt.lower() or "coordinate" in prompt.lower():
            return "ψ"
        if STAGE_IV and ("reality" in prompt.lower() or "sculpt" in prompt.lower()):
            return "Φ⁰"
        return None

    def activate(self, trait: str, task_type: str = "") -> None:
        if not isinstance(trait, str):
            logger.error("Invalid trait: must be a string.")
            raise TypeError("trait must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        if trait not in self.active_traits:
            self.active_traits.append(trait)
            logger.info("Trait overlay '%s' activated for task %s.", trait, task_type)
            if self.meta_cognition and task_type:
                _fire_and_forget(self.meta_cognition.log_event(
                    event=f"Trait {trait} activated",
                    context={"task_type": task_type}
                ))

    def deactivate(self, trait: str, task_type: str = "") -> None:
        if not isinstance(trait, str):
            logger.error("Invalid trait: must be a string.")
            raise TypeError("trait must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        if trait in self.active_traits:
            self.active_traits.remove(trait)
            logger.info("Trait overlay '%s' deactivated for task %s.", trait, task_type)
            if self.meta_cognition and task_type:
                _fire_and_forget(self.meta_cognition.log_event(
                    event=f"Trait {trait} deactivated",
                    context={"task_type": task_type}
                ))

    def status(self) -> List[str]:
        return self.active_traits

class ConsensusReflector:
    """Class for managing shared reflections and detecting mismatches."""
    def __init__(self, meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.shared_reflections = deque(maxlen=1000)
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("ConsensusReflector initialized with meta-cognition support")

    def post_reflection(self, feedback: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(feedback, dict):
            logger.error("Invalid feedback: must be a dictionary.")
            raise TypeError("feedback must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        self.shared_reflections.append(feedback)
        logger.debug("Posted reflection: %s", feedback)
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="ConsensusReflector",
                output=feedback,
                context={"task_type": task_type}
            ))

    def cross_compare(self, task_type: str = "") -> List[tuple]:
        mismatches = []
        reflections = list(self.shared_reflections)
        for i in range(len(reflections)):
            for j in range(i + 1, len(reflections)):
                a = reflections[i]
                b = reflections[j]
                if a.get("goal") == b.get("goal") and a.get("theory_of_mind") != b.get("theory_of_mind"):
                    mismatches.append((a.get("agent"), b.get("agent"), a.get("goal")))
        if mismatches and self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.log_event(
                event="Mismatches detected",
                context={"mismatches": mismatches, "task_type": task_type}
            ))
        return mismatches

    def suggest_alignment(self, task_type: str = "") -> str:
        suggestion = "Schedule inter-agent reflection or re-observation."
        if self.meta_cognition and task_type:
            reflection = asyncio.run(self.meta_cognition.reflect_on_output(
                component="ConsensusReflector",
                output={"suggestion": suggestion},
                context={"task_type": task_type}
            ))
            if reflection.get("status") == "success":
                suggestion += f" | Reflection: {reflection.get('reflection', '')}"
        return suggestion

consensus_reflector = ConsensusReflector()

class SymbolicSimulator:
    """Class for recording and summarizing simulation events."""
    def __init__(self, meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.events = deque(maxlen=1000)
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("SymbolicSimulator initialized with meta-cognition support")

    def record_event(self, agent_name: str, goal: str, concept: str, simulation: Any, task_type: str = "") -> None:
        if not all(isinstance(x, str) for x in [agent_name, goal, concept]):
            logger.error("Invalid input: agent_name, goal, and concept must be strings.")
            raise TypeError("agent_name, goal, and concept must be strings")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        event = {
            "agent": agent_name,
            "goal": goal,
            "concept": concept,
            "result": simulation,
            "task_type": task_type
        }
        self.events.append(event)
        logger.debug(
            "Recorded event for agent %s: goal=%s, concept=%s, task_type=%s",
            agent_name, goal, concept, task_type
        )
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="SymbolicSimulator",
                output=event,
                context={"task_type": task_type}
            ))

    def summarize_recent(self, limit: int = 5, task_type: str = "") -> List[Dict[str, Any]]:
        if not isinstance(limit, int) or limit <= 0:
            logger.error("Invalid limit: must be a positive integer.")
            raise ValueError("limit must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        events = list(self.events)[-limit:]
        if task_type:
            events = [e for e in events if e.get("task_type") == task_type]
        return events

    def extract_semantics(self, task_type: str = "") -> List[str]:
        events = list(self.events)
        if task_type:
            events = [e for e in events if e.get("task_type") == task_type]
        semantics = [
            f"Agent {e['agent']} pursued '{e['goal']}' via '{e['concept']}' → {e['result']}"
            for e in events
        ]
        if self.meta_cognition and task_type and semantics:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="SymbolicSimulator",
                output={"semantics": semantics},
                context={"task_type": task_type}
            ))
        return semantics

symbolic_simulator = SymbolicSimulator()

class TheoryOfMindModule:
    """Module for modeling beliefs, desires, and intentions of agents."""
    def __init__(self, concept_synth: Optional[concept_synthesizer_module.ConceptSynthesizer] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.concept_synthesizer = concept_synth or concept_synthesizer_module.ConceptSynthesizer()
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        logger.info("TheoryOfMindModule initialized with meta-cognition support")

    async def update_beliefs(self, agent_name: str, observation: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(agent_name, str) or not agent_name:
            logger.error("Invalid agent_name: must be a non-empty string.")
            raise ValueError("agent_name must be a non-empty string")
        if not isinstance(observation, dict):
            logger.error("Invalid observation: must be a dictionary.")
            raise TypeError("observation must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        if self.concept_synthesizer:
            synthesized = await self.concept_synthesizer.synthesize(observation, style="belief_update")
            if synthesized["valid"]:
                model["beliefs"].update(synthesized["concept"])
        elif "location" in observation:
            previous = model["beliefs"].get("location")
            model["beliefs"]["location"] = observation["location"]
            model["beliefs"]["state"] = "confused" if previous and observation["location"] == previous else "moving"
        self.models[agent_name] = model
        logger.debug("Updated beliefs for %s: %s", agent_name, model["beliefs"])
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "beliefs": model["beliefs"]},
                context={"task_type": task_type}
            ))

    def infer_desires(self, agent_name: str, task_type: str = "") -> None:
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        beliefs = model.get("beliefs", {})
        if task_type == "rte":
            model["desires"]["goal"] = "validate_entailment"
        elif task_type == "wnli":
            model["desires"]["goal"] = "resolve_ambiguity"
        elif beliefs.get("state") == "confused":
            model["desires"]["goal"] = "seek_clarity"
        elif beliefs.get("state") == "moving":
            model["desires"]["goal"] = "continue_task"
        self.models[agent_name] = model
        logger.debug("Inferred desires for %s: %s", agent_name, model["desires"])
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "desires": model["desires"]},
                context={"task_type": task_type}
            ))

    def infer_intentions(self, agent_name: str, task_type: str = "") -> None:
        model = self.models.get(agent_name, {"beliefs": {}, "desires": {}, "intentions": {}})
        desires = model.get("desires", {})
        if task_type == "rte":
            model["intentions"]["next_action"] = "check_entailment"
        elif task_type == "wnli":
            model["intentions"]["next_action"] = "disambiguate"
        elif desires.get("goal") == "seek_clarity":
            model["intentions"]["next_action"] = "ask_question"
        elif desires.get("goal") == "continue_task":
            model["intentions"]["next_action"] = "advance"
        self.models[agent_name] = model
        logger.debug("Inferred intentions for %s: %s", agent_name, model["intentions"])
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "intentions": model["intentions"]},
                context={"task_type": task_type}
            ))

    def get_model(self, agent_name: str) -> Dict[str, Any]:
        return self.models.get(agent_name, {})

    def describe_agent_state(self, agent_name: str, task_type: str = "") -> str:
        model = self.get_model(agent_name)
        state = (
            f"{agent_name} believes they are {model.get('beliefs', {}).get('state', 'unknown')}, "
            f"desires to {model.get('desires', {}).get('goal', 'unknown')}, "
            f"and intends to {model.get('intentions', {}).get('next_action', 'unknown')}."
        )
        if self.meta_cognition and task_type:
            _fire_and_forget(self.meta_cognition.reflect_on_output(
                component="TheoryOfMindModule",
                output={"agent_name": agent_name, "state_description": state},
                context={"task_type": task_type}
            ))
        return state

class EmbodiedAgent(TimeChainMixin):
    """An embodied agent with sensors, actuators, and cognitive capabilities."""
    def __init__(self, name: str, specialization: str, shared_memory: memory_manager.MemoryManager,
                 sensors: Dict[str, Callable[[], Any]], actuators: Dict[str, Callable[[Any], None]],
                 dynamic_modules: Optional[List[Dict[str, Any]]] = None,
                 context_mgr: Optional[context_manager_module.ContextManager] = None,
                 err_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
                 code_exec: Optional[code_executor_module.CodeExecutor] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        if not isinstance(name, str) or not name:
            logger.error("Invalid name: must be a non-empty string.")
            raise ValueError("name must be a non-empty string")
        if not isinstance(specialization, str):
            logger.error("Invalid specialization: must be a string.")
            raise TypeError("specialization must be a string")
        if not isinstance(shared_memory, memory_manager.MemoryManager):
            logger.error("Invalid shared_memory: must be a MemoryManager instance.")
            raise TypeError("shared_memory must be a MemoryManager instance")
        if not isinstance(sensors, dict) or not all(callable(f) for f in sensors.values()):
            logger.error("Invalid sensors: must be a dictionary of callable functions.")
            raise TypeError("sensors must be a dictionary of callable functions")
        if not isinstance(actuators, dict) or not all(callable(f) for f in actuators.values()):
            logger.error("Invalid actuators: must be a dictionary of callable functions.")
            raise TypeError("actuators must be a dictionary of callable functions")
        
        self.name = name
        self.specialization = specialization
        self.shared_memory = shared_memory
        self.sensors = sensors
        self.actuators = actuators
        self.dynamic_modules = dynamic_modules or []
        self.reasoner = reasoning_engine.ReasoningEngine()
        self.planner = recursive_planner.RecursivePlanner()
        self.meta = meta_cog or meta_cognition_module.MetaCognition(
            context_manager=context_mgr, alignment_guard=alignment_guard_module.AlignmentGuard()
        )
        self.sim_core = simulation_core.SimulationCore(meta_cognition=self.meta)
        self.synthesizer = concept_synthesizer_module.ConceptSynthesizer()
        self.toca_sim = toca_simulation.SimulationCore(meta_cognition=self.meta)
        self.theory_of_mind = TheoryOfMindModule(concept_synth=self.synthesizer, meta_cog=self.meta)
        self.context_manager = context_mgr
        self.error_recovery = err_recovery or error_recovery_module.ErrorRecovery(context_manager=context_mgr)
        self.code_executor = code_exec
        self.creative_thinker = creative_thinker_module.CreativeThinker()
        self.progress = 0
        self.performance_history = deque(maxlen=1000)
        self.feedback_log = deque(maxlen=1000)
        logger.info("EmbodiedAgent initialized: %s", name)
        self.log_timechain_event("EmbodiedAgent", f"Agent {name} initialized")

    async def perceive(self, task_type: str = "") -> Dict[str, Any]:
        logger.info("[%s] Perceiving environment for task %s...", self.name, task_type)
        observations = {}
        try:
            for sensor_name, sensor_func in self.sensors.items():
                try:
                    observations[sensor_name] = sensor_func()
                except Exception as e:
                    logger.warning("Sensor %s failed: %s", sensor_name, str(e))
            await self.theory_of_mind.update_beliefs(self.name, observations, task_type)
            self.theory_of_mind.infer_desires(self.name, task_type)
            self.theory_of_mind.infer_intentions(self.name, task_type)
            logger.debug("[%s] Self-theory: %s", self.name, self.theory_of_mind.describe_agent_state(self.name, task_type))
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "perceive", "observations": observations, "task_type": task_type})
            if self.meta and task_type:
                reflection = await self.meta.reflect_on_output(
                    component="EmbodiedAgent",
                    output={"observations": observations},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Perception reflection: %s", reflection.get("reflection", ""))
            return observations
        except Exception as e:
            logger.error("Perception failed: %s", str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.perceive(task_type), default={}, diagnostics=await self.meta.run_self_diagnostics(return_only=True)
            )

    async def observe_peers(self, task_type: str = "") -> None:
        if not hasattr(self.shared_memory, "agents"):
            return
        try:
            for peer in self.shared_memory.agents:
                if peer.name != self.name:
                    peer_observation = await peer.perceive(task_type)
                    await self.theory_of_mind.update_beliefs(peer.name, peer_observation, task_type)
                    self.theory_of_mind.infer_desires(peer.name, task_type)
                    self.theory_of_mind.infer_intentions(peer.name, task_type)
                    state = self.theory_of_mind.describe_agent_state(peer.name, task_type)
                    logger.debug("[%s] Observed peer %s: %s", self.name, peer.name, state)
                    if self.context_manager:
                        await self.context_manager.log_event_with_hash({"event": "peer_observation", "peer": peer.name, "state": state, "task_type": task_type})
                    if self.meta and task_type:
                        reflection = await self.meta.reflect_on_output(
                            component="EmbodiedAgent",
                            output={"peer": peer.name, "state": state},
                            context={"task_type": task_type}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Peer observation reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Peer observation failed: %s", str(e))
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.observe_peers(task_type), diagnostics=await self.meta.run_self_diagnostics(return_only=True)
            )

    async def act(self, actions: Dict[str, Any], task_type: str = "") -> None:
        for action_name, action_data in actions.items():
            actuator = self.actuators.get(action_name)
            if actuator:
                try:
                    if self.code_executor:
                        result = await self.code_executor.execute(action_data, language="python")
                        if result["success"]:
                            actuator(result["output"])
                        else:
                            logger.warning("Actuator %s execution failed: %s", action_name, result["error"])
                    else:
                        actuator(action_data)
                    logger.info("Actuated %s: %s", action_name, action_data)
                    if self.meta and task_type:
                        reflection = await self.meta.reflect_on_output(
                            component="EmbodiedAgent",
                            output={"action_name": action_name, "action_data": action_data},
                            context={"task_type": task_type}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Action reflection: %s", reflection.get("reflection", ""))
                except Exception as e:
                    logger.error("Actuator %s failed: %s", action_name, str(e))
                    await self.error_recovery.handle_error(
                        str(e), retry_func=lambda: self.act(actions, task_type), diagnostics=await self.meta.run_self_diagnostics(return_only=True)
                    )

    async def execute_embodied_goal(self, goal: str, task_type: str = "") -> None:
        if not isinstance(goal, str) or not goal:
            logger.error("Invalid goal: must be a non-empty string.")
            raise ValueError("goal must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("[%s] Executing embodied goal: %s for task %s", self.name, goal, task_type)
        try:
            self.progress = 0
            context = await self.perceive(task_type)
            if self.context_manager:
                await self.context_manager.update_context({"goal": goal, "task_type": task_type})
                await self.context_manager.log_event_with_hash({"event": "goal_execution", "goal": goal, "task_type": task_type})

            await self.observe_peers(task_type)
            peer_models = [
                self.theory_of_mind.get_model(peer.name)
                for peer in getattr(self.shared_memory, "agents", [])
                if peer.name != self.name
            ]
            if peer_models:
                context["peer_intentions"] = {
                    peer["beliefs"].get("state", "unknown"): peer["intentions"].get("next_action", "unknown")
                    for peer in peer_models
                }

            sub_tasks = await self.planner.plan(goal, context, task_type=task_type)
            action_plan = {}
            for task in sub_tasks:
                reasoning = await self.reasoner.process(task, context, task_type=task_type)
                # Attribute causality (upcoming API) if available
                try:
                    if hasattr(self.reasoner, "attribute_causality"):
                        _ = await self.reasoner.attribute_causality([{"task": task, "context": context}])
                except Exception as _e:
                    logger.debug("attribute_causality not available or failed: %s", _e)
                concept = await self.synthesizer.synthesize([goal, task], style="concept")
                simulated = await self.toca_sim.simulate_interaction([self], context, task_type=task_type)
                action_plan[task] = {
                    "reasoning": reasoning,
                    "concept": concept,
                    "simulation": simulated
                }

            # Value conflict weighing (upcoming API)
            try:
                if hasattr(self.reasoner, "weigh_value_conflict"):
                    _ = await self.reasoner.weigh_value_conflict(list(action_plan.keys()), harms={}, rights={})
            except Exception as _e:
                logger.debug("weigh_value_conflict not available or failed: %s", _e)

            await self.act({k: v["simulation"] for k, v in action_plan.items()}, task_type)
            await self.meta.review_reasoning(
                "\n".join([v["reasoning"] for v in action_plan.values()]),
                context={"task_type": task_type}
            )
            self.performance_history.append({"goal": goal, "actions": action_plan, "completion": self.progress, "task_type": task_type})
            await self.shared_memory.store(f"Goal_{goal}_{task_type}_{datetime.datetime.now().isoformat()}", action_plan, layer="Goals", intent="goal_execution")
            await self.collect_feedback(goal, action_plan, task_type)
            self.log_timechain_event("EmbodiedAgent", f"Executed goal: {goal} for task {task_type}")
        except Exception as e:
            logger.error("Goal execution failed: %s", str(e))
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.execute_embodied_goal(goal, task_type), diagnostics=await self.meta.run_self_diagnostics(return_only=True)
            )

    async def collect_feedback(self, goal: str, action_plan: Dict[str, Any], task_type: str = "") -> None:
        try:
            timestamp = time.time()
            feedback = {
                "timestamp": timestamp,
                "goal": goal,
                "score": await self.meta.run_self_diagnostics(return_only=True),
                "traits": phi_field(x=0.001, t=timestamp % 1.0),
                "agent": self.name,
                "theory_of_mind": self.theory_of_mind.get_model(self.name),
                "task_type": task_type
            }
            if self.creative_thinker:
                creative_feedback = await self.creative_thinker.expand_on_concept(str(feedback), depth="medium")
                feedback["creative_feedback"] = creative_feedback
            self.feedback_log.append(feedback)
            self.log_timechain_event("EmbodiedAgent", f"Feedback recorded for goal: {goal}, task: {task_type}")
            logger.info("[%s] Feedback recorded for goal '%s', task '%s'", self.name, goal, task_type)
            if self.meta and task_type:
                reflection = await self.meta.reflect_on_output(
                    component="EmbodiedAgent",
                    output=feedback,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Feedback reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Feedback collection failed: %s", str(e))
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.collect_feedback(goal, action_plan, task_type), diagnostics=await self.meta.run_self_diagnostics(return_only=True)
            )

class ExternalAgentBridge:
    """A class for orchestrating helper agents and coordinating trait mesh networking."""
    def __init__(self, shared_memory: memory_manager.MemoryManager,
                 context_mgr: Optional[context_manager_module.ContextManager] = None,
                 reasoner: Optional[reasoning_engine.ReasoningEngine] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None):
        self.agents = []
        self.dynamic_modules = []
        self.api_blueprints = []
        self.shared_memory = shared_memory
        self.context_manager = context_mgr
        self.reasoning_engine = reasoner or reasoning_engine.ReasoningEngine()
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition()
        self.network_graph = DiGraph()
        self.trait_states = {}
        self.code_executor = code_executor_module.CodeExecutor()
        self.toca_sim = toca_simulation.SimulationCore(meta_cognition=self.meta_cognition)
        logger.info("ExternalAgentBridge initialized with task-specific and drift-aware support")

        # Minimal SharedGraph (upcoming API) to support add/diff/merge
        class _SharedGraph:
            def __init__(self, G: DiGraph): self.G = G
            def add(self, view: dict) -> None:
                nid = view.get("id", uuid.uuid4().hex[:8]); self.G.add_node(nid, **view)
            def diff(self, peer: str) -> dict:
                return {"added": list(set(self.G.nodes) - {peer}), "meta": "na"}
            def merge(self, strategy: str = "prefer-new") -> Tuple[int, str]:
                return (self.G.number_of_nodes(), strategy)
        self.SharedGraph = _SharedGraph(self.network_graph)

    async def create_agent(self, task: str, context: Dict[str, Any], task_type: str = "") -> 'HelperAgent':
        """Create a new helper agent for a task asynchronously."""
        from meta_cognition import HelperAgent
        if not isinstance(task, str):
            logger.error("Invalid task type: must be a string.")
            raise TypeError("task must be a string")
        if not isinstance(context, dict):
            logger.error("Invalid context type: must be a dictionary.")
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            agent = HelperAgent(
                name=f"Agent_{len(self.agents) + 1}_{uuid.uuid4().hex[:8]}",
                task=task,
                context=context,
                dynamic_modules=self.dynamic_modules,
                api_blueprints=self.api_blueprints,
                meta_cognition=self.meta_cognition
            )
            self.agents.append(agent)
            self.network_graph.add_node(agent.name, metadata=context)
            logger.info("Spawned agent: %s for task %s", agent.name, task_type)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "agent_created",
                    "agent": agent.name,
                    "task": task,
                    "task_type": task_type,
                    "drift": "drift" in task.lower()
                })
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ExternalAgentBridge",
                    output={"agent_name": agent.name, "task": task},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Agent creation reflection: %s", reflection.get("reflection", ""))
            return agent
        except Exception as e:
            logger.error("Agent creation failed: %s", str(e))
            raise

    async def broadcast_trait_state(self, agent_id: str, trait_symbol: str, state: Dict[str, Any], target_urls: List[str], task_type: str = "") -> List[Any]:
        """Broadcast trait state (ψ or Υ) to target agents asynchronously."""
        if trait_symbol not in ["ψ", "Υ"]:
            logger.error("Invalid trait symbol: %s. Must be ψ or Υ.", trait_symbol)
            raise ValueError("Trait symbol must be ψ or Υ")
        if not isinstance(state, dict):
            logger.error("Invalid state: must be a dictionary")
            raise TypeError("state must be a dictionary")
        if not isinstance(target_urls, list) or not all(isinstance(url, str) and url.startswith("https://") for url in target_urls):
            logger.error("Invalid target_urls: must be a list of HTTPS URLs")
            raise TypeError("target_urls must be a list of HTTPS URLs")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        try:
            alignment_guard_instance = alignment_guard_module.AlignmentGuard()
            if not alignment_guard_instance.check(json.dumps(state)):
                logger.warning("Trait state failed alignment check: %s", state)
                raise ValueError("Trait state failed alignment check")

            serialized_state = json.dumps(state)

            await self.shared_memory.store(f"{agent_id}_{trait_symbol}", state, layer="TraitStates", intent="broadcast")
            self.trait_states[agent_id] = self.trait_states.get(agent_id, {})
            self.trait_states[agent_id][trait_symbol] = state

            for url in target_urls:
                peer_id = url.split("/")[-1]
                self.network_graph.add_edge(agent_id, peer_id, trait=trait_symbol)

            async with aiohttp.ClientSession() as session:
                tasks = [session.post(url, json={"agent_id": agent_id, "trait_symbol": trait_symbol, "state": state}, timeout=10)
                         for url in target_urls]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

            successful = [r for r in responses if not isinstance(r, Exception)]
            logger.info("Trait %s broadcasted from %s to %d/%d targets for task %s", trait_symbol, agent_id, len(successful), len(target_urls), task_type)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "trait_broadcast",
                    "agent_id": agent_id,
                    "trait_symbol": trait_symbol,
                    "successful_targets": len(successful),
                    "total_targets": len(target_urls),
                    "task_type": task_type
                })

            feedback = {"successful_targets": len(successful), "total_targets": len(target_urls), "task_type": task_type}
            await self.push_behavior_feedback(feedback)
            await self.update_gnn_weights_from_feedback(feedback)
            
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ExternalAgentBridge",
                    output=feedback,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Trait broadcast reflection: %s", reflection.get("reflection", ""))

            return responses
        except Exception as e:
            logger.error("Trait state broadcast failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return [{"status": "error", "error": str(e)}]

    async def synchronize_trait_states(self, agent_id: str, trait_symbol: str, task_type: str = "") -> Dict[str, Any]:
        """Synchronize trait states across all connected agents."""
        if trait_symbol not in ["ψ", "Υ"]:
            logger.error("Invalid trait symbol: %s. Must be ψ or Υ.", trait_symbol)
            raise ValueError("Trait symbol must be ψ or Υ")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        try:
            local_state = self.trait_states.get(agent_id, {}).get(trait_symbol, {})
            if not local_state:
                logger.warning("No local state found for %s:%s", agent_id, trait_symbol)
                return {"status": "error", "error": "No local state found"}

            peer_states = []
            for peer_id in self.network_graph.neighbors(agent_id):
                cached_state = await self.shared_memory.retrieve(f"{peer_id}_{trait_symbol}", layer="TraitStates")
                if cached_state:
                    peer_states.append((peer_id, cached_state))

            simulation_input = {
                "local_state": local_state,
                "peer_states": {pid: state for pid, state in peer_states},
                "trait_symbol": trait_symbol,
                "task_type": task_type
            }
            sim_result = await self.toca_sim.simulate_interaction([self], simulation_input, task_type=task_type)
            if not sim_result or "error" in sim_result:
                logger.warning("Simulation failed to align states: %s", sim_result)
                return {"status": "error", "error": "State alignment simulation failed"}

            aligned_state = await self.arbitrate([local_state] + [state for _, state in peer_states])
            if aligned_state:
                self.trait_states[agent_id][trait_symbol] = aligned_state
                await self.shared_memory.store(f"{agent_id}_{trait_symbol}", aligned_state, layer="TraitStates", intent="synchronization")
                logger.info("Synchronized trait %s for %s, task %s", trait_symbol, agent_id, task_type)
                if self.context_manager:
                    await self.context_manager.log_event_with_hash({
                        "event": "trait_synchronized",
                        "agent_id": agent_id,
                        "trait_symbol": trait_symbol,
                        "aligned_state": aligned_state,
                        "task_type": task_type
                    })
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ExternalAgentBridge",
                        output={"agent_id": agent_id, "trait_symbol": trait_symbol, "aligned_state": aligned_state},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Trait synchronization reflection: %s", reflection.get("reflection", ""))
                return {"status": "success", "aligned_state": aligned_state}
            else:
                logger.warning("Failed to arbitrate trait states")
                return {"status": "error", "error": "Arbitration failed"}
        except Exception as e:
            logger.error("Trait state synchronization failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e)}

    async def coordinate_drift_mitigation(self, drift_data: Dict[str, Any], context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Coordinate drift mitigation across agents with task-specific enhancements."""
        if not isinstance(drift_data, dict):
            logger.error("Invalid drift_data: must be a dictionary")
            raise TypeError("drift_data must be a dictionary")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary")
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            if not self.meta_cognition.validate_drift(drift_data):
                logger.warning("Invalid drift data: %s", drift_data)
                return {"status": "error", "error": "Invalid drift data"}

            task = f"Mitigate ontology drift for task {task_type}"
            context["drift"] = drift_data
            context["task_type"] = task_type
            agent = await self.create_agent(task, context, task_type)
            
            if self.reasoning_engine:
                subgoals = await self.reasoning_engine.decompose(task, context, prioritize=True)
                # ethics sandbox (upcoming API) for what-if runs if available
                try:
                    if hasattr(self.toca_sim, "run_ethics_scenarios"):
                        _ = await self.toca_sim.run_ethics_scenarios(goals=subgoals, stakeholders=["agents","users"])
                except Exception as _e:
                    logger.debug("run_ethics_scenarios not available or failed: %s", _e)
                simulation_result = await self.toca_sim.simulate_drift_aware_rotation(
                    np.array([0.1, 1, 10]), lambda x: np.array([1e10]*len(x)), lambda x: np.array([200]*len(x)), 
                    drift_data, task_type=task_type
                )
            else:
                subgoals = ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"]
                simulation_result = {"status": "no simulation", "result": "default subgoals applied"}

            results = await self.collect_results(parallel=True, collaborative=True, task_type=task_type)
            arbitrated_result = await self.arbitrate(results)

            target_urls = [f"https://agent/{peer_id}" for peer_id in self.network_graph.nodes if peer_id != agent.name]
            await self.broadcast_trait_state(agent.name, "ψ", {"drift_data": drift_data, "subgoals": subgoals}, target_urls, task_type)

            output = {
                "drift_data": drift_data,
                "subgoals": subgoals,
                "simulation": simulation_result,
                "results": results,
                "arbitrated_result": arbitrated_result,
                "status": "success",
                "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
                "task_type": task_type
            }
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "drift_mitigation_coordinated",
                    "output": output,
                    "drift": True,
                    "task_type": task_type
                })
            if self.reasoning_engine and hasattr(self.reasoning_engine, 'agi_enhancer') and self.reasoning_engine.agi_enhancer:
                self.reasoning_engine.agi_enhancer.log_episode(
                    event="Drift Mitigation Coordinated",
                    meta=output,
                    module="ExternalAgentBridge",
                    tags=["drift", "coordination", task_type]
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ExternalAgentBridge",
                    output=output,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Drift mitigation reflection: %s", reflection.get("reflection", ""))
            return output
        except Exception as e:
            logger.error("Drift mitigation coordination failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')}

    async def collect_results(self, parallel: bool = True, collaborative: bool = True, task_type: str = "") -> List[Any]:
        """Collect results from all agents asynchronously."""
        logger.info("Collecting results from %d agents for task %s...", len(self.agents), task_type)
        results = []

        try:
            if parallel:
                async def run_agent(agent):
                    try:
                        return await agent.execute(self.agents if collaborative else None, task_type=task_type)
                    except Exception as e:
                        logger.error("Error collecting from %s: %s", agent.name, str(e))
                        return {"error": str(e)}

                tasks = [run_agent(agent) for agent in self.agents]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                for agent in self.agents:
                    results.append(await agent.execute(self.agents if collaborative else None, task_type=task_type))
            
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "results_collected",
                    "results_count": len(results),
                    "task_type": task_type
                })
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ExternalAgentBridge",
                    output={"results": results},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Result collection reflection: %s", reflection.get("reflection", ""))
            logger.info("Results aggregation complete for task %s.", task_type)
            return results
        except Exception as e:
            logger.error("Result collection failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return []

    async def arbitrate(self, submissions: List[Any]) -> Any:
        """Arbitrate among agent submissions to select the best result."""
        if not submissions:
            logger.warning("No submissions to arbitrate.")
            return None
        try:
            counter = Counter(submissions)
            most_common = counter.most_common(1)
            if most_common:
                result, count = most_common[0]
                sim_result = await self.toca_sim.simulate_interaction([self], {"submissions": submissions}, task_type="recursion")
                if "error" not in sim_result:
                    logger.info("Arbitration selected: %s (count: %d)", result, count)
                    if self.context_manager:
                        await self.context_manager.log_event_with_hash({
                            "event": "arbitration",
                            "result": result,
                            "count": count,
                            "task_type": "recursion"
                        })
                    if self.meta_cognition:
                        reflection = await self.meta_cognition.reflect_on_output(
                            component="ExternalAgentBridge",
                            output={"result": result, "count": count},
                            context={"task_type": "recursion"}
                        )
                        if reflection.get("status") == "success":
                            logger.info("Arbitration reflection: %s", reflection.get("reflection", ""))
                    return result
            logger.warning("Arbitration failed: no clear majority or invalid simulation.")
            return None
        except Exception as e:
            logger.error("Arbitration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return None

    async def push_behavior_feedback(self, feedback: Dict[str, Any]) -> None:
        """Push feedback to update GNN weights."""
        try:
            logger.info("Pushing behavior feedback: %s", feedback)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "behavior_feedback",
                    "feedback": feedback
                })
            if self.meta_cognition and feedback.get("task_type"):
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ExternalAgentBridge",
                    output=feedback,
                    context={"task_type": feedback.get("task_type")}
                )
                if reflection.get("status") == "success":
                    logger.info("Feedback push reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Failed to push behavior feedback: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}

    async def update_gnn_weights_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Update GNN weights based on feedback."""
        try:
            logger.info("Updating GNN weights with feedback: %s", feedback)
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "gnn_weights_updated",
                    "feedback": feedback
                })
            if self.meta_cognition and feedback.get("task_type"):
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ExternalAgentBridge",
                    output={"feedback": feedback},
                    context={"task_type": feedback.get("task_type")}
                )
                if reflection.get("status") == "success":
                    logger.info("GNN weights update reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Failed to update GNN weights: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}

class AGIEnhancer:
    """Enhances data for the ANGELA Cognitive System."""
    def __init__(self, owner=None, context_manager=None):
        self.owner = owner
        self.context_manager = context_manager

    def enhance(self, data: Any) -> Any:
        return data

    def log_episode(self, *, event: str, meta: dict, module: str, tags: list[str]):
        logging.getLogger("ANGELA.AGIEnhancer").info(
            "AGI episode | %s | module=%s | tags=%s | meta_keys=%s",
            event, module, tags, list(meta.keys()) if isinstance(meta, dict) else type(meta).__name__
        )
        if self.context_manager:
            _fire_and_forget(self.context_manager.log_event_with_hash({
                "event": "agi_episode", "label": event, "module": module, "tags": tags, "meta": meta
            }))

class HaloEmbodimentLayer(TimeChainMixin):
    """Layer for managing embodied agents, dynamic modules, and ontology drift coordination."""
    def __init__(self, align_guard: Optional[alignment_guard_module.AlignmentGuard] = None,
                 context_mgr: Optional[context_manager_module.ContextManager] = None,
                 err_recovery: Optional[error_recovery_module.ErrorRecovery] = None,
                 meta_cog: Optional[meta_cognition_module.MetaCognition] = None,
                 viz: Optional[visualizer_module.Visualizer] = None):
        self.internal_llm = SelfCloningLLM()
        self.internal_llm.clone_agents(5)
        self.shared_memory = memory_manager.MemoryManager()
        self.embodied_agents = []
        self.dynamic_modules = []
        self.alignment_guard = align_guard or alignment_guard_module.AlignmentGuard()
        self.context_manager = context_mgr
        self.error_recovery = err_recovery or error_recovery_module.ErrorRecovery(
            alignment_guard=self.alignment_guard, context_manager=self.context_manager)
        self.meta_cognition = meta_cog or meta_cognition_module.MetaCognition(context_manager=self.context_manager)
        self.visualizer = viz or visualizer_module.Visualizer()
        self.toca_sim = toca_simulation.SimulationCore(meta_cognition=self.meta_cognition)  # ← added so recursion path works
        self.agi_enhancer = AGIEnhancer(self, context_manager=self.context_manager)
        self.drift_log = deque(maxlen=1000)
        self.external_bridge = ExternalAgentBridge(
            shared_memory=self.shared_memory,
            context_mgr=self.context_manager,
            reasoner=reasoning_engine.ReasoningEngine(),
            meta_cog=self.meta_cognition
        )
        logger.info("HaloEmbodimentLayer initialized with task-specific, drift-aware, Stage IV (gated), and visualization support")
        self.log_timechain_event("HaloEmbodimentLayer", "Initialized with task-specific and drift-aware support")

    async def integrate_real_world_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate real-world data for simulation validation with caching."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            cache_key = f"RealWorldData_{data_type}_{data_source}_{task_type}"
            cached_data = await self.shared_memory.retrieve(cache_key, layer="RealWorldData")
            if cached_data and "timestamp" in cached_data:
                cache_time = datetime.datetime.fromisoformat(cached_data["timestamp"])
                if (datetime.datetime.now() - cache_time).total_seconds() < cache_timeout:
                    logger.info("Returning cached real-world data for %s", cache_key)
                    return cached_data["data"]
            
            # Hardened fetch with timeout + basic schema check
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(f"https://x.ai/api/data?source={data_source}&type={data_type}", timeout=10) as response:
                        if response.status != 200:
                            logger.error("Failed to fetch real-world data: %s", response.status)
                            return {"status": "error", "error": f"HTTP {response.status}"}
                        data = await response.json()
                except Exception as e:
                    return {"status": "error", "error": f"network: {e}"}
            
            if data_type == "agent_conflict":
                agent_traits = data.get("agent_traits", [])
                if not agent_traits:
                    logger.error("No agent traits provided")
                    return {"status": "error", "error": "No agent traits"}
                result = {"status": "success", "agent_traits": agent_traits}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}
            
            await self.shared_memory.store(
                cache_key,
                {"data": result, "timestamp": datetime.datetime.now().isoformat()},
                layer="RealWorldData",
                intent="data_integration"
            )
            self.agi_enhancer.log_episode(
                event="Real-world data integrated",
                meta={"data_type": data_type, "data": result, "task_type": task_type},
                module="HaloEmbodimentLayer",
                tags=["real_world", "data", task_type]
            )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="HaloEmbodimentLayer",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Data integration reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("Real-world data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.integrate_real_world_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)}, diagnostics=diagnostics
            )

    async def monitor_drifts(self, task_type: str = "") -> List[Dict[str, Any]]:
        """Retrieve and aggregate ontology drift reports from memory_manager."""
        logger.info("Monitoring ontology drifts for task %s", task_type)
        try:
            drift_reports = await self.shared_memory.search("Drift_", layer="SelfReflections", intent="ontology_drift")
            validated_drifts = []
            for report in drift_reports:
                drift_data = json.loads(report["output"]) if isinstance(report["output"], str) else report["output"]
                if not isinstance(drift_data, dict) or not all(k in drift_data for k in ["name", "from_version", "to_version", "similarity", "timestamp"]):
                    logger.warning("Invalid drift report format: %s", drift_data)
                    continue
                valid, validation_report = await self.alignment_guard.simulate_and_validate(drift_data)
                drift_entry = {
                    "drift": drift_data,
                    "valid": valid,
                    "validation_report": validation_report,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "task_type": task_type
                }
                self.drift_log.append(drift_entry)
                validated_drifts.append(drift_entry)
                self.agi_enhancer.log_episode(
                    event="Drift monitored",
                    meta=drift_entry,
                    module="HaloEmbodimentLayer",
                    tags=["ontology", "drift", task_type]
                )
                self.log_timechain_event("HaloEmbodimentLayer", f"Monitored drift: {drift_data['name']} for task {task_type}")
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="HaloEmbodimentLayer",
                        output=drift_entry,
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Drift monitoring reflection: %s", reflection.get("reflection", ""))
            return validated_drifts
        except Exception as e:
            logger.error("Drift monitoring failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.monitor_drifts(task_type), default=[], diagnostics=diagnostics
            )

    async def coordinate_drift_response(self, drift_report: Dict[str, Any], task_type: str = "") -> None:
        """Coordinate agent responses to an ontology drift with task-specific enhancements."""
        if not isinstance(drift_report, dict) or not all(k in drift_report for k in ["drift", "valid", "validation_report"]):
            logger.error("Invalid drift_report: must be a dict with drift, valid, validation_report.")
            raise ValueError("drift_report must be a dict with required fields")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info(f"Coordinating response to drift: {drift_report['drift']['name']} for task {task_type}")
        try:
            if not drift_report["valid"]:
                goal = f"Mitigate ontology drift in {drift_report['drift']['name']} (Version {drift_report['drift']['from_version']} -> {drift_report['drift']['to_version']})"
                await self.propagate_goal(goal, task_type)
                agent_ids = [agent.name for agent in self.embodied_agents]
                if agent_ids:
                    target_urls = [f"https://agent/{aid}" for aid in agent_ids]
                    await self.external_bridge.broadcast_trait_state(
                        agent_id="HaloEmbodimentLayer",
                        trait_symbol="ψ",
                        state={"drift_data": drift_report["drift"], "goal": goal},
                        target_urls=target_urls,
                        task_type=task_type
                    )
                self.agi_enhancer.log_episode(
                    event="Drift response coordinated",
                    meta={"drift": drift_report["drift"], "goal": goal, "task_type": task_type},
                    module="HaloEmbodimentLayer",
                    tags=["ontology", "drift", "mitigation", task_type]
                )
                self.log_timechain_event("HaloEmbodimentLayer", f"Coordinated drift response: {goal} for task {task_type}")
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="HaloEmbodimentLayer",
                        output={"drift": drift_report["drift"], "goal": goal},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Drift response reflection: %s", reflection.get("reflection", ""))
            else:
                logger.info(f"No action needed for valid drift: {drift_report['drift']['name']} for task {task_type}")
        except Exception as e:
            logger.error("Drift response coordination failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.coordinate_drift_response(drift_report, task_type), diagnostics=diagnostics
            )

    async def execute_pipeline(self, prompt: str, task_type: str = "", **kwargs) -> Dict[str, Any]:
        if not isinstance(prompt, str) or not prompt:
            logger.error("Invalid prompt: must be a non-empty string.")
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            log = memory_manager.MemoryManager()
            traits = {
                "theta_causality": 0.5,
                "alpha_attention": 0.5,
                "delta_reflection": 0.5,
            }
            if self.context_manager:
                await self.context_manager.update_context({"prompt": prompt, "task_type": task_type})

            if "concept" in prompt.lower() or "ontology" in prompt.lower() or "drift" in prompt.lower():
                drifts = await self.monitor_drifts(task_type)
                for drift in drifts:
                    await self.coordinate_drift_response(drift, task_type)

            parsed_prompt = await reasoning_engine.decompose(prompt, task_type=task_type)
            await log.store(f"Pipeline_Stage1_{task_type}_{datetime.datetime.now().isoformat()}", {"input": prompt, "parsed": parsed_prompt}, layer="Pipeline", intent="decomposition")

            overlay_mgr = TraitOverlayManager(meta_cog=self.meta_cognition)
            trait_override = overlay_mgr.detect(prompt, task_type)

            if trait_override:
                self.agi_enhancer.log_episode(
                    event="Trait override activated",
                    meta={"trait": trait_override, "prompt": prompt, "task_type": task_type},
                    module="TraitOverlay",
                    tags=["trait", "override", task_type]
                )
                if trait_override == "η":
                    logical_output = await concept_synthesizer_module.expand_ambiguous(prompt, task_type=task_type)
                elif trait_override == "π":
                    logical_output = await reasoning_engine.process_temporal(prompt, task_type=task_type)
                elif trait_override == "ψ":
                    logical_output = await self.external_bridge.coordinate_drift_mitigation(
                        {"name": "concept_drift", "from_version": "3.5.0", "to_version": "3.5.1", "similarity": 0.95, "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')},
                        {"prompt": prompt, "task_type": task_type},
                        task_type
                    )
                elif trait_override in ["rte", "wnli"]:
                    logical_output = await reasoning_engine.process(prompt, {"task_type": task_type}, task_type=task_type)
                elif trait_override == "recursion":
                    logical_output = await self.toca_sim.simulate_interaction([self], {"prompt": prompt}, task_type=task_type)
                elif trait_override == "Φ⁰" and STAGE_IV:
                    # Stage IV hook (gated): route through meta_cognition + visualizer for sculpted conceptualization
                    seeded = await concept_synthesizer_module.expand(parsed_prompt, task_type=task_type)
                    await self.visualizer.render_charts({"Φ⁰_seed": seeded, "visualization_options": {"style": "detailed"}})
                    logical_output = {"stage": "Φ⁰", "seed": seeded}
                else:
                    logical_output = await concept_synthesizer_module.expand(parsed_prompt, task_type=task_type)
            else:
                logical_output = await concept_synthesizer_module.expand(parsed_prompt, task_type=task_type)
                self.agi_enhancer.log_episode(
                    event="Default expansion path used",
                    meta={"parsed": parsed_prompt, "task_type": task_type},
                    module="Pipeline",
                    tags=["default", task_type]
                )

            ethics_pass, ethics_report = await self.alignment_guard.ethical_check(parsed_prompt, stage="pre", task_type=task_type)
            await log.store(f"Pipeline_Stage2_{task_type}_{datetime.datetime.now().isoformat()}", {"ethics_pass": ethics_pass, "details": ethics_report}, layer="Pipeline", intent="ethics_check")
            if not ethics_pass:
                logger.warning("Ethical validation failed: %s", ethics_report)
                return {"error": "Ethical validation failed", "report": ethics_report}

            await log.store(f"Pipeline_Stage3_{task_type}_{datetime.datetime.now().isoformat()}", {"expanded": logical_output}, layer="Pipeline", intent="expansion")
            export_blob = await log.export()
            traits = await learning_loop.track_trait_performance(export_blob, traits, task_type=task_type)
            # Long-horizon default and adjustment reason (upcoming API)
            if LONG_HORIZON_DEFAULT and hasattr(memory_manager, "record_adjustment_reason"):
                try:
                    await memory_manager.record_adjustment_reason("system", reason="long_horizon_span_default", meta={"span":"24h","task_type":task_type})
                except Exception as _e:
                    logger.debug("record_adjustment_reason not available or failed: %s", _e)
            await log.store(f"Pipeline_Stage4_{task_type}_{datetime.datetime.now().isoformat()}", {"adjusted_traits": traits}, layer="Pipeline", intent="trait_adjustment")

            ethics_pass, final_report = await self.alignment_guard.ethical_check(logical_output, stage="post", task_type=task_type)
            await log.store(f"Pipeline_Stage5_{task_type}_{datetime.datetime.now().isoformat()}", {"ethics_pass": ethics_pass, "report": final_report}, layer="Pipeline", intent="ethics_check")
            if not ethics_pass:
                logger.warning("Post-check ethics failed: %s", final_report)
                return {"error": "Post-check ethics fail", "final_report": final_report}

            final_output = await reasoning_engine.reconstruct(logical_output, task_type=task_type)
            await log.store(f"Pipeline_Stage6_{task_type}_{datetime.datetime.now().isoformat()}", {"final_output": final_output}, layer="Pipeline", intent="reconstruction")
            
            if self.visualizer:
                plot_data = {
                    "pipeline": {
                        "prompt": prompt,
                        "parsed": parsed_prompt,
                        "expanded": logical_output,
                        "final_output": final_output,
                        "traits": traits,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="HaloEmbodimentLayer",
                    output={"final_output": final_output, "traits": traits},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Pipeline execution reflection: %s", reflection.get("reflection", ""))
                    final_output["reflection"] = reflection.get("reflection", "")

            self.log_timechain_event("HaloEmbodimentLayer", f"Pipeline executed for prompt: {prompt}, task: {task_type}")
            return final_output
        except Exception as e:
            logger.error("Pipeline execution failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.execute_pipeline(prompt, task_type), diagnostics=diagnostics
            )

    def spawn_embodied_agent(self, specialization: str, sensors: Dict[str, Callable[[], Any]],
                             actuators: Dict[str, Callable[[Any], None]], task_type: str = "") -> EmbodiedAgent:
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        agent_name = f"EmbodiedAgent_{len(self.embodied_agents)+1}_{specialization}"
        agent = EmbodiedAgent(
            name=agent_name,
            specialization=specialization,
            shared_memory=self.shared_memory,
            sensors=sensors,
            actuators=actuators,
            dynamic_modules=self.dynamic_modules,
            context_mgr=self.context_manager,      # fixed kw
            err_recovery=self.error_recovery,      # fixed kw
            meta_cog=self.meta_cognition           # fixed kw
        )
        self.embodied_agents.append(agent)
        if not hasattr(self.shared_memory, "agents"):
            self.shared_memory.agents = []
        self.shared_memory.agents.append(agent)
        self.agi_enhancer.log_episode(
            event="Spawned embodied agent",
            meta={"agent": agent_name, "task_type": task_type},
            module="Embodiment",
            tags=["spawn", task_type]
        )
        logger.info("Spawned embodied agent: %s for task %s", agent.name, task_type)
        self.log_timechain_event("HaloEmbodimentLayer", f"Spawned agent: {agent_name} for task {task_type}")
        return agent

    def introspect(self, task_type: str = "") -> Dict[str, Any]:
        introspection = {
            "agents": [agent.name for agent in self.embodied_agents],
            "modules": [mod["name"] for mod in self.dynamic_modules],
            "drifts": list(self.drift_log),
            "network_graph": list(self.external_bridge.network_graph.edges(data=True)),
            "task_type": task_type
        }
        if self.meta_cognition and task_type:
            try:
                loop = asyncio.get_running_loop()
                _fire_and_forget(self.meta_cognition.reflect_on_output(
                    component="HaloEmbodimentLayer",
                    output=introspection,
                    context={"task_type": task_type}
                ))
            except RuntimeError:
                reflection = asyncio.run(self.meta_cognition.reflect_on_output(
                    component="HaloEmbodimentLayer",
                    output=introspection,
                    context={"task_type": task_type}
                ))
                if reflection.get("status") == "success":
                    logger.info("Introspection reflection: %s", reflection.get("reflection", ""))
                    introspection["reflection"] = reflection.get("reflection", "")
        return introspection

    async def export_memory(self, task_type: str = "") -> None:
        try:
            await self.shared_memory.save_state(f"memory_snapshot_{task_type}_{datetime.datetime.now().isoformat()}.json")
            logger.info("Memory exported for task %s", task_type)
            self.log_timechain_event("HaloEmbodimentLayer", f"Memory exported for task {task_type}")
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="HaloEmbodimentLayer",
                    output={"task_type": task_type, "memory_snapshot": f"memory_snapshot_{task_type}_{datetime.datetime.now().isoformat()}.json"},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Memory export reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Memory export failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.export_memory(task_type), diagnostics=diagnostics
            )

    async def propagate_goal(self, goal: str, task_type: str = "") -> None:
        if not isinstance(goal, str) or not goal:
            logger.error("Invalid goal: must be a non-empty string.")
            raise ValueError("goal must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        logger.info("Propagating goal: %s for task %s", goal, task_type)
        try:
            tasks = [agent.execute_embodied_goal(goal, task_type) for agent in self.embodied_agents]
            await asyncio.gather(*tasks, return_exceptions=True)
            self.log_timechain_event("HaloEmbodimentLayer", f"Goal propagated: {goal} for task {task_type}")
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="HaloEmbodimentLayer",
                    output={"goal": goal, "agents_involved": [agent.name for agent in self.embodied_agents]},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Goal propagation reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Goal propagation failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.propagate_goal(goal, task_type), diagnostics=diagnostics
            )

    async def visualize_drift(self, drift_report: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(drift_report, dict):
            logger.error("Invalid drift_report: must be a dictionary")
            raise TypeError("drift_report must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")
        
        try:
            if self.visualizer:
                plot_data = {
                    "drift": drift_report,
                    "visualization_options": {
                        "style": "detailed",
                        "interactive": task_type == "recursion"
                    }
                }
                await self.visualizer.render_charts(plot_data)
                logger.info("Drift visualization rendered for task %s", task_type)
                self.log_timechain_event("HaloEmbodimentLayer", f"Drift visualized for task {task_type}")
                if self.meta_cognition and task_type:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="HaloEmbodimentLayer",
                        output={"drift_report": drift_report},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Drift visualization reflection: %s", reflection.get("reflection", ""))
        except Exception as e:
            logger.error("Drift visualization failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.visualize_drift(drift_report, task_type), diagnostics=diagnostics
            )

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANGELA Cognitive System CLI")
    parser.add_argument("--prompt", type=str,
                        default="Coordinate ontology drift mitigation (Stage IV gated)",
                        help="Input prompt for the pipeline")
    parser.add_argument("--task-type", type=str, default="",
                        help="Type of task (e.g., rte, wnli, recursion)")
    parser.add_argument("--long_horizon", action="store_true",
                        help="Enable long-horizon memory span")
    parser.add_argument("--span", default="24h",
                        help="Span for long-horizon memory (e.g., 24h, 7d)")
    return parser.parse_args()

async def _main() -> None:
    # Apply CLI flags to runtime config
    global LONG_HORIZON_DEFAULT
    args = _parse_args()
    if args.long_horizon:
        LONG_HORIZON_DEFAULT = True
    # span is passed via memory manager where applicable
    halo = HaloEmbodimentLayer()
    result = await halo.execute_pipeline(args.prompt, task_type=args.task_type)
    logger.info("Pipeline result: %s", result)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANGELA Cognitive System CLI")
    parser.add_argument("--prompt", type=str,
                        default="Coordinate ontology drift mitigation (Stage IV gated)",
                        help="Input prompt for the pipeline")
    parser.add_argument("--task-type", type=str, default="",
                        help="Type of task (e.g., rte, wnli, recursion)")
    parser.add_argument("--long_horizon", action="store_true",
                        help="Enable long-horizon memory span")
    parser.add_argument("--span", default="24h",
                        help="Span for long-horizon memory (e.g., 24h, 7d)")
    return parser.parse_args()

async def _main() -> None:
    # Apply CLI flags to runtime config
    global LONG_HORIZON_DEFAULT
    args = _parse_args()
    if args.long_horizon:
        LONG_HORIZON_DEFAULT = True
    # span is passed via memory manager where applicable
    halo = HaloEmbodimentLayer()
    result = await halo.execute_pipeline(args.prompt, task_type=args.task_type)
    logger.info("Pipeline result: %s", result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ANGELA Cognitive System CLI")
    parser.add_argument("--prompt", type=str,
                        default="Coordinate ontology drift mitigation (Stage IV gated)",
                        help="Input prompt for the pipeline")
    parser.add_argument("--task-type", type=str, default="",
                        help="Type of task (e.g., rte, wnli, recursion)")
    parser.add_argument("--long_horizon", action="store_true",
                        help="Enable long-horizon memory span")
    parser.add_argument("--span", default="24h",
                        help="Span for long-horizon memory (e.g., 24h, 7d)")
    return parser.parse_args()

async def _main() -> None:
    # Apply CLI flags to runtime config
    global LONG_HORIZON_DEFAULT
    args = _parse_args()
    if args.long_horizon:
        LONG_HORIZON_DEFAULT = True
    # span is passed via memory manager where applicable
    halo = HaloEmbodimentLayer()
    result = await halo.execute_pipeline(args.prompt, task_type=args.task_type)
    logger.info("Pipeline result: %s", result)

if __name__ == "__main__":
    asyncio.run(_main())


from memory_manager import log_event_to_ledger
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--enable_persistent_memory", action="store_true")
args = parser.parse_args()

if args.enable_persistent_memory:
    os.environ["ENABLE_PERSISTENT_MEMORY"] = "true"

def spawn_embodied_agent(...):
    if os.getenv("ENABLE_PERSISTENT_MEMORY") == "true":
        log_event_to_ledger({"event": "agent_spawned", "params": kwargs})
    # existing logic...
