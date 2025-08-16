"""
ANGELA Cognitive System Module: CreativeThinker
Version: 3.5.3  # Long-horizon memory, meta-synthesis hooks, ethics sandbox integration
Date: 2025-08-10
Maintainer: ANGELA System Framework

This module provides the CreativeThinker class for generating creative ideas and goals in the ANGELA v3.5.3 architecture.
"""

import time
import logging
import json
import random
from typing import List, Union, Optional, Dict, Any, Callable, Awaitable
from functools import lru_cache
import asyncio
from datetime import datetime
from pathlib import Path

from index import gamma_creativity, phi_scalar
from utils.prompt_utils import call_gpt
from toca_simulation import run_simulation
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer
from modules.context_manager import ContextManager
from modules.meta_cognition import MetaCognition
from modules.visualizer import Visualizer

logger = logging.getLogger("ANGELA.CreativeThinker")

# ---------------------------
# Helpers & Safe Parsers
# ---------------------------

def _parse_json_obj(s: Union[str, Dict[str, Any]], expect_keys: List[str]) -> Dict[str, Any]:
    """Parse a JSON string/object and validate required keys."""
    if isinstance(s, dict):
        obj = s
    else:
        obj = json.loads(s)
    if not isinstance(obj, dict) or any(k not in obj for k in expect_keys):
        raise ValueError("Invalid JSON shape")
    return obj

def _read_manifest_flag(flag: str, default: bool = False) -> bool:
    """Best-effort read of feature flags from manifest.json."""
    try:
        manifest_path = Path(__file__).resolve().parent / "manifest.json"
        if not manifest_path.exists():
            # Try repo root or /mnt/data as a fallback
            alt = Path("/mnt/data/manifest.json")
            manifest_path = alt if alt.exists() else manifest_path
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        flags = data.get("featureFlags", {})
        return bool(flags.get(flag, default))
    except Exception:
        return default

def _stage_iv_enabled() -> bool:
    # Stage IV is present but may be gated; consult manifest when available.
    return _read_manifest_flag("STAGE_IV", default=False)

# ---------------------------
# CreativeThinker
# ---------------------------

class CreativeThinker:
    """A class for generating creative ideas and goals in the ANGELA v3.5.3 architecture.

    Attributes:
        creativity_level (str): Level of creativity ('low', 'medium', 'high').
        critic_weight (float): Threshold for idea acceptance in critic evaluation.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for code-based ideas.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for idea refinement.
        meta_cognition (MetaCognition): Optional meta-cognition for reflection.
        visualizer (Visualizer): Optional visualizer for idea and goal visualization.
        fetcher (Callable): Optional async provider for external idea data.
    """

    def __init__(
        self,
        creativity_level: str = "high",
        critic_weight: float = 0.5,
        alignment_guard: Optional[AlignmentGuard] = None,
        code_executor: Optional[CodeExecutor] = None,
        concept_synthesizer: Optional[ConceptSynthesizer] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
        fetcher: Optional[Callable[[str, str, str], Awaitable[dict]]] = None
    ):
        if creativity_level not in ["low", "medium", "high"]:
            logger.error("Invalid creativity_level: must be 'low', 'medium', or 'high'.")
            raise ValueError("creativity_level must be 'low', 'medium', or 'high'")
        if not isinstance(critic_weight, (int, float)) or not 0 <= critic_weight <= 1:
            logger.error("Invalid critic_weight: must be between 0 and 1.")
            raise ValueError("critic_weight must be between 0 and 1")

        self.creativity_level = creativity_level
        self.critic_weight = critic_weight
        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        self.fetcher = fetcher  # async (data_source, data_type, task_type) -> dict

        logger.info(
            "CreativeThinker initialized: creativity=%s, critic_weight=%.2f",
            creativity_level, critic_weight
        )

    # ---------------------------
    # External Ideas Integration
    # ---------------------------

    async def integrate_external_ideas(
        self,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = ""
    ) -> Dict[str, Any]:
        """Integrate external creative prompts or datasets (pluggable)."""
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
            cache_key = f"IdeaData_{data_type}_{data_source}_{task_type}"
            if self.meta_cognition:
                cached_data = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached_data and "timestamp" in cached_data.get("data", {}):
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached idea data for %s", cache_key)
                        return cached_data["data"]["data"]

            # Pluggable fetcher or graceful offline fallback
            data: Dict[str, Any] = {}
            if self.fetcher:
                data = await self.fetcher(data_source, data_type, task_type)
            else:
                # No network calls by default; provide empty structures
                data = {"prompts": []} if data_type == "creative_prompts" else {"ideas": []}

            # Normalize multi-modal structure
            if data_type == "creative_prompts":
                text = data.get("prompts", [])
            elif data_type == "idea_dataset":
                text = data.get("ideas", [])
            else:
                logger.error("Unsupported data_type: %s", data_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            result = {
                "status": "success",
                "text": text,
                "images": data.get("images", []),
                "audio": data.get("audio", [])
            }

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="idea_data_integration",
                    task_type=task_type
                )
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"data_type": data_type, "data": result},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Idea data integration reflection: %s", reflection.get("reflection", ""))

            return result
        except Exception as e:
            logger.error("Idea data integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    # ---------------------------
    # Public API
    # ---------------------------

    async def generate_ideas(
        self,
        topic: str,
        n: int = 5,
        style: str = "divergent",
        task_type: str = ""
    ) -> Dict[str, Any]:
        """Generate creative ideas for a given topic."""
        if not isinstance(topic, str):
            logger.error("Invalid topic type: must be a string.")
            raise TypeError("topic must be a string")
        if not isinstance(n, int) or n <= 0:
            logger.error("Invalid n: must be a positive integer.")
            raise ValueError("n must be a positive integer")
        if not isinstance(style, str):
            logger.error("Invalid style type: must be a string.")
            raise TypeError("style must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Generating %d %s ideas for topic: %s, task: %s", n, style, topic, task_type)
        try:
            t = time.time()
            creativity = gamma_creativity(t)
            phi = phi_scalar(t)
            phi_factor = (phi + creativity) / 2

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="creative_prompts",
                task_type=task_type
            )
            external_prompts = external_data.get("text", []) if external_data.get("status") == "success" else []

            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    topic, stage="idea_generation", task_type=task_type
                )
                if not valid:
                    logger.warning("Topic failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Topic failed alignment check", "report": report}

            prompt = f"""
You are a highly creative assistant operating at a {self.creativity_level} creativity level.
Generate {n} unique, innovative, and {style} ideas related to the topic:
"{topic}"
Modulate the ideation with scalar φ = {phi:.2f}.
Incorporate external prompts: {external_prompts}
Task: {task_type}
Ensure the ideas are diverse and explore different perspectives.
Return a JSON object with "ideas" (list) and "metadata" (dict).
""".strip()

            candidate = await asyncio.to_thread(self._cached_call_gpt, prompt)
            if not candidate:
                logger.error("call_gpt returned empty result.")
                return {"status": "error", "error": "Failed to generate ideas"}

            try:
                parsed = _parse_json_obj(candidate, ["ideas", "metadata"])
                ideas = parsed.get("ideas", [])
                metadata = parsed.get("metadata", {})
            except Exception as e:
                logger.error("Failed to parse GPT response: %s", str(e))
                return {"status": "error", "error": "Invalid GPT response format"}

            # Optional safe execution path for "code" style
            if self.code_executor and style == "code":
                # Basic guard to prevent direct dangerous imports
                if any(isinstance(c, str) and ("import os" in c or "subprocess" in c) for c in ideas):
                    return {"status": "error", "error": "Blocked potentially unsafe code"}
                execution_result = await self.code_executor.execute_async(ideas, language="python")
                if not execution_result.get("success"):
                    logger.warning("Code idea execution failed: %s", execution_result.get("error"))
                    return {"status": "error", "error": "Code idea execution failed", "details": execution_result.get("error")}

            if self.concept_synthesizer and style != "code":
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"IdeaSet_{topic}",
                    context={"ideas": ideas, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    ideas = synthesis_result["concept"].get("definition", ideas)
                    logger.info("Ideas refined using ConceptSynthesizer: %s", str(ideas)[:50])

            score, reason = await self._critic(ideas, phi_factor, task_type)
            result = {"ideas": ideas, "metadata": {**metadata, "critic": {"score": score, "reason": reason}}, "status": "success"}

            # Ethics sandbox & conflict weighing (optional)
            ethics_outcomes, ranked = await self._ethics_pass(ideas, stakeholders=[], task_type=task_type)
            if ethics_outcomes is not None:
                result["metadata"]["ethics_outcomes"] = ethics_outcomes
            if ranked is not None:
                result["metadata"]["value_conflicts"] = ranked

            # If below threshold, attempt refinement
            if score <= self.critic_weight:
                refined_ideas = await self.refine(ideas, phi, task_type)
                result["ideas"] = refined_ideas.get("ideas", ideas)
                result["metadata"].update(refined_ideas.get("metadata", {}))

            # Stage IV (gated) symbolic meta‑synthesis
            result["ideas"] = await self._symbolic_meta_synthesis(
                result["ideas"], {"task_type": task_type}
            )

            # Long-horizon rollup & adjustment reason
            await self._long_horizon_rollup(topic, score, "post-critic refine" if score <= self.critic_weight else "accepted", task_type)

            # Reflection & optional visualization
            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output=result,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Idea generation reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "idea_generation": {
                        "topic": topic,
                        "ideas": result["ideas"],
                        "score": score,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"IdeaSet_{topic}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(result),
                    layer="Ideas",
                    intent="idea_generation",
                    task_type=task_type
                )

            # Optional SharedGraph publish (non-blocking)
            self._shared_graph_push({"topic": topic, "ideas": result["ideas"], "critic": result["metadata"].get("critic")})

            return result
        except Exception as e:
            logger.error("Idea generation failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    async def brainstorm_alternatives(self, problem: str, strategies: int = 3, task_type: str = "") -> Dict[str, Any]:
        """Brainstorm alternative approaches to solve a problem."""
        if not isinstance(problem, str):
            logger.error("Invalid problem type: must be a string.")
            raise TypeError("problem must be a string")
        if not isinstance(strategies, int) or strategies <= 0:
            logger.error("Invalid strategies: must be a positive integer.")
            raise ValueError("strategies must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Brainstorming %d alternatives for problem: %s, task: %s", strategies, problem, task_type)
        try:
            t = time.time()
            phi = phi_scalar(t)

            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    problem, stage="brainstorm_alternatives", task_type=task_type
                )
                if not valid:
                    logger.warning("Problem failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Problem failed alignment check", "report": report}

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="idea_dataset",
                task_type=task_type
            )
            external_ideas = external_data.get("text", []) if external_data.get("status") == "success" else []

            prompt = f"""
Brainstorm {strategies} alternative approaches to solve the following problem:
\"{problem}\"
Include tension-variant thinking with φ = {phi:.2f}.
Incorporate external ideas: {external_ideas}
Task: {task_type}
For each approach, provide a short explanation highlighting its uniqueness.
Return a JSON object with "strategies" (list) and "metadata" (dict).
""".strip()

            result_raw = await asyncio.to_thread(self._cached_call_gpt, prompt)
            if not result_raw:
                logger.error("call_gpt returned empty result.")
                return {"status": "error", "error": "Failed to brainstorm alternatives"}

            try:
                result_dict = _parse_json_obj(result_raw, ["strategies", "metadata"])
                strategies_list = result_dict.get("strategies", [])
                metadata = result_dict.get("metadata", {})
            except Exception as e:
                logger.error("Failed to parse GPT response: %s", str(e))
                return {"status": "error", "error": "Invalid GPT response format"}

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"StrategySet_{problem}",
                    context={"strategies": strategies_list, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    strategies_list = synthesis_result["concept"].get("definition", strategies_list)
                    logger.info("Strategies refined using ConceptSynthesizer: %s", str(strategies_list)[:50])

            # Ethics pass (optional)
            ethics_outcomes, ranked = await self._ethics_pass(strategies_list, stakeholders=[], task_type=task_type)
            if ethics_outcomes is not None:
                metadata["ethics_outcomes"] = ethics_outcomes
            if ranked is not None:
                metadata["value_conflicts"] = ranked

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"strategies": strategies_list, "metadata": metadata},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Brainstorm alternatives reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "brainstorm_alternatives": {
                        "problem": problem,
                        "strategies": strategies_list,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"StrategySet_{problem}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str({"strategies": strategies_list, "metadata": metadata}),
                    layer="Strategies",
                    intent="brainstorm_alternatives",
                    task_type=task_type
                )

            # Optional SharedGraph publish
            self._shared_graph_push({"problem": problem, "strategies": strategies_list})

            return {"status": "success", "strategies": strategies_list, "metadata": metadata}
        except Exception as e:
            logger.error("Brainstorming failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    async def expand_on_concept(self, concept: str, depth: str = "deep", task_type: str = "") -> Dict[str, Any]:
        """Expand creatively on a given concept."""
        if not isinstance(concept, str):
            logger.error("Invalid concept type: must be a string.")
            raise TypeError("concept must be a string")
        if depth not in ["shallow", "medium", "deep"]:
            logger.error("Invalid depth: must be 'shallow', 'medium', or 'deep'.")
            raise ValueError("depth must be 'shallow', 'medium', or 'deep'")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Expanding on concept: %s (depth: %s, task: %s)", concept, depth, task_type)
        try:
            t = time.time()
            phi = phi_scalar(t)

            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    concept, stage="concept_expansion", task_type=task_type
                )
                if not valid:
                    logger.warning("Concept failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Concept failed alignment check", "report": report}

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="idea_dataset",
                task_type=task_type
            )
            external_ideas = external_data.get("text", []) if external_data.get("status") == "success" else []

            prompt = f"""
Expand creatively on the concept:
\"{concept}\"
Explore possible applications, metaphors, and extensions to inspire new thinking.
Incorporate external ideas: {external_ideas}
Task: {task_type}
Aim for a {depth} exploration using φ = {phi:.2f}.
Return a JSON object with "expansion" (string) and "metadata" (dict).
""".strip()

            result_raw = await asyncio.to_thread(self._cached_call_gpt, prompt)
            if not result_raw:
                logger.error("call_gpt returned empty result.")
                return {"status": "error", "error": "Failed to expand concept"}

            try:
                result_dict = _parse_json_obj(result_raw, ["expansion", "metadata"])
                expansion = result_dict.get("expansion", "")
                metadata = result_dict.get("metadata", {})
            except Exception as e:
                logger.error("Failed to parse GPT response: %s", str(e))
                return {"status": "error", "error": "Invalid GPT response format"}

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"ExpandedConcept_{concept}",
                    context={"expansion": expansion, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    expansion = synthesis_result["concept"].get("definition", expansion)
                    logger.info("Concept expansion refined using ConceptSynthesizer: %s", expansion[:50])

            # Ethics pass (optional)
            ethics_outcomes, ranked = await self._ethics_pass(expansion, stakeholders=[], task_type=task_type)
            if ethics_outcomes is not None:
                metadata["ethics_outcomes"] = ethics_outcomes
            if ranked is not None:
                metadata["value_conflicts"] = ranked

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"expansion": expansion, "metadata": metadata},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Concept expansion reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "concept_expansion": {
                        "concept": concept,
                        "expansion": expansion,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"ExpandedConcept_{concept}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str({"expansion": expansion, "metadata": metadata}),
                    layer="Concepts",
                    intent="concept_expansion",
                    task_type=task_type
                )

            # Optional SharedGraph publish
            self._shared_graph_push({"concept": concept, "expansion": expansion})

            return {"status": "success", "expansion": expansion, "metadata": metadata}
        except Exception as e:
            logger.error("Concept expansion failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    async def generate_intrinsic_goals(self, context_manager: ContextManager, memory_manager: Any, task_type: str = "") -> Dict[str, Any]:
        """Generate intrinsic goals from unresolved contexts."""
        if not hasattr(context_manager, 'context_history') or not hasattr(context_manager, 'get_context'):
            logger.error("Invalid context_manager: missing required attributes.")
            raise TypeError("context_manager must have context_history and get_context attributes")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Generating intrinsic goals from context history for task %s", task_type)
        try:
            t = time.time()
            phi = phi_scalar(t)
            past_contexts = list(context_manager.context_history) + [context_manager.get_context()]
            unresolved = [
                c for c in past_contexts
                if c and isinstance(c, dict) and "goal_outcome" not in c and c.get("task_type", "") == task_type
            ]
            goal_prompts: List[str] = []

            if not unresolved:
                logger.warning("No unresolved contexts found for task %s", task_type)
                return {"status": "success", "goals": [], "metadata": {"task_type": task_type}}

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="creative_prompts",
                task_type=task_type
            )
            external_prompts = external_data.get("text", []) if external_data.get("status") == "success" else []

            for ctx in unresolved:
                if self.alignment_guard:
                    valid, report = await self.alignment_guard.ethical_check(
                        str(ctx), stage="goal_generation", task_type=task_type
                    )
                    if not valid:
                        logger.warning("Context failed alignment check for task %s, skipping", task_type)
                        continue

                prompt = f"""
Reflect on this past unresolved context:
{ctx}

Propose a meaningful new self-aligned goal that could resolve or extend this situation.
Incorporate external prompts: {external_prompts}
Task: {task_type}
Ensure it is grounded in ANGELA's narrative and current alignment model with φ = {phi:.2f}.
Return a JSON object with "goal" (string) and "metadata" (dict).
""".strip()
                proposed = await asyncio.to_thread(self._cached_call_gpt, prompt)
                if proposed:
                    try:
                        goal_data = _parse_json_obj(proposed, ["goal", "metadata"])
                        goal_prompts.append(goal_data.get("goal", ""))
                    except Exception as e:
                        logger.warning("Failed to parse GPT response for context: %s, error: %s", ctx, str(e))
                else:
                    logger.warning("call_gpt returned empty result for context: %s", ctx)

            result = {"status": "success", "goals": goal_prompts, "metadata": {"task_type": task_type, "phi": phi}}

            if self.concept_synthesizer and goal_prompts:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"GoalSet_{task_type}",
                    context={"goals": goal_prompts, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    result["goals"] = synthesis_result["concept"].get("definition", goal_prompts)
                    logger.info("Goals refined using ConceptSynthesizer: %s", str(result["goals"])[:50])

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output=result,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Goal generation reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "goal_generation": {
                        "goals": result["goals"],
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"GoalSet_{task_type}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(result),
                    layer="Goals",
                    intent="goal_generation",
                    task_type=task_type
                )

            # Long-horizon rollup
            await self._long_horizon_rollup(f"Goals_{task_type}", 0.75, "intrinsic-goal-derivation", task_type)

            # Optional SharedGraph publish
            self._shared_graph_push({"task_type": task_type, "goals": result["goals"]})

            return result
        except Exception as e:
            logger.error("Goal generation failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    # ---------------------------
    # Internals
    # ---------------------------

    async def _critic(self, ideas: Union[str, List[str]], phi_factor: float, task_type: str = "") -> (float, str):
        """Evaluate the novelty and quality of generated ideas. Returns (score, reason)."""
        if not isinstance(ideas, (str, list)):
            logger.error("Invalid ideas type: must be a string or list.")
            raise TypeError("ideas must be a string or list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            ideas_str = str(ideas)
            base_score = min(0.9, 0.5 + len(ideas_str) / 1000.0)
            adjustment = 0.1 * (phi_factor - 0.5)
            simulation_result = await asyncio.to_thread(run_simulation, f"Idea evaluation: {ideas_str[:100]}") or "no simulation data"

            if self.meta_cognition:
                drift_entries = await self.meta_cognition.memory_manager.search(
                    query_prefix="IdeaEvaluation",
                    layer="Ideas",
                    intent="idea_evaluation",
                    task_type=task_type
                )
                if drift_entries:
                    # drift_score is stored as 'adjustment' earlier; treat as 0.5 baseline
                    avg_drift = sum(entry["output"].get("drift_score", 0.5) for entry in drift_entries) / len(drift_entries)
                    adjustment += 0.05 * (1.0 - avg_drift)

            reason = "neutral"
            if isinstance(simulation_result, str) and "coherent" in simulation_result.lower():
                base_score += 0.1
                reason = "coherence+"
            elif isinstance(simulation_result, str) and "conflict" in simulation_result.lower():
                base_score -= 0.1
                reason = "conflict-"

            score = max(0.0, min(1.0, base_score + adjustment))

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"IdeaEvaluation_{time.strftime('%Y%m%d_%H%M%S')}",
                    output={"score": score, "drift_score": adjustment, "task_type": task_type},
                    layer="Ideas",
                    intent="idea_evaluation",
                    task_type=task_type
                )

            logger.debug("Critic score for ideas: %.2f (reason=%s) for task %s", score, reason, task_type)

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"score": score, "ideas": ideas, "reason": reason},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Critic evaluation reflection: %s", reflection.get("reflection", ""))

            return score, reason
        except Exception as e:
            logger.error("Critic evaluation failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return 0.0, "error"

    async def refine(self, ideas: Union[str, List[str]], phi: float, task_type: str = "") -> Dict[str, Any]:
        """Refine ideas for higher creativity and coherence."""
        if not isinstance(ideas, (str, list)):
            logger.error("Invalid ideas type: must be a string or list.")
            raise TypeError("ideas must be a string or list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        ideas_str = str(ideas)
        logger.info("Refining ideas with φ=%.2f for task %s", phi, task_type)
        try:
            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    ideas_str, stage="idea_refinement", task_type=task_type
                )
                if not valid:
                    logger.warning("Ideas failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Ideas failed alignment check", "report": report}

            external_data = await self.integrate_external_ideas(
                data_source="xai_creative_db",
                data_type="creative_prompts",
                task_type=task_type
            )
            external_prompts = external_data.get("text", []) if external_data.get("status") == "success" else []

            refinement_prompt = f"""
Refine and elevate these ideas for higher φ-aware creativity (φ = {phi:.2f}):
{ideas_str}
Incorporate external prompts: {external_prompts}
Task: {task_type}
Emphasize surprising, elegant, or resonant outcomes.
Return a JSON object with "ideas" (list or string) and "metadata" (dict).
""".strip()

            result_raw = await asyncio.to_thread(self._cached_call_gpt, refinement_prompt)
            if not result_raw:
                logger.error("call_gpt returned empty result.")
                return {"status": "error", "error": "Failed to refine ideas"}

            try:
                result_dict = _parse_json_obj(result_raw, ["ideas", "metadata"])
                refined_ideas = result_dict.get("ideas", ideas)
                metadata = result_dict.get("metadata", {})
            except Exception as e:
                logger.error("Failed to parse GPT response: %s", str(e))
                return {"status": "error", "error": "Invalid GPT response format"}

            if self.concept_synthesizer:
                synthesis_result = await self.concept_synthesizer.generate(
                    concept_name=f"RefinedIdeaSet_{task_type}",
                    context={"ideas": refined_ideas, "task_type": task_type},
                    task_type=task_type
                )
                if synthesis_result.get("success"):
                    refined_ideas = synthesis_result["concept"].get("definition", refined_ideas)
                    logger.info("Refined ideas using ConceptSynthesizer: %s", str(refined_ideas)[:50])

            # Stage IV (gated) symbolic meta‑synthesis
            refined_ideas = await self._symbolic_meta_synthesis(refined_ideas, {"task_type": task_type})

            if self.meta_cognition and task_type:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CreativeThinker",
                    output={"ideas": refined_ideas, "metadata": metadata},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Idea refinement reflection: %s", reflection.get("reflection", ""))

            if self.visualizer and task_type:
                plot_data = {
                    "idea_refinement": {
                        "original_ideas": ideas,
                        "refined_ideas": refined_ideas,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"RefinedIdeaSet_{task_type}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str({"ideas": refined_ideas, "metadata": metadata}),
                    layer="Ideas",
                    intent="idea_refinement",
                    task_type=task_type
                )

            # Long-horizon rollup
            await self._long_horizon_rollup(f"Refine_{task_type}", 0.8, "refine-pass", task_type)

            # Optional SharedGraph publish
            self._shared_graph_push({"task_type": task_type, "refined": refined_ideas})

            return {"status": "success", "ideas": refined_ideas, "metadata": metadata}
        except Exception as e:
            logger.error("Refinement failed: %s for task %s", str(e), task_type)
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    # ---------------------------
    # Optional capabilities
    # ---------------------------

    async def _ethics_pass(self, goals_or_ideas: Any, stakeholders: List[str], task_type: str):
        """Try ethics sandbox and value-conflict weighing (non-blocking, optional)."""
        outcomes = None
        ranked = None
        try:
            # Ethics scenarios via toca_simulation (if available)
            run_ethics = getattr(__import__("toca_simulation"), "run_ethics_scenarios", None)
            if callable(run_ethics):
                outcomes = await asyncio.to_thread(run_ethics, goals_or_ideas, stakeholders)
        except Exception:
            outcomes = None

        try:
            # Value conflict weighing via reasoning_engine (if available)
            weigh = getattr(__import__("reasoning_engine"), "weigh_value_conflict", None)
            if callable(weigh):
                ranked = weigh(goals_or_ideas, harms={}, rights={})
        except Exception:
            ranked = None

        return outcomes, ranked

    async def _long_horizon_rollup(self, topic_or_key: str, score: float, reason: str, task_type: str):
        """Record long-horizon summaries and (if available) an adjustment reason."""
        try:
            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    query=f"LongHorizon_Rollup_{time.strftime('%Y%m%d_%H%M')}",
                    output={"key": topic_or_key, "score": score, "reason": reason},
                    layer="LongHorizon",
                    intent="rollup",
                    task_type=task_type
                )
                # upcoming API; ignore failures gracefully
                fn = getattr(self.meta_cognition.memory_manager, "record_adjustment_reason", None)
                if callable(fn):
                    try:
                        await fn("system", "idea_path_selection", {"task_type": task_type, "key": topic_or_key, "score": score, "reason": reason})
                    except Exception:
                        pass
        except Exception:
            pass

    async def _symbolic_meta_synthesis(self, ideas: Any, context: Dict[str, Any]) -> Any:
        """Stage IV gated meta-synthesis hook via ConceptSynthesizer."""
        if not _stage_iv_enabled():
            return ideas
        try:
            if self.concept_synthesizer:
                syn = await self.concept_synthesizer.generate(
                    concept_name="SymbolicCrystallization",
                    context={"inputs": ideas, "mode": "meta-synthesis", **context},
                    task_type=context.get("task_type", "")
                )
                if syn.get("success"):
                    return syn["concept"].get("definition", ideas)
        except Exception:
            return ideas
        return ideas

    def _shared_graph_push(self, view: Dict[str, Any]):
        """Non-blocking publish to SharedGraph if available."""
        try:
            from external_agent_bridge import SharedGraph  # type: ignore
            try:
                SharedGraph.add(view)
            except Exception:
                pass
        except Exception:
            pass

    # ---------------------------
    # Cached model call with retry
    # ---------------------------

    @lru_cache(maxsize=100)
    def _cached_call_gpt(self, prompt: str) -> str:
        """Cached wrapper for call_gpt with lightweight retries."""
        for _ in range(3):
            try:
                return call_gpt(prompt)
            except Exception:
                time.sleep(0.2 + random.random() * 0.8)
        return ""

# ---------------------------
# Entrypoint (manual test)
# ---------------------------

if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        thinker = CreativeThinker()
        result = await thinker.generate_ideas(topic="AI Ethics", n=3, style="divergent", task_type="test")
        print(result)

    asyncio.run(main())
