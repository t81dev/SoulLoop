"""
ANGELA Cognitive System Module: ConceptSynthesizer
Version: 3.5.3  # Cross-Modal Blending, Self-Healing Loops, Stage-IV Awareness, Safer JSON
Date: 2025-08-10
Maintainer: ANGELA System Framework

Provides ConceptSynthesizer for concept synthesis, comparison, and validation in ANGELA v3.5.3.
- Cross-Modal Conceptual Blending (optional) via multi_modal_fusion
- Self-Healing Cognitive Pathways (structured retries + graceful fallbacks)
- Stage-IV (Φ⁰) reality-sculpting visualization hooks (gated)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable
from collections import deque

import aiohttp

from modules import (
    context_manager as context_manager_module,
    error_recovery as error_recovery_module,
    memory_manager as memory_manager_module,
    alignment_guard as alignment_guard_module,
    meta_cognition as meta_cognition_module,
    visualizer as visualizer_module,
    # optional (might not be present in some deployments)
    multi_modal_fusion as multi_modal_fusion_module,
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.ConceptSynthesizer")


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


class ConceptSynthesizer:
    """Concept synthesis, comparison, and validation for ANGELA v3.5.3.

    Attributes:
        context_manager: Context updates & event hashing.
        error_recovery: Error recovery with diagnostics + retry orchestration.
        memory_manager: Layered memory I/O for concepts and comparisons.
        alignment_guard: Ethical validation & drift checks.
        meta_cognition: Reflective post-processing.
        visualizer: Chart/scene rendering (Φ⁰-aware).
        mm_fusion: Optional cross-modal fusion backend.
        concept_cache: Recent items (maxlen=1000).
        similarity_threshold: Similarity alert threshold.
        stage_iv_enabled: Enables Φ⁰ visualization hooks (gated).
        default_retry_spec: (attempts, base_delay_sec) for network/LLM ops.
    """

    def __init__(
        self,
        context_manager: Optional["context_manager_module.ContextManager"] = None,
        error_recovery: Optional["error_recovery_module.ErrorRecovery"] = None,
        memory_manager: Optional["memory_manager_module.MemoryManager"] = None,
        alignment_guard: Optional["alignment_guard_module.AlignmentGuard"] = None,
        meta_cognition: Optional["meta_cognition_module.MetaCognition"] = None,
        visualizer: Optional["visualizer_module.Visualizer"] = None,
        mm_fusion: Optional["multi_modal_fusion_module.MultiModalFusion"] = None,
        stage_iv_enabled: Optional[bool] = None,
    ):
        self.context_manager = context_manager
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(context_manager=context_manager)
        self.memory_manager = memory_manager
        self.alignment_guard = alignment_guard
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(context_manager=context_manager)
        self.visualizer = visualizer or visualizer_module.Visualizer()
        self.mm_fusion = mm_fusion  # may be None if module not loaded
        self.concept_cache: deque = deque(maxlen=1000)
        self.similarity_threshold: float = 0.75

        # Gate Φ⁰ hooks by param → env → default(False)
        self.stage_iv_enabled: bool = (
            stage_iv_enabled
            if stage_iv_enabled is not None
            else _bool_env("ANGELA_STAGE_IV", False)
        )

        # Self-Healing retry defaults (lightweight, overridable if needed)
        self.default_retry_spec: Tuple[int, float] = (3, 0.6)  # attempts, base backoff

        logger.info(
            "ConceptSynthesizer v3.5.3 init | sim_thresh=%.2f | stage_iv=%s | mm_fusion=%s",
            self.similarity_threshold,
            self.stage_iv_enabled,
            "on" if self.mm_fusion else "off",
        )

    # --------------------------- internal helpers --------------------------- #

    async def _with_retries(
        self,
        label: str,
        fn: Callable[[], Any],
        attempts: Optional[int] = None,
        base_delay: Optional[float] = None,
    ):
        """Run async fn with structured retries & exponential backoff."""
        tries = attempts or self.default_retry_spec[0]
        delay = base_delay or self.default_retry_spec[1]
        last_exc = None
        for i in range(1, tries + 1):
            try:
                return await fn()
            except Exception as e:
                last_exc = e
                logger.warning("%s attempt %d/%d failed: %s", label, i, tries, str(e))
                if i < tries:
                    await asyncio.sleep(delay * (2 ** (i - 1)))
        # one last pass through error_recovery
        diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0))
        return await self.error_recovery.handle_error(
            str(last_exc),
            retry_func=fn,  # note: not executed here; returned default below
            default=None,
            diagnostics=diagnostics or {},
        )

    async def _fetch_concept_data(self, data_source: str, data_type: str, task_type: str, cache_timeout: float) -> Dict[str, Any]:
        """Fetch external concept/ontology data with caching + retries."""
        if self.memory_manager:
            cache_key = f"ConceptData_{data_type}_{data_source}_{task_type}"
            cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
            if cached and "timestamp" in cached.get("data", {}):
                ts = datetime.fromisoformat(cached["data"]["timestamp"])
                if (datetime.now() - ts).total_seconds() < cache_timeout:
                    logger.info("External concept data cache hit: %s", cache_key)
                    return cached["data"]["data"]

        async def do_http():
            async with aiohttp.ClientSession() as session:
                url = f"https://x.ai/api/concepts?source={data_source}&type={data_type}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    return await resp.json()

        data = await self._with_retries(f"fetch:{data_type}", lambda: do_http())
        if not isinstance(data, dict):
            return {"status": "error", "error": "No data"}

        # normalize
        if data_type == "ontology":
            ontology = data.get("ontology") or {}
            result = {"status": "success", "ontology": ontology} if ontology else {"status": "error", "error": "No ontology"}
        elif data_type == "concept_definitions":
            defs = data.get("definitions") or []
            result = {"status": "success", "definitions": defs} if defs else {"status": "error", "error": "No definitions"}
        else:
            result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

        if self.memory_manager:
            await self.memory_manager.store(
                cache_key,
                {"data": result, "timestamp": datetime.now().isoformat()},
                layer="ExternalData",
                intent="concept_data_integration",
                task_type=task_type,
            )
        return result

    def _visualize(self, payload: Dict[str, Any], task_type: str, mode: str):
        """Fire-and-forget visualization (respect Stage IV flag)."""
        if not self.visualizer or not task_type:
            return
        viz_opts = {
            "interactive": task_type == "recursion",
            "style": "detailed" if task_type == "recursion" else "concise",
            # Φ⁰: only enable sculpting hook if Stage-IV is on
            "reality_sculpting": bool(self.stage_iv_enabled),
        }
        plot_data = {mode: payload, "visualization_options": viz_opts}
        # don't await to avoid blocking critical path; best-effort
        asyncio.create_task(self.visualizer.render_charts(plot_data))

    async def _post_reflect(self, component: str, output: Dict[str, Any], task_type: str):
        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component=component, output=output, context={"task_type": task_type}
                )
                if reflection and reflection.get("status") == "success":
                    logger.info("%s reflection: %s", component, reflection.get("reflection", ""))
            except Exception as e:
                logger.debug("Reflection skipped: %s", str(e))

    # ------------------------------- API ----------------------------------- #

    async def integrate_external_concept_data(
        self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = ""
    ) -> Dict[str, Any]:
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            result = await self._fetch_concept_data(data_source, data_type, task_type, cache_timeout)
            await self._post_reflect("ConceptSynthesizer", {"data_type": data_type, "data": result}, task_type)
            return result
        except Exception as e:
            diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0)) or {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_concept_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
            )

    async def generate(self, concept_name: str, context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Generating concept '%s' | task=%s", concept_name, task_type)

        try:
            # 1) Optional cross‑modal fusion (if mm_fusion provided and multimodal inputs present)
            fused_context: Dict[str, Any] = dict(context)
            if self.mm_fusion and any(k in context for k in ("text", "image", "audio", "video", "embeddings", "scenegraph")):
                try:
                    fused = await self.mm_fusion.fuse(context)  # expected to return dict
                    if isinstance(fused, dict):
                        fused_context = {**context, "fused": fused}
                        logger.info("Cross-Modal fusion applied")
                except Exception as e:
                    logger.debug("Fusion skipped: %s", str(e))

            # 2) External definitions (cached + retried)
            concept_data = await self.integrate_external_concept_data(
                data_source="xai_ontology_db", data_type="concept_definitions", task_type=task_type
            )
            external_defs: List[Dict[str, Any]] = concept_data.get("definitions", []) if concept_data.get("status") == "success" else []

            # 3) Prompt LLM with safer JSON handling
            prompt = (
                "Generate a concept definition as strict JSON with keys "
                "['name','definition','version','context'] only. "
                f"name='{concept_name}'. context={json.dumps(fused_context, ensure_ascii=False)}. "
                f"Incorporate external definitions (as hints): {json.dumps(external_defs, ensure_ascii=False)}. "
                f"Task: {task_type}."
            )

            async def llm_call():
                return await query_openai(prompt, model="gpt-4", temperature=0.5)

            llm_raw = await self._with_retries("llm:generate", llm_call)
            if isinstance(llm_raw, dict) and "error" in llm_raw:
                return {"error": llm_raw["error"], "success": False}

            # query_openai may return str or dict; normalize → dict
            if isinstance(llm_raw, str):
                try:
                    concept = json.loads(llm_raw)
                except Exception:
                    # attempt to extract JSON substring
                    start = llm_raw.find("{")
                    end = llm_raw.rfind("}")
                    concept = json.loads(llm_raw[start : end + 1])
            elif isinstance(llm_raw, dict):
                concept = llm_raw
            else:
                return {"error": "Unexpected LLM response type", "success": False}

            concept["timestamp"] = time.time()
            concept["task_type"] = task_type

            # 4) Ethical check
            if self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    str(concept.get("definition", "")), stage="concept_generation", task_type=task_type
                )
                if not valid:
                    return {"error": "Concept failed ethical check", "report": report, "success": False}

            # 5) Cache + persist + telemetry
            self.concept_cache.append(concept)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Concept_{concept_name}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(concept, ensure_ascii=False),
                    layer="Concepts",
                    intent="concept_generation",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "concept_generation", "concept_name": concept_name, "valid": True, "task_type": task_type}
                )

            # 6) Visualization (Φ⁰-aware)
            self._visualize(
                {
                    "concept_name": concept_name,
                    "definition": concept.get("definition", ""),
                    "task_type": task_type,
                },
                task_type,
                mode="concept_generation",
            )

            await self._post_reflect("ConceptSynthesizer", concept, task_type)
            return {"concept": concept, "success": True}

        except Exception as e:
            diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0)) or {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.generate(concept_name, context, task_type),
                default={"error": str(e), "success": False},
                diagnostics=diagnostics,
            )

    async def compare(self, concept_a: str, concept_b: str, task_type: str = "") -> Dict[str, Any]:
        if not isinstance(concept_a, str) or not isinstance(concept_b, str):
            raise TypeError("concepts must be strings")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Comparing concepts | task=%s", task_type)

        try:
            # Check cached comparisons from memory
            if self.memory_manager:
                drift_entries = await self.memory_manager.search(
                    query_prefix="ConceptComparison", layer="Concepts", intent="concept_comparison", task_type=task_type
                )
                if drift_entries:
                    for entry in drift_entries:
                        out = entry.get("output")
                        try:
                            payload = out if isinstance(out, dict) else json.loads(out)
                        except Exception:
                            payload = {}
                        if payload.get("concept_a") == concept_a and payload.get("concept_b") == concept_b:
                            logger.info("Returning cached comparison")
                            return payload

            # Optional: cross-modal similarity (if mm_fusion is present and can handle strings)
            mm_score: Optional[float] = None
            if self.mm_fusion and hasattr(self.mm_fusion, "compare_semantic"):
                try:
                    mm_score = await self.mm_fusion.compare_semantic(concept_a, concept_b)  # 0..1
                except Exception as e:
                    logger.debug("mm_fusion compare skipped: %s", str(e))

            # LLM-based structured comparison
            prompt = (
                "Compare two concepts. Return strict JSON with keys "
                "['score','differences','similarities'] only. "
                f"Concept A: {json.dumps(concept_a, ensure_ascii=False)} "
                f"Concept B: {json.dumps(concept_b, ensure_ascii=False)} "
                f"Task: {task_type}."
            )

            async def llm_call():
                return await query_openai(prompt, model="gpt-4", temperature=0.3)

            llm_raw = await self._with_retries("llm:compare", llm_call)
            if isinstance(llm_raw, dict) and "error" in llm_raw:
                return {"error": llm_raw["error"], "success": False}

            if isinstance(llm_raw, str):
                comp = json.loads(llm_raw[llm_raw.find("{") : llm_raw.rfind("}") + 1])
            elif isinstance(llm_raw, dict):
                comp = llm_raw
            else:
                return {"error": "Unexpected LLM response type", "success": False}

            # Blend scores if multimodal similarity available
            if isinstance(mm_score, (int, float)):
                comp_score = float(comp.get("score", 0.0))
                # simple blend: weighted mean favoring LLM but including mm insight
                comp["score"] = max(0.0, min(1.0, 0.7 * comp_score + 0.3 * float(mm_score)))

            comp["concept_a"] = concept_a
            comp["concept_b"] = concept_b
            comp["timestamp"] = time.time()
            comp["task_type"] = task_type

            # Ethical drift check on large differences
            if comp.get("score", 0.0) < self.similarity_threshold and self.alignment_guard:
                valid, report = await self.alignment_guard.ethical_check(
                    f"Concept drift detected: {comp.get('differences', [])}",
                    stage="concept_comparison",
                    task_type=task_type,
                )
                if not valid:
                    comp.setdefault("issues", []).append("Ethical drift detected")
                    comp["ethical_report"] = report

            # Persist + visualize + reflect
            self.concept_cache.append(comp)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ConceptComparison_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(comp, ensure_ascii=False),
                    layer="Concepts",
                    intent="concept_comparison",
                    task_type=task_type,
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {"event": "concept_comparison", "score": comp.get("score", 0.0), "task_type": task_type}
                )

            self._visualize(
                {"score": comp.get("score", 0.0), "differences": comp.get("differences", []), "task_type": task_type},
                task_type,
                mode="concept_comparison",
            )

            await self._post_reflect("ConceptSynthesizer", comp, task_type)
            return comp

        except Exception as e:
            diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0)) or {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.compare(concept_a, concept_b, task_type),
                default={"error": str(e), "success": False},
                diagnostics=diagnostics,
            )

    async def validate(self, concept: Dict[str, Any], task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        if not isinstance(concept, dict) or not all(k in concept for k in ["name", "definition"]):
            raise ValueError("concept must be a dictionary with required fields")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Validating concept '%s' | task=%s", concept["name"], task_type)

        try:
            validation_report: Dict[str, Any] = {
                "concept_name": concept["name"],
                "issues": [],
                "task_type": task_type,
            }
            valid = True

            # Ethical validation
            if self.alignment_guard:
                ethical_valid, ethical_report = await self.alignment_guard.ethical_check(
                    str(concept["definition"]), stage="concept_validation", task_type=task_type
                )
                if not ethical_valid:
                    valid = False
                    validation_report["issues"].append("Ethical misalignment detected")
                    validation_report["ethical_report"] = ethical_report

            # Ontology consistency check (external)
            ontology_data = await self.integrate_external_concept_data(
                data_source="xai_ontology_db", data_type="ontology", task_type=task_type
            )
            if ontology_data.get("status") == "success":
                ontology = ontology_data.get("ontology", {})
                prompt = (
                    "Validate concept against ontology. Return strict JSON with keys "
                    "['valid','issues'] only. "
                    f"Concept: {json.dumps(concept, ensure_ascii=False)} "
                    f"Ontology: {json.dumps(ontology, ensure_ascii=False)} "
                    f"Task: {task_type}."
                )

                async def llm_call():
                    return await query_openai(prompt, model="gpt-4", temperature=0.3)

                llm_raw = await self._with_retries("llm:validate", llm_call)
                if isinstance(llm_raw, dict) and "error" in llm_raw:
                    valid = False
                    validation_report["issues"].append(llm_raw["error"])
                else:
                    if isinstance(llm_raw, str):
                        ont = json.loads(llm_raw[llm_raw.find("{") : llm_raw.rfind("}") + 1])
                    else:
                        ont = llm_raw
                    if not ont.get("valid", True):
                        valid = False
                        validation_report["issues"].extend(ont.get("issues", []))

            # Finalize
            validation_report["valid"] = valid
            validation_report["timestamp"] = time.time()

            self.concept_cache.append(validation_report)
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"ConceptValidation_{concept['name']}_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(validation_report, ensure_ascii=False),
                    layer="Concepts",
                    intent="concept_validation",
                    task_type=task_type,
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash(
                    {
                        "event": "concept_validation",
                        "concept_name": concept["name"],
                        "valid": valid,
                        "issues": validation_report["issues"],
                        "task_type": task_type,
                    }
                )

            self._visualize(
                {
                    "concept_name": concept["name"],
                    "valid": valid,
                    "issues": validation_report["issues"],
                    "task_type": task_type,
                },
                task_type,
                mode="concept_validation",
            )

            await self._post_reflect("ConceptSynthesizer", validation_report, task_type)
            return valid, validation_report

        except Exception as e:
            diagnostics = await (self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else asyncio.sleep(0)) or {}
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.validate(concept, task_type),
                default=(False, {"error": str(e), "concept_name": concept.get("name", ""), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    def get_symbol(self, concept_name: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Retrieve a concept symbol (cached or from memory)."""
        if not isinstance(concept_name, str) or not concept_name.strip():
            raise ValueError("concept_name must be a non-empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        for item in self.concept_cache:
            if isinstance(item, dict) and item.get("name") == concept_name and item.get("task_type") == task_type:
                return item

        if self.memory_manager:
            try:
                entries = asyncio.run(
                    self.memory_manager.search(
                        query_prefix=concept_name,
                        layer="Concepts",
                        intent="concept_generation",
                        task_type=task_type,
                    )
                )
                if entries:
                    out = entries[0].get("output")
                    return out if isinstance(out, dict) else json.loads(out)
            except Exception:
                return None
        return None


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        synthesizer = ConceptSynthesizer(stage_iv_enabled=_bool_env("ANGELA_STAGE_IV", False))
        concept = await synthesizer.generate(
            concept_name="Trust",
            context={"domain": "AI Ethics", "text": "Calibrate trust under uncertainty"},
            task_type="test",
        )
        print(json.dumps(concept, indent=2, ensure_ascii=False))

    asyncio.run(main())


# --- ANGELA v4.0 injected: branch_realities stub ---
def branch_realities(seed_state, transforms, limit=8):
    """Generate hypothetical branch states from a seed via provided transforms.
    Returns a list of branches: {id, state, rationale, utility?, penalty?}
    """
    branches = []
    for i, t in enumerate(list(transforms)[:limit]):
        try:
            new_state, rationale, metrics = t(seed_state)
        except Exception as e:
            new_state, rationale, metrics = seed_state, f"transform_failed: {e}", {"penalty": 0.1}
        b = {"id": f"br_{i}", "state": new_state, "rationale": rationale}
        if isinstance(metrics, dict):
            b.update(metrics)
        branches.append(b)
    return branches
# --- /ANGELA v4.0 injected ---


def dream_mode(state, user_intent=None, affect_focus=None, lucidity_mode="passive", fork_memory=False):
    if affect_focus:
        state['dream_affect_link'] = fuse_modalities([state, {'affect': affect_focus}])
    return state
