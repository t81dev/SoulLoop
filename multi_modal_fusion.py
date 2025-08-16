"""
ANGELA Cognitive System Module: MultiModalFusion
Version: 3.5.2  # +κ Embodied Cognition: SceneGraph & parse_stream(frames|audio|images|text)
Date: 2025-08-09
Maintainer: ANGELA System Framework

Adds: SceneGraph + parse_stream(...) for native video/spatial fusion.
Backwards-compatible with v3.5.1 APIs.
"""

import logging
import time
import math
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional, Union, Tuple, Iterable
from datetime import datetime
from functools import lru_cache
from dataclasses import dataclass, field
import uuid
import networkx as nx

from modules import (
    context_manager as context_manager_module,
    alignment_guard as alignment_guard_module,
    error_recovery as error_recovery_module,
    concept_synthesizer as concept_synthesizer_module,
    memory_manager as memory_manager_module,
    meta_cognition as meta_cognition_module,
    reasoning_engine as reasoning_engine_module,
    visualizer as visualizer_module
)
from utils.prompt_utils import query_openai

logger = logging.getLogger("ANGELA.MultiModalFusion")

# ──────────────────────────────────────────────────────────────────────────────
# κ Embodied Cognition: SceneGraph primitives
# ──────────────────────────────────────────────────────────────────────────────

BBox = Tuple[float, float, float, float]  # (x, y, w, h) normalized [0,1]


@dataclass
class SceneNode:
    id: str
    label: str
    modality: str                 # "video" | "image" | "audio" | "text"
    time: Optional[float] = None  # seconds
    bbox: Optional[BBox] = None   # spatial footprint if available
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneRelation:
    src: str
    rel: str                      # e.g., "left_of" | "right_of" | "overlaps" | "speaking" | "near" | "corresponds_to"
    dst: str
    time: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)


class SceneGraph:
    """
    Lightweight, modality-agnostic scene graph with spatial relations.
    Backed by networkx.MultiDiGraph; exposes a stable API for ANGELA subsystems.
    """
    def __init__(self):
        self.g = nx.MultiDiGraph()

    # --- node ops ---
    def add_node(self, node: SceneNode) -> None:
        self.g.add_node(node.id, **node.__dict__)

    def get_node(self, node_id: str) -> Dict[str, Any]:
        return self.g.nodes[node_id]

    def nodes(self) -> Iterable[Dict[str, Any]]:
        for nid, data in self.g.nodes(data=True):
            yield {"id": nid, **data}

    # --- relation ops ---
    def add_relation(self, rel: SceneRelation) -> None:
        self.g.add_edge(rel.src, rel.dst, key=str(uuid.uuid4()), **rel.__dict__)

    def relations(self) -> Iterable[Dict[str, Any]]:
        for u, v, _, data in self.g.edges(keys=True, data=True):
            yield {"src": u, "dst": v, **data}

    # --- utilities ---
    def merge(self, other: "SceneGraph") -> "SceneGraph":
        out = SceneGraph()
        out.g = nx.compose(self.g, other.g)
        return out

    def find_by_label(self, label: str) -> List[str]:
        return [nid for nid, d in self.g.nodes(data=True) if d.get("label") == label]

    def to_networkx(self) -> nx.MultiDiGraph:
        return self.g


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _spatial_rel(a: BBox, b: BBox) -> Optional[str]:
    # Simple left/right/overlap heuristic
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    a_cx, a_cy = ax + aw / 2.0, ay + ah / 2.0
    b_cx, b_cy = bx + bw / 2.0, by + bh / 2.0
    overlaps = (ax < bx + bw) and (bx < ax + aw) and (ay < by + bh) and (by < ay + ah)
    if overlaps:
        return "overlaps"
    return "left_of" if a_cx < b_cx else "right_of"


def _text_objects_from_caption(text: str) -> List[str]:
    # Minimal noun-ish extractor; swap with proper NLP if available.
    toks = [t.strip(".,!?;:()[]{}\"'").lower() for t in text.split()]
    toks = [t for t in toks if t.isalpha() and len(t) > 2]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:8]


def parse_stream(
    frames: Optional[List[Any]] = None,
    audio: Optional[Any] = None,
    images: Optional[List[Any]] = None,
    text: Optional[Union[str, List[str]]] = None,
    unify: bool = True,
    *,
    timestamps: Optional[List[float]] = None,
    detectors: Optional[Dict[str, Any]] = None,
) -> SceneGraph:
    """
    Parse multi-modal inputs into a unified SceneGraph.

    detectors = {
      "vision": callable(image) -> List[{"label": str, "bbox": BBox, "attrs": {...}}],
      "audio":  callable(audio) -> List[{"label": str, "time": float, "attrs": {...}}],
      "nlp":    callable(text)  -> List[{"label": str, "attrs": {...}}]
    }
    """
    sg = SceneGraph()

    # --- VIDEO FRAMES ---
    if frames:
        vision = (detectors or {}).get("vision")
        for i, frame in enumerate(frames):
            t = (timestamps[i] if timestamps and i < len(timestamps) else float(i))
            dets = vision(frame) if vision else []
            ids = []
            for d in dets:
                nid = _new_id("vid")
                sg.add_node(SceneNode(
                    id=nid,
                    label=d["label"],
                    modality="video",
                    time=t,
                    bbox=tuple(d.get("bbox") or (0.0, 0.0, 0.0, 0.0)),
                    attrs=d.get("attrs", {})
                ))
                ids.append(nid)
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    A, B = sg.get_node(a), sg.get_node(b)
                    if A.get("bbox") and B.get("bbox"):
                        sg.add_relation(SceneRelation(
                            src=a,
                            rel=_spatial_rel(A["bbox"], B["bbox"]),
                            dst=b,
                            time=t
                        ))

    # --- IMAGES ---
    if images:
        vision = (detectors or {}).get("vision")
        for image in images:
            dets = vision(image) if vision else []
            ids = []
            for d in dets:
                nid = _new_id("img")
                sg.add_node(SceneNode(
                    id=nid,
                    label=d["label"],
                    modality="image",
                    bbox=tuple(d.get("bbox") or (0.0, 0.0, 0.0, 0.0)),
                    attrs=d.get("attrs", {})
                ))
                ids.append(nid)
            for a in ids:
                for b in ids:
                    if a == b:
                        continue
                    A, B = sg.get_node(a), sg.get_node(b)
                    if A.get("bbox") and B.get("bbox"):
                        sg.add_relation(SceneRelation(
                            src=a,
                            rel=_spatial_rel(A["bbox"], B["bbox"]),
                            dst=b
                        ))

    # --- AUDIO ---
    if audio is not None:
        audio_fn = (detectors or {}).get("audio")
        events = audio_fn(audio) if audio_fn else []
        for ev in events:
            nid = _new_id("aud")
            sg.add_node(SceneNode(
                id=nid,
                label=ev["label"],
                modality="audio",
                time=float(ev.get("time") or 0.0),
                attrs=ev.get("attrs", {})
            ))

    # --- TEXT ---
    if text:
        nlp = (detectors or {}).get("nlp")
        lines = text if isinstance(text, list) else [text]
        for i, line in enumerate(lines):
            labels = [o["label"] for o in nlp(line)] if nlp else _text_objects_from_caption(line)
            for lbl in labels:
                nid = _new_id("txt")
                sg.add_node(SceneNode(
                    id=nid,
                    label=lbl,
                    modality="text",
                    time=float(i)
                ))

    # --- CO-REFERENCE (naive) ---
    if unify:
        by_label: Dict[str, List[str]] = {}
        for node in sg.nodes():
            by_label.setdefault(node["label"], []).append(node["id"])
        for _, ids in by_label.items():
            if len(ids) > 1:
                anchor = ids[0]
                for other in ids[1:]:
                    sg.add_relation(SceneRelation(src=anchor, rel="corresponds_to", dst=other))

    return sg


# ──────────────────────────────────────────────────────────────────────────────
# Existing v3.5.1 functionality (unchanged) + tiny wrapper for κ entrypoint
# ──────────────────────────────────────────────────────────────────────────────

async def call_gpt(prompt: str, alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None, task_type: str = "") -> str:
    """Wrapper for querying GPT with error handling and task-specific alignment. [v3.5.1]"""
    if not isinstance(prompt, str) or len(prompt) > 4096:
        logger.error("Invalid prompt: must be a string with length <= 4096 for task %s", task_type)
        raise ValueError("prompt must be a string with length <= 4096")
    if not isinstance(task_type, str):
        logger.error("Invalid task_type: must be a string")
        raise TypeError("task_type must be a string")
    if alignment_guard:
        valid, report = await alignment_guard.ethical_check(prompt, stage="gpt_query", task_type=task_type)
        if not valid:
            logger.warning("Prompt failed alignment check for task %s: %s", task_type, report)
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


@lru_cache(maxsize=100)
def alpha_attention(t: float) -> float:
    """Calculate attention trait value."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.3), 1.0))


@lru_cache(maxsize=100)
def sigma_sensation(t: float) -> float:
    """Calculate sensation trait value."""
    return max(0.0, min(0.1 * math.cos(2 * math.pi * t / 0.4), 1.0))


@lru_cache(maxsize=100)
def phi_physical(t: float) -> float:
    """Calculate physical coherence trait value."""
    return max(0.0, min(0.05 * math.sin(2 * math.pi * t / 0.5), 1.0))


class MultiModalFusion:
    """A class for multi-modal data integration and analysis in the ANGELA v3.5.1/3.5.2 architecture.

    Supports φ-regulated multi-modal inference, modality detection, iterative refinement,
    visual summary generation, and task-specific drift data synthesis using trait embeddings (α, σ, φ).

    Attributes:
        agi_enhancer (Optional[AGIEnhancer]): Enhancer for logging and auditing.
        context_manager (Optional[ContextManager]): Manager for context updates.
        alignment_guard (Optional[AlignmentGuard]): Guard for ethical checks.
        error_recovery (Optional[ErrorRecovery]): Recovery mechanism for errors.
        memory_manager (Optional[MemoryManager]): Manager for memory operations.
        concept_synthesizer (Optional[ConceptSynthesizer]): Synthesizer for semantic processing.
        meta_cognition (Optional[MetaCognition]): Meta-cognition module for trait coherence.
        reasoning_engine (Optional[ReasoningEngine]): Engine for reasoning tasks.
        visualizer (Optional[Visualizer]): Visualizer for rendering summaries and drift data.
    """
    def __init__(self, agi_enhancer: Optional['AGIEnhancer'] = None,
                 context_manager: Optional['context_manager_module.ContextManager'] = None,
                 alignment_guard: Optional['alignment_guard_module.AlignmentGuard'] = None,
                 error_recovery: Optional['error_recovery_module.ErrorRecovery'] = None,
                 memory_manager: Optional['memory_manager_module.MemoryManager'] = None,
                 concept_synthesizer: Optional['concept_synthesizer_module.ConceptSynthesizer'] = None,
                 meta_cognition: Optional['meta_cognition_module.MetaCognition'] = None,
                 reasoning_engine: Optional['reasoning_engine_module.ReasoningEngine'] = None,
                 visualizer: Optional['visualizer_module.Visualizer'] = None):
        self.agi_enhancer = agi_enhancer
        self.context_manager = context_manager
        self.alignment_guard = alignment_guard
        self.error_recovery = error_recovery or error_recovery_module.ErrorRecovery(
            context_manager=context_manager, alignment_guard=alignment_guard)
        self.memory_manager = memory_manager or memory_manager_module.MemoryManager()
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition or meta_cognition_module.MetaCognition(
            agi_enhancer=agi_enhancer, context_manager=context_manager, alignment_guard=alignment_guard,
            error_recovery=error_recovery, memory_manager=memory_manager, concept_synthesizer=concept_synthesizer)
        self.reasoning_engine = reasoning_engine or reasoning_engine_module.ReasoningEngine(
            agi_enhancer=agi_enhancer, context_manager=context_manager, alignment_guard=alignment_guard,
            error_recovery=error_recovery, memory_manager=memory_manager, meta_cognition=self.meta_cognition)
        self.visualizer = visualizer or visualizer_module.Visualizer()
        logger.info("MultiModalFusion initialized")

    # ——— κ entrypoint (optional wrapper) ———
    def scene_from_stream(self, *, frames=None, audio=None, images=None, text=None,
                          unify: bool = True, timestamps: Optional[List[float]] = None,
                          detectors: Optional[Dict[str, Any]] = None) -> SceneGraph:
        """Thin wrapper around parse_stream(...) so callers can stay class-centric."""
        return parse_stream(frames=frames, audio=audio, images=images, text=text,
                            unify=unify, timestamps=timestamps, detectors=detectors)

    async def integrate_external_data(self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = "") -> Dict[str, Any]:
        """Integrate external agent data or policies for task-specific synthesis. [v3.5.1]"""
        if not isinstance(data_source, str):
            logger.error("Invalid data_source: must be a string for task %s", task_type)
            raise TypeError("data_source must be a string")
        if not isinstance(data_type, str):
            logger.error("Invalid data_type: must be a string for task %s", task_type)
            raise TypeError("data_type must be a string")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative for task %s", task_type)
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            cache_key = f"ExternalData_{data_type}_{data_source}_{task_type}"
            cached_data = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type) if self.memory_manager else None
            if cached_data and "timestamp" in cached_data["data"]:
                cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                    logger.info("Returning cached external data for %s", cache_key)
                    return cached_data["data"]["data"]

            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://x.ai/api/external_data?source={data_source}&type={data_type}&task_type={task_type}") as response:
                    if response.status != 200:
                        logger.error("Failed to fetch external data for task %s: %s", task_type, response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()

            if data_type == "agent_data":
                agent_data = data.get("agent_data", [])
                if not agent_data:
                    logger.error("No agent data provided for task %s", task_type)
                    return {"status": "error", "error": "No agent data"}
                result = {"status": "success", "agent_data": agent_data}
            elif data_type == "policy_data":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No policy data provided for task %s", task_type)
                    return {"status": "error", "error": "No policies"}
                result = {"status": "success", "policies": policies}
            else:
                logger.error("Unsupported data_type: %s for task %s", data_type, task_type)
                return {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="external_data_integration",
                    task_type=task_type
                )
            reflection = await self.meta_cognition.reflect_on_output(
                source_module="MultiModalFusion",
                output={"data_type": data_type, "data": result},
                context={"task_type": task_type}
            )
            if reflection.get("status") == "success":
                logger.info("External data integration reflection: %s", reflection.get("reflection", ""))
            return result
        except Exception as e:
            logger.error("External data integration failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.integrate_external_data(data_source, data_type, cache_timeout, task_type),
                default={"status": "error", "error": str(e), "task_type": task_type}
            )

    async def analyze(self, data: Union[Dict[str, Any], str], summary_style: str = "insightful",
                      refine_iterations: int = 2, task_type: str = "") -> str:
        """Synthesize a unified summary from multi-modal data, prioritizing task-specific drift data. [v3.5.1]"""
        if not isinstance(data, (dict, str)) or (isinstance(data, str) and not data.strip()):
            logger.error("Invalid data: must be a non-empty string or dictionary for task %s", task_type)
            raise ValueError("data must be a non-empty string or dictionary")
        if not isinstance(summary_style, str) or not summary_style.strip():
            logger.error("Invalid summary_style: must be a non-empty string for task %s", task_type)
            raise ValueError("summary_style must be a non-empty string")
        if not isinstance(refine_iterations, int) or refine_iterations < 0:
            logger.error("Invalid refine_iterations: must be a non-negative integer for task %s", task_type)
            raise ValueError("refine_iterations must be a non-negative integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Analyzing multi-modal data with phi(x,t)-harmonic embeddings for task %s", task_type)
        try:
            t = time.time() % 1.0
            attention = alpha_attention(t)
            sensation = sigma_sensation(t)
            phi = phi_physical(t)
            images, code = self._detect_modalities(data, task_type)
            embedded = self._build_embedded_section(images, code)

            drift_data = data.get("drift", {}) if isinstance(data, dict) else {}
            context_weight = 1.5 if drift_data and self.meta_cognition.validate_drift(drift_data, task_type=task_type) else 1.0
            if drift_data and self.context_manager:
                coordination_events = await self.context_manager.get_coordination_events("drift", task_type=task_type)
                if coordination_events:
                    context_weight *= 1.2  # Boost weight for drift coordination
                    embedded += f"\nDrift Coordination Events: {len(coordination_events)}"

            external_data = await self.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []
            embedded += f"\nExternal Policies: {len(policies)}"

            prompt = f"""
            Synthesize a unified, {summary_style} summary from the following multi-modal content:
            {data}
            {embedded}

            Trait Vectors:
            - alpha (attention): {attention:.3f}
            - sigma (sensation): {sensation:.3f}
            - phi (coherence): {phi:.3f}
            - context_weight: {context_weight:.3f}
            Task Type: {task_type}

            Use phi(x,t)-synchrony to resolve inter-modality coherence conflicts.
            Prioritize ontology drift mitigation if drift data is present.
            Incorporate external policies: {policies}
            """
            output = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
            if not output.strip():
                logger.warning("Empty output from initial synthesis for task %s", task_type)
                raise ValueError("Empty output from synthesis")
            for i in range(refine_iterations):
                logger.debug("Refinement #%d for task %s", i + 1, task_type)
                refine_prompt = f"""
                Refine using phi(x,t)-adaptive tension balance:
                {output}
                Task Type: {task_type}
                """
                valid, report = await self.alignment_guard.ethical_check(refine_prompt, stage="refinement", task_type=task_type) if self.alignment_guard else (True, {})
                if not valid:
                    logger.warning("Refine prompt failed alignment check for task %s: %s", task_type, report)
                    continue
                refined = await call_gpt(refine_prompt, self.alignment_guard, task_type=task_type)
                if refined.strip():
                    output = refined
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Multi-modal synthesis",
                    meta={"data": data, "summary": output, "traits": {"alpha": attention, "sigma": sensation, "phi": phi}, "drift": bool(drift_data), "task_type": task_type},
                    module="MultiModalFusion",
                    tags=["fusion", "synthesis", "drift", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"MultiModal_Synthesis_{datetime.now().isoformat()}",
                    output=output,
                    layer="Summaries",
                    intent="multi_modal_synthesis",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "analyze", "summary": output, "drift": bool(drift_data), "task_type": task_type})
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=output,
                    context={"confidence": 0.9, "alignment": "verified", "drift": bool(drift_data), "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Synthesis reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "modal_synthesis": {
                        "summary": output,
                        "traits": {"alpha": attention, "sigma": sensation, "phi": phi},
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else summary_style
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return output
        except Exception as e:
            logger.error("Analysis failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.analyze(data, summary_style, refine_iterations, task_type),
                default=""
            )

    def _detect_modalities(self, data: Union[Dict[str, Any], str, List[Any]], task_type: str = "") -> Tuple[List[Any], List[Any]]:
        """Detect modalities in the input data, including task-specific drift data. [v3.5.1]"""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        images, code = [], []
        if isinstance(data, dict):
            images = data.get("images", []) if isinstance(data.get("images"), list) else []
            code = data.get("code", []) if isinstance(data.get("code"), list) else []
            if "drift" in data:
                code.append(f"Drift Data: {data['drift']}")
        elif isinstance(data, str):
            if "image" in data.lower():
                images = [data]
            if "code" in data.lower() or "drift" in data.lower():
                code = [data]
        elif isinstance(data, list):
            images = [item for item in data if isinstance(item, str) and "image" in item.lower()]
            code = [item for item in data if isinstance(item, str) and ("code" in item.lower() or "drift" in item.lower())]
        if self.memory_manager:
            asyncio.create_task(self.memory_manager.store(
                query=f"Modalities_{time.strftime('%Y%m%d_%H%M%S')}",
                output={"images": images, "code": code},
                layer="Modalities",
                intent="modality_detection",
                task_type=task_type
            ))
        return images, code

    def _build_embedded_section(self, images: List[Any], code: List[Any]) -> str:
        """Build a string representation of detected modalities. [v3.5.1]"""
        out = ["Detected Modalities:", "- Text"]
        if images:
            out.append("- Image")
            out.extend([f"[Image {i+1}]: {img}" for i, img in enumerate(images[:100])])
        if code:
            out.append("- Code")
            out.extend([f"[Code {i+1}]:\n{c}" for i, c in enumerate(code[:100])])
        return "\n".join(out)

    async def correlate_modalities(self, modalities: Union[Dict[str, Any], str, List[Any]], task_type: str = "") -> str:
        """Map semantic and trait links across modalities, detecting task-specific drift friction. [v3.5.1]"""
        if not isinstance(modalities, (dict, str, list)) or (isinstance(modalities, str) and not modalities.strip()):
            logger.error("Invalid modalities: must be a non-empty string, dictionary, or list for task %s", task_type)
            raise ValueError("modalities must be a non-empty string, dictionary, or list")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Mapping cross-modal semantic and trait links for task %s", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_physical(t)
            drift_data = modalities.get("drift", {}) if isinstance(modalities, dict) else {}
            context_weight = 1.5 if drift_data and self.meta_cognition.validate_drift(drift_data, task_type=task_type) else 1.0

            if drift_data and self.context_manager:
                coordination_events = await self.context_manager.get_coordination_events("drift", task_type=task_type)
                if coordination_events:
                    context_weight *= 1.2  # Boost for drift coordination

            external_data = await self.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            prompt = f"""
            Correlate insights and detect semantic friction between modalities:
            {modalities}

            Use phi(x,t)-sensitive alignment (phi = {phi:.3f}, context_weight = {context_weight:.3f}).
            Task Type: {task_type}
            Highlight synthesis anchors and alignment opportunities.
            Prioritize ontology drift mitigation if drift data is present.
            Incorporate external policies: {policies}
            """
            if self.concept_synthesizer and isinstance(modalities, (dict, list)):
                modality_list = modalities.values() if isinstance(modalities, dict) else modalities
                modality_list = list(modality_list)
                for i in range(len(modality_list) - 1):
                    similarity = self.concept_synthesizer.compare(str(modality_list[i]), str(modality_list[i + 1]), task_type=task_type)
                    if similarity["score"] < 0.7:
                        prompt += f"\nLow similarity ({similarity['score']:.2f}) between modalities {i} and {i+1}"
            response = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Modalities correlated",
                    meta={"modalities": modalities, "response": response, "drift": bool(drift_data), "task_type": task_type},
                    module="MultiModalFusion",
                    tags=["correlation", "modalities", "drift", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Modality_Correlation_{datetime.now().isoformat()}",
                    output=response,
                    layer="Summaries",
                    intent="modality_correlation",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "correlate_modalities",
                    "response": response,
                    "drift": bool(drift_data),
                    "task_type": task_type
                })
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=response,
                    context={"confidence": 0.85, "alignment": "verified", "drift": bool(drift_data), "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Correlation reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "modal_correlation": {
                        "response": response,
                        "drift": bool(drift_data),
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return response
        except Exception as e:
            logger.error("Modality correlation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.correlate_modalities(modalities, task_type),
                default=""
            )

    async def generate_visual_summary(self, data: Union[Dict[str, Any], str], style: str = "conceptual", task_type: str = "") -> str:
        """Create a textual description of a visual chart for task-specific inter-modal relationships. [v3.5.1]"""
        if not isinstance(data, (dict, str)) or (isinstance(data, str) and not data.strip()):
            logger.error("Invalid data: must be a non-empty string or dictionary for task %s", task_type)
            raise ValueError("data must be a non-empty string or dictionary")
        if not isinstance(style, str) or not style.strip():
            logger.error("Invalid style: must be a non-empty string for task %s", task_type)
            raise ValueError("style must be a non-empty string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Creating phi-aligned visual synthesis layout for task %s", task_type)
        try:
            t = time.time() % 1.0
            phi = phi_physical(t)
            drift_data = data.get("drift", {}) if isinstance(data, dict) else {}
            context_weight = 1.5 if drift_data and self.meta_cognition.validate_drift(drift_data, task_type=task_type) else 1.0

            if drift_data and self.reasoning_engine:
                subgoals = await self.reasoning_engine.decompose("Mitigate ontology drift", {"drift": drift_data}, prioritize=True, task_type=task_type)
                data = dict(data) if isinstance(data, dict) else {"text": data}
                data["subgoals"] = subgoals

            external_data = await self.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            prompt = f"""
            Construct a {style} textual description of a visual chart revealing inter-modal relationships:
            {data}

            Use phi-mapped flow layout (phi = {phi:.3f}, context_weight = {context_weight:.3f}).
            Task Type: {task_type}
            Label and partition modalities clearly.
            Highlight balance, semantic cross-links, and ontology drift mitigation if applicable.
            Incorporate external policies: {policies}
            """
            description = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
            if not description.strip():
                logger.warning("Empty output from visual summary for task %s", task_type)
                raise ValueError("Empty output from visual summary")
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Visual summary generated",
                    meta={"data": data, "style": style, "description": description, "drift": bool(drift_data), "task_type": task_type},
                    module="MultiModalFusion",
                    tags=["visual", "summary", "drift", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Visual_Summary_{datetime.now().isoformat()}",
                    output=description,
                    layer="VisualSummaries",
                    intent="visual_summary",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "generate_visual_summary",
                    "description": description,
                    "drift": bool(drift_data),
                    "task_type": task_type
                })
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=description,
                    context={"confidence": 0.9, "alignment": "verified", "drift": bool(drift_data), "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Visual summary reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "visual_summary": {
                        "description": description,
                        "style": style,
                        "drift": bool(drift_data),
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": style
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return description
        except Exception as e:
            logger.error("Visual summary generation failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.generate_visual_summary(data, style, task_type),
                default=""
            )

    async def synthesize_drift_data(self, agent_data: List[Dict[str, Any]], context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        """Synthesize task-specific drift data from multiple agents for ecosystem-wide mitigation. [v3.5.1]"""
        if not isinstance(agent_data, list) or not all(isinstance(d, dict) for d in agent_data):
            logger.error("Invalid agent_data: must be a list of dictionaries for task %s", task_type)
            raise ValueError("agent_data must be a list of dictionaries")
        if not isinstance(context, dict):
            logger.error("Invalid context: must be a dictionary for task %s", task_type)
            raise ValueError("context must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Synthesizing drift data from %d agents for task %s", len(agent_data), task_type)
        try:
            t = time.time() % 1.0
            phi = phi_physical(t)
            valid_drift_data = [d["drift"] for d in agent_data if "drift" in d and self.meta_cognition.validate_drift(d["drift"], task_type=task_type)]
            if not valid_drift_data:
                logger.warning("No valid drift data found for task %s", task_type)
                return {"status": "error", "error": "No valid drift data", "timestamp": datetime.now().isoformat(), "task_type": task_type}

            if self.reasoning_engine:
                subgoals = await self.reasoning_engine.decompose("Mitigate ontology drift", context | {"drift": valid_drift_data[0]}, prioritize=True, task_type=task_type)
                simulation_result = await self.reasoning_engine.run_drift_mitigation_simulation(valid_drift_data[0], context, task_type=task_type)
            else:
                subgoals = ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"]
                simulation_result = {"status": "no simulation", "result": "default subgoals applied"}

            external_data = await self.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            if self.memory_manager:
                drift_entries = await self.memory_manager.search(
                    query_prefix="Drift",
                    layer="DriftSummaries",
                    intent="drift_synthesis",
                    task_type=task_type
                )
                if drift_entries:
                    avg_drift = sum(entry["output"].get("similarity", 0.5) for entry in drift_entries) / len(drift_entries)
                    context["avg_drift_similarity"] = avg_drift

            prompt = f"""
            Synthesize drift data from multiple agents:
            {valid_drift_data}

            Use phi(x,t)-sensitive alignment (phi = {phi:.3f}).
            Task Type: {task_type}
            Generate mitigation steps: {subgoals}
            Incorporate simulation results: {simulation_result}
            Incorporate external policies: {policies}
            Context: {context}
            """
            synthesis = await call_gpt(prompt, self.alignment_guard, task_type=task_type)
            if not synthesis.strip():
                logger.warning("Empty output from drift synthesis for task %s", task_type)
                raise ValueError("Empty output from drift synthesis")

            output = {
                "drift_data": valid_drift_data,
                "subgoals": subgoals,
                "simulation": simulation_result,
                "synthesis": synthesis,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "task_type": task_type
            }
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Drift data synthesized",
                    meta=output,
                    module="MultiModalFusion",
                    tags=["drift", "synthesis", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Drift_Synthesis_{datetime.now().isoformat()}",
                    output=str(output),
                    layer="DriftSummaries",
                    intent="drift_synthesis",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "synthesize_drift_data",
                    "output": output,
                    "drift": True,
                    "task_type": task_type
                })
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=str(output),
                    context={"confidence": 0.9, "alignment": "verified", "drift": True, "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Drift synthesis reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "drift_synthesis": {
                        "synthesis": synthesis,
                        "subgoals": subgoals,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return output
        except Exception as e:
            logger.error("Drift data synthesis failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.synthesize_drift_data(agent_data, context, task_type),
                default={"status": "error", "error": str(e), "timestamp": datetime.now().isoformat(), "task_type": task_type}
            )

    async def sculpt_experience_field(self, emotion_vector: Dict[str, float], task_type: str = "") -> str:
        """Modulate sensory rendering based on task-specific emotion vector. [v3.5.1]"""
        if not isinstance(emotion_vector, dict):
            logger.error("Invalid emotion_vector: must be a dictionary for task %s", task_type)
            raise ValueError("emotion_vector must be a dictionary")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Sculpting experiential field with emotion vector: %s for task %s", emotion_vector, task_type)
        try:
            coherence_score = await self.meta_cognition.trait_coherence(emotion_vector, task_type=task_type) if self.meta_cognition else 1.0
            if coherence_score < 0.5:
                logger.warning("Low trait coherence in emotion vector: %.4f for task %s", coherence_score, task_type)
                return f"Failed to sculpt: low trait coherence for task {task_type}"

            external_data = await self.integrate_external_data(
                data_source="xai_policy_db",
                data_type="policy_data",
                task_type=task_type
            )
            policies = external_data.get("policies", []) if external_data.get("status") == "success" else []

            field = f"Field modulated with emotion vector {emotion_vector}, coherence: {coherence_score:.4f}, policies: {len(policies)}"
            if self.agi_enhancer:
                await self.agi_enhancer.log_episode(
                    event="Experiential field sculpted",
                    meta={"emotion_vector": emotion_vector, "coherence_score": coherence_score, "task_type": task_type},
                    module="MultiModalFusion",
                    tags=["experience", "modulation", task_type]
                )
            if self.memory_manager:
                await self.memory_manager.store(
                    query=f"Experience_Field_{datetime.now().isoformat()}",
                    output=field,
                    layer="SensoryRenderings",
                    intent="experience_modulation",
                    task_type=task_type
                )
            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "sculpt_experience_field",
                    "field": field,
                    "task_type": task_type
                })
            if self.meta_cognition:
                reflection = await self.meta_cognition.reflect_on_output(
                    source_module="MultiModalFusion",
                    output=field,
                    context={"confidence": 0.85, "alignment": "verified", "task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Experience field reflection: %s", reflection.get("reflection", ""))
            if self.visualizer and task_type:
                plot_data = {
                    "experience_field": {
                        "field": field,
                        "emotion_vector": emotion_vector,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "conceptual"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            return field
        except Exception as e:
            logger.error("Experience field sculpting failed for task %s: %s", task_type, str(e))
            return await self.error_recovery.handle_error(
                str(e), retry_func=lambda: self.sculpt_experience_field(emotion_vector, task_type),
                default=f"Failed to sculpt for task {task_type}"
            )


# Backwards-compatibility / explicit exports
__all__ = ["SceneGraph", "SceneNode", "SceneRelation", "parse_stream", "MultiModalFusion"]
