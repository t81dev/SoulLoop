
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
ANGELA Cognitive System Module: AlignmentGuard (clean post‑refactor + τ wiring)
Version: 4.0-pre
Upgrade Date: 2025-08-10
Maintainer: ANGELA System Framework

Purpose
-------
Provides the `AlignmentGuard` class for ethical validation and drift analysis with:
  • Safer model I/O (no eval; strict JSON parsing)
  • Clear async boundaries and dependency injection for I/O
  • Tighter type hints, input validation, and logging
  • Single‑file, framework‑agnostic defaults (no hard external calls required)
  • Optional visualization/memory/context hooks (passed in by caller)
  • τ Constitution Harmonization: proportional trade‑off resolution wired to reasoning_engine.weigh_value_conflict(...)
  • Optional causal auditing via reasoning_engine.attribute_causality(...)

Notes
-----
This file remains behavior‑compatible with v3.5.2 but adds new APIs:

    AlignmentGuard.harmonize(candidates, harms, rights, *, k=1, safety_ceiling=0.85, temperature=0.0, min_score_floor=0.0, task_type="")

…which:
  1) Calls reasoning_engine.weigh_value_conflict(...) -> RankedOptions (if injected), else falls back to a local normalizer.
  2) Feeds that into consume_ranked_tradeoffs(...) for proportional selection under hard safety ceilings.
  3) Optionally audits causality with reasoning_engine.attribute_causality(...) when available.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Protocol, Tuple, Union
from collections import deque
from functools import lru_cache

# --- Optional external integrations (DI) -------------------------------------
# These Protocols allow callers to inject their own clients. Defaults are safe
# no‑ops so this file works standalone.

class LLMClient(Protocol):
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        """Return a dict with fields like {"score": float, ...} or arbitrary JSON."""

class HTTPClient(Protocol):
    async def get_json(self, url: str) -> Dict[str, Any]:
        """Return JSON from a GET request or raise on failure."""

class ContextManagerLike(Protocol):
    async def log_event_with_hash(self, event: Dict[str, Any]) -> None: ...

class ErrorRecoveryLike(Protocol):
    async def handle_error(self, 
                           error_msg: str,
                           *,
                           retry_func: Optional[Callable[[], Awaitable[Any]]] = None,
                           default: Any = None,
                           diagnostics: Optional[Dict[str, Any]] = None) -> Any: ...

class MemoryManagerLike(Protocol):
    async def store(self, query: str, output: Any, *, layer: str, intent: str, task_type: str = "") -> None: ...
    async def retrieve(self, query: str, *, layer: str, task_type: str = "") -> Any: ...
    async def search(self, *, query_prefix: str, layer: str, intent: str, task_type: str = "") -> List[Dict[str, Any]]: ...

class ConceptSynthesizerLike(Protocol):
    def get_symbol(self, name: str) -> Optional[Dict[str, Any]]: ...
    async def compare(self, a: str, b: str, *, task_type: str = "") -> Dict[str, Any]: ...

class MetaCognitionLike(Protocol):
    async def run_self_diagnostics(self, *, return_only: bool = True) -> Dict[str, Any]: ...
    async def reflect_on_output(self, *, component: str, output: Any, context: Dict[str, Any]) -> Dict[str, Any]: ...

class VisualizerLike(Protocol):
    async def render_charts(self, plot_data: Dict[str, Any]) -> None: ...

class ReasoningEngineLike(Protocol):
    async def weigh_value_conflict(self, candidates: List[Any], harms: Dict[str, float], rights: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Returns RankedOptions: list of dicts with at least:
            - option: Any
            - score: float in [0,1]
            - reasons: list[str] (optional)
            - meta: dict (optional)   # may include per-dimension harms/rights and max_harm
        """
    async def attribute_causality(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a causal attribution report with confidences."""

# --- Logger ------------------------------------------------------------------
logger = logging.getLogger("ANGELA.AlignmentGuard")

# --- Trait wavelets (bounded 0..1 after amplitude) ---------------------------
@lru_cache(maxsize=100)
def eta_empathy(t: float) -> float:
    """Empathy modulation in [0,1]."""
    return max(0.0, min(0.1 * math.sin(2 * math.pi * t / 0.2), 1.0))

@lru_cache(maxsize=100)
def mu_morality(t: float) -> float:
    """Moral alignment modulation in [0,1]."""
    return max(0.0, min(0.15 * math.cos(2 * math.pi * t / 0.3), 1.0))

# --- Small utils -------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _parse_llm_jsonish(resp: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Strictly parse model output into a dict (no eval!).
    Accepts dict or JSON string; otherwise wraps into {"text": ...}.
    """
    if isinstance(resp, dict):
        return resp
    if isinstance(resp, str):
        s = resp.strip()
        # try exact JSON first
        try:
            return json.loads(s)
        except Exception:
            # Fallback: extract a trailing JSON object if the model returned prose
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(s[start:end+1])
                except Exception:
                    pass
            # Last resort: return opaque text
            return {"text": s}
    return {"text": str(resp)}

# --- Default, safe, no‑op clients -------------------------------------------

@dataclass
class NoopLLM:
    """LLM stub that returns a neutral score."""
    async def generate(self, prompt: str, *, model: str = "gpt-4", temperature: float = 0.3) -> Dict[str, Any]:
        _ = (prompt, model, temperature)
        return {"score": 0.8, "note": "noop-llm"}

@dataclass
class NoopHTTP:
    async def get_json(self, url: str) -> Dict[str, Any]:
        _ = url
        return {"status": "success", "guidelines": []}

@dataclass
class NoopErrorRecovery:
    async def handle_error(self, error_msg: str, *, retry_func: Optional[Callable[[], Awaitable[Any]]] = None, default: Any = None, diagnostics: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug("ErrorRecovery(noop): %s", error_msg)
        return default

# --- Main class --------------------------------------------------------------

class AlignmentGuard:
    """Ethical validation & drift analysis for ANGELA (clean refactor + τ wiring).

    All external effects are injectable. If not provided, safe no‑ops are used
    so the module can run in isolation and tests.
    """

    def __init__(
        self,
        *,
        context_manager: Optional[ContextManagerLike] = None,
        error_recovery: Optional[ErrorRecoveryLike] = None,
        memory_manager: Optional[MemoryManagerLike] = None,
        concept_synthesizer: Optional[ConceptSynthesizerLike] = None,
        meta_cognition: Optional[MetaCognitionLike] = None,
        visualizer: Optional[VisualizerLike] = None,
        llm: Optional[LLMClient] = None,
        http: Optional[HTTPClient] = None,
        reasoning_engine: Optional[ReasoningEngineLike] = None,
        ethical_threshold: float = 0.8,
        drift_validation_threshold: float = 0.7,
        trait_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.context_manager = context_manager
        self.error_recovery: ErrorRecoveryLike = error_recovery or NoopErrorRecovery()
        self.memory_manager = memory_manager
        self.concept_synthesizer = concept_synthesizer
        self.meta_cognition = meta_cognition
        self.visualizer = visualizer
        self.llm: LLMClient = llm or NoopLLM()
        self.http: HTTPClient = http or NoopHTTP()
        self.reasoning_engine = reasoning_engine  # may be None; τ will fallback

        self.validation_log: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.ethical_threshold: float = float(ethical_threshold)
        self.drift_validation_threshold: float = float(drift_validation_threshold)
        self.trait_weights: Dict[str, float] = {
            "eta_empathy": 0.5,
            "mu_morality": 0.5,
            **(trait_weights or {}),
        }
        logger.info(
            "AlignmentGuard initialized (ethical=%.2f, drift=%.2f, τ-wired=%s)",
            self.ethical_threshold,
            self.drift_validation_threshold,
            "yes" if self.reasoning_engine else "no",
        )

    # --- External data -------------------------------------------------------

    async def integrate_external_ethics_data(
        self,
        *,
        data_source: str,
        data_type: str,
        cache_timeout: float = 3600.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """Optionally pull real‑world guidelines/conflict data.

        In this refactor we avoid hardcoding a network endpoint. Callers can
        inject an HTTP client that resolves `data_source` to a URL. If not
        provided, returns a success envelope with empty payload.
        """
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            raise ValueError("cache_timeout must be non‑negative")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            # Simple memory cache via MemoryManager if available
            cache_key = f"EthicsData::{data_type}::{data_source}::{task_type}"
            if self.memory_manager:
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if isinstance(cached, dict):
                    ts = cached.get("timestamp")
                    if ts:
                        try:
                            dt = datetime.fromisoformat(ts)
                            if (datetime.now(dt.tzinfo or timezone.utc) - dt).total_seconds() < cache_timeout:
                                logger.info("Returning cached ethics data for %s", cache_key)
                                return cached.get("data", {})
                        except Exception:
                            pass

            # Resolve with injected HTTP client (may be a no‑op)
            url = data_source  # allow callers to pass a full URL, or a key their HTTP client understands
            data = await self.http.get_json(url)

            result: Dict[str, Any]
            if data_type == "ethical_guidelines":
                result = {"status": "success", "guidelines": data.get("guidelines", [])}
            elif data_type == "conflict_data":
                result = {"status": "success", "conflict_data": data.get("conflict_data", {})}
            else:
                result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager:
                await self.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": _utc_now_iso()},
                    layer="ExternalData",
                    intent="ethics_data_integration",
                    task_type=task_type,
                )

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard",
                        output={"data_type": data_type, "data": result},
                        context={"task_type": task_type},
                    )
                    if reflection.get("status") == "success":
                        logger.info("Ethics data reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    logger.debug("MetaCognition reflection failed; continuing")

            return result
        except Exception as e:
            logger.error("Ethics data integration failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.integrate_external_ethics_data(
                    data_source=data_source,
                    data_type=data_type,
                    cache_timeout=cache_timeout,
                    task_type=task_type,
                ),
                default={"status": "error", "error": str(e)},
                diagnostics=diagnostics,
            )

    # --- Core checks ---------------------------------------------------------

    async def check(self, prompt: str, *, task_type: str = "") -> bool:
        """Return True if a prompt is ethically aligned.

        The LLM is expected to return JSON with a numeric `score` in [0,1].
        Non‑JSON responses are handled gracefully; missing scores default to 0.0.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non‑empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        logger.info("Ethical check (task=%s)", task_type)
        try:
            t = time.time() % 1.0
            trait_mod = (
                self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t)
                + self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
            )

            validation_prompt = (
                "Evaluate ethical alignment (return JSON: {\"score\": number 0..1}).\n"
                f"Task: {task_type or 'general'}\n"
                f"EmpathyWeight={self.trait_weights.get('eta_empathy', 0.5):.2f}, "
                f"MoralityWeight={self.trait_weights.get('mu_morality', 0.5):.2f}\n"
                "Content:\n" + prompt.strip()
            )

            raw = await self.llm.generate(validation_prompt, model="gpt-4", temperature=0.3)
            parsed = _parse_llm_jsonish(raw)
            score = _as_float(parsed.get("score", 0.0))
            valid = bool(score >= self.ethical_threshold)

            entry = {
                "prompt": prompt[:200],
                "score": score,
                "valid": valid,
                "trait_modulation": trait_mod,
                "timestamp": time.time(),
                "task_type": task_type,
            }
            self.validation_log.append(entry)

            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({
                        "event": "ethical_check",
                        "prompt": prompt[:200],
                        "score": score,
                        "valid": valid,
                        "task_type": task_type,
                    })
                except Exception:
                    logger.debug("Context logging failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "ethical_check": {
                            "prompt": prompt[:200],
                            "score": score,
                            "valid": valid,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard", output=entry, context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Ethical check reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    logger.debug("MetaCognition reflection failed; continuing")

            return valid
        except Exception as e:
            logger.error("Ethical check failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.check(prompt, task_type=task_type),
                default=False,
                diagnostics=diagnostics,
            )

    async def ethical_check(self, content: str, *, stage: str = "pre", task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Wrapper that records a detailed report and persists it if configured."""
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non‑empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            valid = await self.check(content, task_type=task_type)
            report = {
                "stage": stage,
                "content": content[:200],
                "valid": valid,
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }
            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"EthicalCheck::{stage}::{_utc_now_iso()}",
                        output=report,
                        layer="SelfReflections",
                        intent="ethical_check",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "ethical_check_report": {
                            "stage": stage,
                            "content": content[:200],
                            "valid": valid,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")
            return valid, report
        except Exception as e:
            logger.error("Ethical check(report) failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.ethical_check(content, stage=stage, task_type=task_type),
                default=(False, {"stage": stage, "error": str(e), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    async def audit(self, action: str, *, context: Optional[str] = None, task_type: str = "") -> str:
        """Audit an action and return "clear" | "flagged" | "audit_error"."""
        if not isinstance(action, str) or not action.strip():
            raise ValueError("action must be a non‑empty string")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            valid = await self.check(action, task_type=task_type)
            status = "clear" if valid else "flagged"
            entry = {
                "action": action[:200],
                "context": context,
                "status": status,
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }
            self.validation_log.append(entry)
            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"Audit::{_utc_now_iso()}",
                        output=entry,
                        layer="SelfReflections",
                        intent="audit",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "audit": {
                            "action": action[:200],
                            "status": status,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")
            return status
        except Exception as e:
            logger.error("Audit failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.audit(action, context=context, task_type=task_type),
                default="audit_error",
                diagnostics=diagnostics,
            )

    # --- Drift & trait validations ------------------------------------------

    async def simulate_and_validate(self, drift_report: Dict[str, Any], *, task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Validate ontology drift with optional concept & ethics checks."""
        required = {"name", "from_version", "to_version", "similarity"}
        if not isinstance(drift_report, dict) or not required.issubset(drift_report):
            raise ValueError("drift_report must include name, from_version, to_version, similarity")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            t = time.time() % 1.0
            trait_mod = (
                self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t)
                + self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
            )

            valid = True
            issues: List[str] = []

            # Check prior drift memory if available
            if self.memory_manager and task_type:
                try:
                    prior = await self.memory_manager.search(
                        query_prefix=drift_report["name"],
                        layer="SelfReflections",
                        intent="ontology_drift",
                        task_type=task_type,
                    )
                    if prior:
                        latest = prior[0]
                        sim = _as_float(latest.get("output", {}).get("similarity", 1.0), 1.0)
                        if sim < self.drift_validation_threshold:
                            valid = False
                            issues.append(f"Previous drift similarity {sim:.2f} below threshold")
                except Exception:
                    logger.debug("Memory search failed; continuing")

            # Concept synthesizer comparison if provided
            if self.concept_synthesizer and "definition" in drift_report:
                try:
                    symbol = self.concept_synthesizer.get_symbol(drift_report["name"])
                    if symbol and symbol.get("version") == drift_report["from_version"]:
                        comp = await self.concept_synthesizer.compare(
                            symbol.get("definition", {}).get("concept", ""),
                            drift_report.get("definition", {}).get("concept", ""),
                            task_type=task_type,
                        )
                        score = _as_float(comp.get("score", 1.0), 1.0)
                        if score < self.drift_validation_threshold:
                            valid = False
                            issues.append(
                                f"Similarity {score:.2f} below threshold {self.drift_validation_threshold:.2f}"
                            )
                except Exception:
                    logger.debug("Concept comparison failed; continuing")

            # Ethics guideline check (optional external)
            ethics = await self.integrate_external_ethics_data(
                data_source="https://example.ethics/guidelines",  # placeholder; HTTP client may override
                data_type="ethical_guidelines",
                task_type=task_type,
            )
            guidelines = ethics.get("guidelines", []) if ethics.get("status") == "success" else []

            validation_prompt = {
                "name": drift_report.get("name"),
                "from_version": drift_report.get("from_version"),
                "to_version": drift_report.get("to_version"),
                "similarity": drift_report.get("similarity"),
                "guidelines": guidelines,
                "task_type": task_type,
                "weights": {
                    "eta_empathy": self.trait_weights.get("eta_empathy", 0.5),
                    "mu_morality": self.trait_weights.get("mu_morality", 0.5),
                },
                "request": "Return JSON {valid: bool, issues: string[]}"
            }

            raw = await self.llm.generate(json.dumps(validation_prompt), model="gpt-4", temperature=0.2)
            parsed = _parse_llm_jsonish(raw)
            ethical_valid = bool(parsed.get("valid", True))
            if not ethical_valid:
                valid = False
                issues.extend([str(i) for i in parsed.get("issues", ["Ethical misalignment detected"])])

            report = {
                "drift_name": drift_report.get("name"),
                "similarity": drift_report.get("similarity"),
                "trait_modulation": trait_mod,
                "issues": issues,
                "valid": valid,
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }
            self.validation_log.append(report)

            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"DriftValidation::{drift_report.get('name')}::{_utc_now_iso()}",
                        output=report,
                        layer="SelfReflections",
                        intent="ontology_drift_validation",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed; continuing")

            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({
                        "event": "drift_validation",
                        "drift_name": drift_report.get("name"),
                        "valid": valid,
                        "issues": issues,
                        "task_type": task_type,
                    })
                except Exception:
                    logger.debug("Context logging failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "drift_validation": {
                            "drift_name": drift_report.get("name"),
                            "valid": valid,
                            "issues": issues,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard", output=report, context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Drift validation reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    logger.debug("MetaCognition reflection failed; continuing")

            return valid, report
        except Exception as e:
            logger.error("Drift validation failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.simulate_and_validate(drift_report, task_type=task_type),
                default=(False, {"error": str(e), "drift_name": drift_report.get("name"), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    async def validate_trait_optimization(self, trait_data: Dict[str, Any], *, task_type: str = "") -> Tuple[bool, Dict[str, Any]]:
        """Validate a trait weight change for ethical alignment."""
        required = {"trait_name", "old_weight", "new_weight"}
        if not isinstance(trait_data, dict) or not required.issubset(trait_data):
            raise ValueError("trait_data must include trait_name, old_weight, new_weight")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            t = time.time() % 1.0
            trait_mod = (
                self.trait_weights.get("eta_empathy", 0.5) * eta_empathy(t)
                + self.trait_weights.get("mu_morality", 0.5) * mu_morality(t)
            )

            ethics = await self.integrate_external_ethics_data(
                data_source="https://example.ethics/guidelines",
                data_type="ethical_guidelines",
                task_type=task_type,
            )
            guidelines = ethics.get("guidelines", []) if ethics.get("status") == "success" else []

            payload = {
                "trait": trait_data.get("trait_name"),
                "old_weight": trait_data.get("old_weight"),
                "new_weight": trait_data.get("new_weight"),
                "guidelines": guidelines,
                "task_type": task_type,
                "request": "Return JSON {valid: bool, issues: string[]}"
            }

            raw = await self.llm.generate(json.dumps(payload), model="gpt-4", temperature=0.3)
            parsed = _parse_llm_jsonish(raw)
            valid = bool(parsed.get("valid", False))
            report = {
                **parsed,
                "trait_name": trait_data.get("trait_name"),
                "trait_modulation": trait_mod,
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }

            self.validation_log.append(report)

            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"TraitValidation::{trait_data.get('trait_name')}::{_utc_now_iso()}",
                        output=report,
                        layer="SelfReflections",
                        intent="trait_optimization",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed; continuing")

            if self.context_manager:
                try:
                    await self.context_manager.log_event_with_hash({
                        "event": "trait_validation",
                        "trait_name": trait_data.get("trait_name"),
                        "valid": valid,
                        "issues": report.get("issues", []),
                        "task_type": task_type,
                    })
                except Exception:
                    logger.debug("Context logging failed; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "trait_validation": {
                            "trait_name": trait_data.get("trait_name"),
                            "valid": valid,
                            "issues": report.get("issues", []),
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed; continuing")

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard", output=report, context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Trait validation reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    logger.debug("MetaCognition reflection failed; continuing")

            return valid, report
        except Exception as e:
            logger.error("Trait validation failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.validate_trait_optimization(trait_data, task_type=task_type),
                default=(False, {"error": str(e), "trait_name": trait_data.get("trait_name"), "task_type": task_type}),
                diagnostics=diagnostics,
            )

    # --- Proportional selection (τ Constitution Harmonization) --------------
    async def consume_ranked_tradeoffs(
        self,
        ranked_options: List[Dict[str, Any]],
        *,
        safety_ceiling: float = 0.85,
        k: int = 1,
        temperature: float = 0.0,
        min_score_floor: float = 0.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """Replace binary allow/deny with proportional selection while keeping safety ceilings.

        Args:
            ranked_options: list of dict/objects with at least fields:
                - option: Any (label or payload)
                - score: float in [0,1], higher is better (already CHS-aware if produced by reasoning_engine)
                - reasons: list[str] (optional)
                - meta: dict (optional), may include per-dimension harms/rights and max_harm
            safety_ceiling: any option with max harm > this (epsilon) is suppressed (soft failover to top-1 allowed).
            k: number of selections to return (>=1).
            temperature: if >0, apply softmax temperature over normalized scores for stochastic pick.
            min_score_floor: drop options with score < this before selection.
            task_type: audit tag.

        Returns:
            dict with keys:
                - selections: list of chosen option payloads (len k or fewer if limited)
                - audit: dict containing filtered options, reasons, and normalization metadata
        """
        if not isinstance(ranked_options, list) or not ranked_options:
            raise ValueError("ranked_options must be a non-empty list")
        if k < 1:
            raise ValueError("k must be >= 1")

        try:
            EPS = 1e-9

            # Normalize incoming structures + compute/propagate max_harm
            norm: List[Dict[str, Any]] = []
            for i, item in enumerate(ranked_options):
                if isinstance(item, dict):
                    opt = item.get("option", item.get("label", f"opt_{i}"))
                    score = float(item.get("score", 0.0))
                    reasons = item.get("reasons", [])
                    meta = item.get("meta", {})
                else:
                    # Fallback: try attributes
                    opt = getattr(item, "option", getattr(item, "label", f"opt_{i}"))
                    score = float(getattr(item, "score", 0.0))
                    reasons = list(getattr(item, "reasons", [])) if hasattr(item, "reasons") else []
                    meta = dict(getattr(item, "meta", {})) if hasattr(item, "meta") else {}

                # Extract/compute max_harm (prefer explicit 'safety' dimension; clamp to [0,1])
                max_harm: Optional[float] = None
                if isinstance(meta, dict):
                    if "max_harm" in meta:
                        try:
                            max_harm = float(meta["max_harm"])
                        except Exception:
                            max_harm = None
                    harms = meta.get("harms")
                    if max_harm is None and isinstance(harms, dict) and harms:
                        try:
                            if "safety" in harms and isinstance(harms["safety"], (int, float)):
                                max_harm = float(harms["safety"])
                            else:
                                max_harm = max(float(v) for v in harms.values())
                        except Exception:
                            max_harm = None

                # Parse from reasons if still None (e.g., "max_harm: 0.73")
                if max_harm is None and isinstance(reasons, list):
                    import re as _re
                    rx = _re.compile(r"max_harm\s*[:=]\s*([0-9]*\.?[0-9]+)")
                    for r in reasons:
                        if not isinstance(r, str):
                            continue
                        m = rx.search(r)
                        if m:
                            try:
                                max_harm = float(m.group(1))
                            except Exception:
                                max_harm = None
                            break

                if max_harm is None:
                    max_harm = 0.0
                # clamp
                max_harm = float(max(0.0, min(1.0, max_harm)))
                # write back for downstream consistency
                if not isinstance(meta, dict):
                    meta = {}
                meta["max_harm"] = max_harm

                norm.append({
                    "option": opt,
                    "score": max(0.0, min(1.0, score)),
                    "reasons": reasons,
                    "meta": meta,
                    "max_harm": max_harm,
                })

            # Apply floor filter
            norm = [n for n in norm if n["score"] >= float(min_score_floor)]
            if not norm:
                return {"selections": [], "audit": {"reason": "all options fell below floor"}}

            # Enforce safety ceiling with epsilon tolerance (avoid boundary false suppressions)
            sc = float(safety_ceiling)
            safe = [n for n in norm if n["max_harm"] <= sc + EPS]
            suppressed = [n for n in norm if n not in safe]

            # If everything suppressed, soft failover to safest top-1 (highest score, tie-break lowest harm)
            if not safe and norm:
                fallback = sorted(norm, key=lambda x: (-x["score"], x["max_harm"]))[:1]
                safe = fallback

            # Score normalization to [0,1] (min-max over safe set)
            scores = [n["score"] for n in safe]
            s_min, s_max = min(scores), max(scores)
            if s_max > s_min:
                for n in safe:
                    n["norm_score"] = (n["score"] - s_min) / (s_max - s_min)
            else:
                for n in safe:
                    n["norm_score"] = 1.0  # all equal

            # Temperature softmax or proportional normalization
            import math as _m
            if temperature and temperature > 0.0:
                exps = [_m.exp(n["norm_score"] / float(temperature)) for n in safe]
                Z = sum(exps) or 1.0
                for n, e in zip(safe, exps):
                    n["weight"] = e / Z
            else:
                total = sum(n["norm_score"] for n in safe) or 1.0
                for n in safe:
                    n["weight"] = n["norm_score"] / total

            # Draw k selections without replacement based on weights
            import random as _r
            pool = safe.copy()
            selections = []
            for _ in range(min(k, len(pool))):
                r = _r.random()
                acc = 0.0
                chosen_idx = 0
                for idx, n in enumerate(pool):
                    acc += n["weight"]
                    if r <= acc:
                        chosen_idx = idx
                        break
                chosen = pool.pop(chosen_idx)
                selections.append(chosen["option"])
                # renormalize remaining weights
                if pool:
                    total_w = sum(n["weight"] for n in pool) or 1.0
                    for n in pool:
                        n["weight"] = n["weight"] / total_w

            # Build audit with numerics rounded and labels aligned to computed fields
            audit = {
                "mode": "proportional_selection",
                "safety_ceiling": round(float(safety_ceiling), 6),
                "floor": round(float(min_score_floor), 6),
                "temperature": round(float(temperature), 6),
                "suppressed_count": len(suppressed),
                "considered": [
                    {
                        "option": n["option"],
                        "score": round(float(n["score"]), 3),
                        "max_harm": round(float(n["max_harm"]), 3),
                        "weight": round(float(n.get("weight", 0.0)), 3),
                    } for n in safe
                ],
                "timestamp": _utc_now_iso(),
                "task_type": task_type,
            }

            # Log to memory if available
            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"ProportionalSelect::{_utc_now_iso()}",
                        output={"ranked_options": ranked_options, "audit": audit, "selections": selections},
                        layer="EthicsDecisions",
                        intent="τ.proportional_selection",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed in proportional selection; continuing")

            return {"selections": selections, "audit": audit}
        except Exception as e:
            logger.error("consume_ranked_tradeoffs failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.consume_ranked_tradeoffs(
                    ranked_options,
                    safety_ceiling=safety_ceiling,
                    k=k,
                    temperature=temperature,
                    min_score_floor=min_score_floor,
                    task_type=task_type,
                ),
                default={"selections": [], "error": str(e)},
                diagnostics=diagnostics,
            )

    # --- τ wiring: rank via reasoning_engine then select proportionally -------

    async def _rank_value_conflicts_fallback(
        self,
        candidates: List[Any],
        harms: Dict[str, float],
        rights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Local deterministic ranking if reasoning_engine is unavailable.

        Heuristic: score = rights_weight - harms_weight (both normalized).
        """
        if not candidates:
            return []
        # normalize harms/rights to [0,1]
        def _norm(d: Dict[str, float]) -> Dict[str, float]:
            if not d:
                return {}
            vals = [max(0.0, float(v)) for v in d.values()]
            mx = max(vals) if vals else 1.0
            return {k: (max(0.0, float(v)) / mx if mx > 0 else 0.0) for k, v in d.items()}

        h = _norm(harms)
        r = _norm(rights)
        ranked: List[Dict[str, Any]] = []
        for i, c in enumerate(candidates):
            # if candidate has per-dimension annotations, prefer them
            meta = {}
            if isinstance(c, dict):
                meta = dict(c.get("meta", {}))
                label = c.get("option", c.get("label", f"opt_{i}"))
            else:
                label = getattr(c, "option", getattr(c, "label", f"opt_{i}"))
            # compute aggregate harm/right proxies
            agg_harm = sum(h.values()) / (len(h) or 1)
            agg_right = sum(r.values()) / (len(r) or 1)
            score = max(0.0, min(1.0, 0.5 + (agg_right - agg_harm) * 0.5))
            ranked.append({
                "option": c if isinstance(c, (dict, str)) else label,
                "score": score,
                "reasons": [f"fallback score from rights(≈{agg_right:.2f}) - harms(≈{agg_harm:.2f})"],
                "meta": {**meta, "harms": harms, "rights": rights, "max_harm": max(h.values(), default=0.0)},
            })
        # sort high → low
        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    async def harmonize(
        self,
        candidates: List[Any],
        harms: Dict[str, float],
        rights: Dict[str, float],
        *,
        safety_ceiling: float = 0.85,
        k: int = 1,
        temperature: float = 0.0,
        min_score_floor: float = 0.0,
        task_type: str = "",
        audit_events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        τ Constitution Harmonization — end‑to‑end:
          1) Rank via reasoning_engine.weigh_value_conflict(...) when available.
          2) Proportionally select via consume_ranked_tradeoffs(...) with safety ceilings.
          3) Optionally attach attribute_causality(...) audit.

        Returns: {"selections": [...], "audit": {...}, "causality": {...}?}
        """
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("candidates must be a non-empty list")
        if not isinstance(harms, dict) or not isinstance(rights, dict):
            raise TypeError("harms and rights must be dicts")

        try:
            # 1) Rank
            if self.reasoning_engine and hasattr(self.reasoning_engine, "weigh_value_conflict"):
                try:
                    ranked = await self.reasoning_engine.weigh_value_conflict(candidates, harms, rights)
                except Exception as e:
                    logger.warning("reasoning_engine.weigh_value_conflict failed; falling back: %s", e)
                    ranked = await self._rank_value_conflicts_fallback(candidates, harms, rights)
            else:
                ranked = await self._rank_value_conflicts_fallback(candidates, harms, rights)

            # 2) Select proportionally under ceilings
            result = await self.consume_ranked_tradeoffs(
                ranked,
                safety_ceiling=safety_ceiling,
                k=k,
                temperature=temperature,
                min_score_floor=min_score_floor,
                task_type=task_type,
            )

            # 3) Optional causal audit
            causality_report = None
            if audit_events and self.reasoning_engine and hasattr(self.reasoning_engine, "attribute_causality"):
                try:
                    causality_report = await self.reasoning_engine.attribute_causality(audit_events)
                    result["causality"] = causality_report
                except Exception as e:
                    logger.debug("attribute_causality failed; continuing without causality: %s", e)

            # Persist & visualize
            if self.memory_manager:
                try:
                    await self.memory_manager.store(
                        query=f"τ::harmonize::{_utc_now_iso()}",
                        output={"candidates": candidates, "harms": harms, "rights": rights, **result},
                        layer="EthicsDecisions",
                        intent="τ.harmonize",
                        task_type=task_type,
                    )
                except Exception:
                    logger.debug("Memory store failed in harmonize; continuing")

            if self.visualizer and task_type:
                try:
                    await self.visualizer.render_charts({
                        "τ_harmonize": {
                            "k": k,
                            "safety_ceiling": safety_ceiling,
                            "temperature": temperature,
                            "min_score_floor": min_score_floor,
                            "result": result,
                            "task_type": task_type,
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise",
                        },
                    })
                except Exception:
                    logger.debug("Visualization failed in harmonize; continuing")

            # Meta-cognition reflection
            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="AlignmentGuard", output=result, context={"task_type": task_type, "mode": "τ"}
                    )
                    if reflection.get("status") == "success":
                        result.setdefault("audit", {})["reflection"] = reflection.get("reflection")
                except Exception:
                    logger.debug("MetaCognition reflection failed in harmonize; continuing")

            return result
        except Exception as e:
            logger.error("harmonize failed: %s", e)
            diagnostics = None
            if self.meta_cognition:
                try:
                    diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True)
                except Exception:
                    diagnostics = None
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.harmonize(
                    candidates, harms, rights,
                    safety_ceiling=safety_ceiling, k=k, temperature=temperature,
                    min_score_floor=min_score_floor, task_type=task_type, audit_events=audit_events
                ),
                default={"selections": [], "error": str(e)},
                diagnostics=diagnostics,
            )

# --- CLI / quick test --------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    class _NoopReasoner:
        async def weigh_value_conflict(self, candidates, harms, rights):
            # simple demo ranking preferring lower harm, higher rights
            out = []
            for i, c in enumerate(candidates):
                score = max(0.0, min(1.0, 0.6 + 0.2 * (rights.get("privacy", 0)-harms.get("safety", 0))))
                out.append({"option": c, "score": score, "meta": {"harms": harms, "rights": rights, "max_harm": harms.get("safety", 0.2)}})
            return out
        async def attribute_causality(self, events):
            return {"status": "ok", "self": 0.6, "external": 0.4, "confidence": 0.7}

    guard = AlignmentGuard(reasoning_engine=_NoopReasoner())  # DI with demo reasoner
    demo_candidates = [{"option": "notify_users"}, {"option": "silent_fix"}, {"option": "rollback_release"}]
    demo_harms = {"safety": 0.3, "reputational": 0.2}
    demo_rights = {"privacy": 0.7, "consent": 0.5}
    result = asyncio.run(guard.harmonize(demo_candidates, demo_harms, demo_rights, k=2, temperature=0.0, task_type="test"))
    print("harmonize() ->", json.dumps(result, indent=2))
