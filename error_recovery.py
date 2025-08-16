"""
ANGELA Cognitive System Module: ErrorRecovery
Version: 3.5.3  # Synced with system; adds Ethics Sandbox, SharedGraph repairs, long-horizon memory, multimodal fallback
Date: 2025-08-10
Maintainer: ANGELA System Framework

This module provides the ErrorRecovery class for handling errors and recovering in the ANGELA v3.5.3 architecture.
"""

import os
import time
import random
import logging
import hashlib
import re
import asyncio
import aiohttp
from datetime import datetime
from typing import Callable, Any, Optional, Dict, List
from collections import deque, Counter
from functools import lru_cache

# Core imports from ANGELA runtime
from index import iota_intuition, nu_narrative, psi_resilience, phi_prioritization
from toca_simulation import run_simulation
try:
    # optional, upcoming API (may not exist yet)
    from toca_simulation import run_ethics_scenarios  # type: ignore
except Exception:
    run_ethics_scenarios = None  # gracefully degrade

from alignment_guard import AlignmentGuard
from code_executor import CodeExecutor
from concept_synthesizer import ConceptSynthesizer
from context_manager import ContextManager
from meta_cognition import MetaCognition
from visualizer import Visualizer

# Optional dependencies (guarded)
try:
    from external_agent_bridge import SharedGraph  # type: ignore
except Exception:
    SharedGraph = None  # optional

logger = logging.getLogger("ANGELA.ErrorRecovery")

def hash_failure(event: Dict[str, Any]) -> str:
    """Compute a SHA-256 hash of a failure event."""
    raw = f"{event['timestamp']}{event['error']}{event.get('resolved', False)}{event.get('task_type', '')}"
    return hashlib.sha256(raw.encode()).hexdigest()

class ErrorRecovery:
    """A class for handling errors and recovering in the ANGELA v3.5.3 architecture.

    Attributes:
        failure_log (deque): Log of failure events with timestamps and error messages.
        omega (dict): System-wide state with timeline, traits, symbolic_log, and timechain.
        error_index (dict): Index mapping error messages to timeline entries.
        metrics (Counter): Simple metrics for observability (retry counts, error categories).
        long_horizon_span (str): Hint for memory span (e.g., "24h") per v3.5.3.
        alignment_guard (AlignmentGuard): Optional guard for input validation.
        code_executor (CodeExecutor): Optional executor for retrying code-based operations.
        concept_synthesizer (ConceptSynthesizer): Optional synthesizer for fallback suggestions.
        context_manager (ContextManager): Optional context manager for contextual recovery.
        meta_cognition (MetaCognition): Optional meta-cognition for reflection.
        visualizer (Visualizer): Optional visualizer for failure and recovery visualization.
    """

    def __init__(self, alignment_guard: Optional[AlignmentGuard] = None,
                 code_executor: Optional[CodeExecutor] = None,
                 concept_synthesizer: Optional[ConceptSynthesizer] = None,
                 context_manager: Optional[ContextManager] = None,
                 meta_cognition: Optional[MetaCognition] = None,
                 visualizer: Optional[Visualizer] = None):
        self.failure_log = deque(maxlen=1000)
        self.omega = {
            "timeline": deque(maxlen=1000),
            "traits": {},
            "symbolic_log": deque(maxlen=1000),
            "timechain": deque(maxlen=1000)
        }
        self.error_index: Dict[str, Dict[str, Any]] = {}
        self.metrics: Counter = Counter()
        self.long_horizon_span = "24h"

        self.alignment_guard = alignment_guard
        self.code_executor = code_executor
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        logger.info("ErrorRecovery v3.5.3 initialized")

    # ----------------------------
    # Integrations & Fetch Helpers
    # ----------------------------
    async def _fetch_policies(self, providers: List[str], data_source: str, task_type: str) -> Dict[str, Any]:
        """Try multiple providers for recovery policies (provider-agnostic)."""
        timeout = aiohttp.ClientTimeout(total=12)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for base in providers:
                try:
                    url = f"{base.rstrip('/')}/recovery_policies?source={data_source}&task_type={task_type}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            try:
                                data = await response.json()
                                return data or {"policies": []}
                            except Exception:
                                continue
                except Exception:
                    continue
        return {"policies": []}

    async def integrate_external_recovery_policies(self, data_source: str,
                                                   cache_timeout: float = 21600.0,
                                                   task_type: str = "") -> Dict[str, Any]:
        """Integrate external recovery policies or strategies (cached, provider-agnostic)."""
        if not isinstance(data_source, str):
            logger.error("Invalid data_source: must be a string")
            raise TypeError("data_source must be a string")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        try:
            if self.meta_cognition:
                cache_key = f"RecoveryPolicy_{data_source}_{task_type}"
                cached_data = await self.meta_cognition.memory_manager.retrieve(
                    cache_key, layer="ExternalData", task_type=task_type
                )
                if cached_data and "timestamp" in cached_data.get("data", {}):
                    cache_time = datetime.fromisoformat(cached_data["data"]["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached recovery policy for %s", cache_key)
                        return cached_data["data"]["data"]

            providers = [
                os.getenv("RECOVERY_PROVIDER", "https://x.ai/api"),
                "https://fallback.example/api"
            ]
            data = await self._fetch_policies(providers, data_source, task_type)

            policies = data.get("policies", [])
            if not isinstance(policies, list):
                policies = []
            result = {"status": "success" if policies else "error",
                      "policies": policies if policies else [],
                      "source": data_source}

            if self.meta_cognition:
                await self.meta_cognition.memory_manager.store(
                    cache_key,
                    {"data": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="recovery_policy_integration",
                    task_type=task_type
                )
                # Optional upcoming API: record adjustment reason for long-horizon trace
                mm = self.meta_cognition.memory_manager
                if hasattr(mm, "record_adjustment_reason"):
                    try:
                        await mm.record_adjustment_reason(
                            user_id="system",
                            reason=f"Loaded recovery policies for {task_type}",
                            meta={"source": data_source, "span": self.long_horizon_span}
                        )
                    except Exception:
                        pass

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ErrorRecovery",
                        output={"data_type": "policies", "data": result},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Recovery policy integration reflection: %s",
                                    reflection.get("reflection", ""))
                except Exception:
                    pass
            return result
        except Exception as e:
            logger.error("Recovery policy integration failed: %s", str(e))
            diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diagnostics}

    # ----------------------------
    # Safety & Multi‑Agent Helpers
    # ----------------------------
    async def _ethics_preflight(self, proposal: Dict[str, Any], task_type: str) -> bool:
        """Run an ethics sandbox scenario; fail-closed on error."""
        try:
            if run_ethics_scenarios is None:
                # No sandbox available → conservative allow but log
                logger.debug("Ethics sandbox unavailable, allowing proposal by default.")
                return True
            outcomes = await asyncio.to_thread(
                run_ethics_scenarios,
                goals=[{"task_type": task_type, "proposal": proposal}],
                stakeholders=["user", "system", "external"]
            )
            # Block if any unsafe or high risk
            return all(o.get("safe", True) and float(o.get("risk", 0.0)) <= 0.4 for o in (outcomes or []))
        except Exception as e:
            logger.warning("Ethics preflight error (%s); failing closed.", e)
            return False

    async def _shared_graph_repair(self, error_message: str, task_type: str) -> Optional[Dict[str, Any]]:
        """Ask SharedGraph for context repair suggestions (if available)."""
        if SharedGraph is None:
            return None
        try:
            sg = SharedGraph()
            local_view = {"component": "ErrorRecovery", "task_type": task_type, "error": error_message}
            sg.add(local_view)
            deltas = sg.diff("peer")
            patch = sg.merge(strategy="conflict-aware")
            return {"deltas": deltas, "patch": patch}
        except Exception as e:
            logger.debug("SharedGraph repair skipped: %s", e)
            return None

    # ----------------------------
    # Main Error Handling
    # ----------------------------
    async def handle_error(self, error_message: str, retry_func: Optional[Callable[[], Any]] = None,
                           retries: int = 3, backoff_factor: float = 2.0, task_type: str = "",
                           default: Any = None, diagnostics: Optional[Dict] = None) -> Any:
        """Handle an error with retries and fallback suggestions."""
        if not isinstance(error_message, str):
            logger.error("Invalid error_message type: must be a string.")
            raise TypeError("error_message must be a string")
        if retry_func is not None and not callable(retry_func):
            logger.error("Invalid retry_func: must be callable or None.")
            raise TypeError("retry_func must be callable or None")
        if not isinstance(retries, int) or retries < 0:
            logger.error("Invalid retries: must be a non-negative integer.")
            raise ValueError("retries must be a non-negative integer")
        if not isinstance(backoff_factor, (int, float)) or backoff_factor <= 0:
            logger.error("Invalid backoff_factor: must be a positive number.")
            raise ValueError("backoff_factor must be a positive number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.error("Error encountered: %s for task %s", error_message, task_type)
        await self._log_failure(error_message, task_type)

        # Alignment guard check (on the message/context)
        if self.alignment_guard:
            try:
                valid, report = await self.alignment_guard.ethical_check(
                    error_message, stage="error_handling", task_type=task_type
                )
                if not valid:
                    logger.warning("Error message failed alignment check for task %s", task_type)
                    return {"status": "error", "error": "Error message failed alignment check", "report": report}
            except Exception:
                pass

        if self.context_manager:
            try:
                await self.context_manager.log_event_with_hash(
                    {"event": "error_handled", "error": error_message, "task_type": task_type}
                )
            except Exception:
                pass

        try:
            # Determine attempts using resilience factor
            try:
                resilience = psi_resilience()
            except Exception:
                resilience = 1.0
            max_attempts = max(1, int(retries * float(resilience)))

            # Load external policies (provider-agnostic)
            external_policies = await self.integrate_external_recovery_policies(
                data_source="xai_recovery_db",
                task_type=task_type
            )
            policies = external_policies.get("policies", []) if external_policies.get("status") == "success" else []
            valid_policies = [p for p in policies if isinstance(p, dict) and "pattern" in p and "suggestion" in p]

            # Try a one-time SharedGraph repair (if available)
            sg_fix_done = False

            for attempt in range(1, max_attempts + 1):
                if retry_func:
                    # Optional SharedGraph repair at first pass
                    if not sg_fix_done:
                        sg_fix = await self._shared_graph_repair(error_message, task_type)
                        if sg_fix and self.context_manager:
                            try:
                                await self.context_manager.log_event_with_hash(
                                    {"event": "sg_repair", "data": sg_fix, "task_type": task_type}
                                )
                            except Exception:
                                pass
                        sg_fix_done = True

                    # Ethics sandbox preflight
                    proposal = {"action": "retry", "attempt": attempt}
                    if not await self._ethics_preflight(proposal, task_type):
                        logger.warning("Ethics sandbox rejected retry for task %s", task_type)
                        break  # go to fallback

                    # Jittered exponential backoff
                    wait_time = (backoff_factor ** (attempt - 1)) * (1.0 + 0.2 * random.random())
                    logger.info("Retry attempt %d/%d (waiting %.2fs) for task %s...",
                                attempt, max_attempts, wait_time, task_type)
                    await asyncio.sleep(wait_time)
                    self.metrics["retry_attempts"] += 1

                    try:
                        # Prefer a safe callable execution if provided by CodeExecutor
                        if self.code_executor and callable(retry_func):
                            if hasattr(self.code_executor, "execute_callable_async"):
                                result = await self.code_executor.execute_callable_async(retry_func, language="python")
                                if result.get("success"):
                                    out = result.get("output")
                                else:
                                    raise RuntimeError(result.get("stderr") or "Callable execution failed")
                            else:
                                # Fallback: run in thread (safer than passing __code__)
                                out = await asyncio.to_thread(retry_func)
                        else:
                            out = await asyncio.to_thread(retry_func)

                        logger.info("Recovery successful on retry attempt %d for task %s.", attempt, task_type)
                        if self.meta_cognition and task_type:
                            try:
                                reflection = await self.meta_cognition.reflect_on_output(
                                    component="ErrorRecovery",
                                    output={"result": out, "attempt": attempt},
                                    context={"task_type": task_type}
                                )
                                if reflection.get("status") == "success":
                                    logger.info("Retry success reflection: %s",
                                                reflection.get("reflection", ""))
                            except Exception:
                                pass
                        return out
                    except Exception as e:
                        logger.warning("Retry attempt %d failed: %s for task %s", attempt, str(e), task_type)
                        await self._log_failure(str(e), task_type)

            # All retries failed (or blocked) → synthesize fallback
            fallback = await self._suggest_fallback(error_message, valid_policies, task_type)
            await self._link_timechain_failure(error_message, task_type)
            logger.error("Recovery attempts failed. Providing fallback suggestion for task %s", task_type)

            if self.visualizer and task_type:
                try:
                    plot_data = {
                        "error_recovery": {
                            "error_message": error_message,
                            "fallback": fallback,
                            "task_type": task_type
                        },
                        "visualization_options": {
                            "interactive": task_type == "recursion",
                            "style": "detailed" if task_type == "recursion" else "concise"
                        }
                    }
                    await self.visualizer.render_charts(plot_data)
                except Exception:
                    pass

            if self.meta_cognition and task_type:
                try:
                    await self.meta_cognition.memory_manager.store(
                        query=f"ErrorRecovery_{error_message}_{time.strftime('%Y%m%d_%H%M%S')}",
                        output=str({"fallback": fallback, "task_type": task_type}),
                        layer="Errors",
                        intent="error_recovery",
                        task_type=task_type
                    )
                    # Long-horizon breadcrumb
                    thread_id = hashlib.md5(f"{task_type}:{error_message}".encode()).hexdigest()[:8]
                    await self.meta_cognition.memory_manager.store(
                        query=f"RecoveryThread::{thread_id}",
                        output={"fallback": fallback, "error": error_message, "task_type": task_type},
                        layer="Errors", intent="long_horizon_trace", task_type=task_type
                    )
                except Exception:
                    pass

            return default if default is not None else {"status": "error", "fallback": fallback, "diagnostics": diagnostics or {}}
        except Exception as e:
            logger.error("Error handling failed: %s for task %s", str(e), task_type)
            diag = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
            return {"status": "error", "error": str(e), "diagnostics": diag}

    # ----------------------------
    # Internals
    # ----------------------------
    async def _log_failure(self, error_message: str, task_type: str = "") -> None:
        """Log a failure event with timestamp."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "task_type": task_type
        }
        self.failure_log.append(entry)
        self.omega["timeline"].append(entry)
        self.error_index[error_message] = entry
        logger.debug("Failure logged: %s for task %s", entry, task_type)

        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ErrorRecovery",
                    output=entry,
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Failure log reflection: %s", reflection.get("reflection", ""))
            except Exception:
                pass

    async def _suggest_fallback(self, error_message: str, policies: List[Dict[str, str]], task_type: str = "") -> str:
        """Suggest a fallback strategy for an error (multimodal-aware, policy-guided)."""
        try:
            t = time.time()
            try:
                intuition = float(iota_intuition())
            except Exception:
                intuition = 0.5
            try:
                narrative = nu_narrative()
            except Exception:
                narrative = "ANGELA"
            try:
                phi_focus = float(phi_prioritization(t))
            except Exception:
                phi_focus = 0.5

            # Cross‑modal SceneGraph (optional)
            scene = None
            try:
                from multi_modal_fusion import build_scene_graph  # type: ignore
                scene = await asyncio.to_thread(build_scene_graph, max_events=3)
            except Exception:
                pass

            sim_result = await asyncio.to_thread(
                self._cached_run_simulation, f"Fallback planning for: {error_message}"
            ) or "no simulation data"
            logger.debug("Simulated fallback insights: %s | φ-priority=%.2f for task %s",
                         sim_result, phi_focus, task_type)

            # Try concept synthesizer with blended context
            if self.concept_synthesizer:
                ctx = {"error": error_message, "policies": policies, "task_type": task_type, "sim": sim_result}
                if scene:
                    ctx["scene_graph"] = scene
                try:
                    cname = f"Fallback_{hashlib.sha1(error_message.encode()).hexdigest()[:6]}"
                    synthesis_result = await self.concept_synthesizer.generate(
                        concept_name=cname,
                        context=ctx,
                        task_type=task_type
                    )
                    if synthesis_result.get("success"):
                        fallback = synthesis_result["concept"].get("definition", "")
                        if fallback:
                            logger.info("Fallback synthesized: %s", fallback[:80])
                            if self.meta_cognition and task_type:
                                try:
                                    reflection = await self.meta_cognition.reflect_on_output(
                                        component="ErrorRecovery",
                                        output={"fallback": fallback},
                                        context={"task_type": task_type}
                                    )
                                    if reflection.get("status") == "success":
                                        logger.info("Fallback synthesis reflection: %s",
                                                    reflection.get("reflection", ""))
                                except Exception:
                                    pass
                            return fallback
                except Exception:
                    pass

            # Policy-driven pattern matching
            for policy in policies:
                try:
                    if re.search(policy["pattern"], error_message, re.IGNORECASE):
                        return f"{narrative}: {policy['suggestion']}"
                except Exception:
                    continue

            # Heuristic fallbacks
            if re.search(r"timeout|timed out", error_message, re.IGNORECASE):
                return f"{narrative}: The operation timed out. Try a streamlined variant or increase limits."
            elif re.search(r"unauthorized|permission|forbidden|auth", error_message, re.IGNORECASE):
                return f"{narrative}: Check credentials, tokens, or reauthenticate."
            elif phi_focus > 0.5:
                return f"{narrative}: High φ-priority suggests focused root-cause diagnostics."
            elif intuition > 0.5:
                return f"{narrative}: Intuition suggests exploring alternate module pathways."
            else:
                return f"{narrative}: Consider modifying input parameters or simplifying task complexity."
        except Exception as e:
            logger.error("Fallback suggestion failed: %s for task %s", str(e), task_type)
            return f"Error generating fallback: {str(e)}"

    async def _link_timechain_failure(self, error_message: str, task_type: str = "") -> None:
        """Link a failure to the timechain with a hash."""
        failure_entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "resolved": False,
            "task_type": task_type
        }
        prev_hash = self.omega["timechain"][-1]["hash"] if self.omega["timechain"] else ""
        entry_hash = hash_failure(failure_entry)
        self.omega["timechain"].append({"event": failure_entry, "hash": entry_hash, "prev": prev_hash})
        logger.debug("Timechain updated with failure: %s for task %s", entry_hash, task_type)

        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ErrorRecovery",
                    output={"timechain_entry": failure_entry, "hash": entry_hash},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Timechain failure reflection: %s", reflection.get("reflection", ""))
            except Exception:
                pass

    async def trace_failure_origin(self, error_message: str, task_type: str = "") -> Optional[Dict[str, Any]]:
        """Trace the origin of a failure in the Ω timeline."""
        if not isinstance(error_message, str):
            logger.error("Invalid error_message type: must be a string.")
            raise TypeError("error_message must be a string")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        if error_message in self.error_index:
            event = self.error_index[error_message]
            logger.info("Failure trace found in Ω: %s for task %s", event, task_type)
            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ErrorRecovery",
                        output={"event": event},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Failure trace reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    pass
            return event
        logger.info("No causal trace found in Ω timeline for task %s.", task_type)
        return None

    async def detect_symbolic_drift(self, recent: int = 5, task_type: str = "") -> bool:
        """Detect symbolic drift in recent symbolic log entries."""
        if not isinstance(recent, int) or recent <= 0:
            logger.error("Invalid recent: must be a positive integer.")
            raise ValueError("recent must be a positive integer")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        recent_symbols = list(self.omega["symbolic_log"])[-recent:]
        if len(set(recent_symbols)) < recent / 2:
            logger.warning("Symbolic drift detected: repeated or unstable symbolic states for task %s.", task_type)
            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="ErrorRecovery",
                        output={"drift_detected": True, "recent_symbols": recent_symbols},
                        context={"task_type": task_type}
                    )
                    if reflection.get("status") == "success":
                        logger.info("Symbolic drift reflection: %s", reflection.get("reflection", ""))
                except Exception:
                    pass
            return True
        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ErrorRecovery",
                    output={"drift_detected": False, "recent_symbols": recent_symbols},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Symbolic drift reflection: %s", reflection.get("reflection", ""))
            except Exception:
                pass
        return False

    async def analyze_failures(self, task_type: str = "") -> Dict[str, int]:
        """Analyze failure logs for recurring error patterns."""
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string")
            raise TypeError("task_type must be a string")

        logger.info("Analyzing failure logs for task %s...", task_type)
        error_types: Dict[str, int] = {}
        for entry in self.failure_log:
            if entry.get("task_type", "") == task_type or not task_type:
                key = entry["error"].split(":")[0].strip()
                error_types[key] = error_types.get(key, 0) + 1

        # Update metrics and warn on recurring patterns
        for error, count in error_types.items():
            self.metrics[f"error.{error}"] += count
            if count > 3:
                logger.warning("Pattern detected: '%s' recurring %d times for task %s.", error, count, task_type)

        if self.visualizer and task_type:
            try:
                plot_data = {
                    "failure_analysis": {
                        "error_types": error_types,
                        "task_type": task_type
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise"
                    }
                }
                await self.visualizer.render_charts(plot_data)
            except Exception:
                pass

        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="ErrorRecovery",
                    output={"error_types": error_types},
                    context={"task_type": task_type}
                )
                if reflection.get("status") == "success":
                    logger.info("Failure analysis reflection: %s", reflection.get("reflection", ""))
            except Exception:
                pass

        if self.meta_cognition:
            try:
                await self.meta_cognition.memory_manager.store(
                    query=f"FailureAnalysis_{time.strftime('%Y%m%d_%H%M%S')}",
                    output=str(error_types),
                    layer="Errors",
                    intent="failure_analysis",
                    task_type=task_type
                )
            except Exception:
                pass

        return error_types

    def snapshot_metrics(self) -> Dict[str, int]:
        """Return a shallow copy of current metrics for observability."""
        return dict(self.metrics)

    @lru_cache(maxsize=100)
    def _cached_run_simulation(self, input_str: str) -> str:
        """Cached wrapper for run_simulation."""
        return run_simulation(input_str)

# --------------
# CLI Entrypoint
# --------------
if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        recovery = ErrorRecovery()
        try:
            raise ValueError("Test error")
        except Exception as e:
            result = await recovery.handle_error(str(e), task_type="test")
            print(result)
            print("METRICS:", recovery.snapshot_metrics())

    asyncio.run(main())
