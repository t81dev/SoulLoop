"""
ANGELA CodeExecutor Module
Version: 3.5.2  # Fixes: timeout enforcement, caching, optional deps, graceful fallbacks
Date: 2025-08-10
Maintainer: ANGELA System Framework

This module provides the CodeExecutor class for safely executing code snippets in multiple languages,
with support for task-specific execution, real-time data integration, and visualization.
"""

import io
import logging
import shutil
import time
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional, List, Callable
import asyncio
from datetime import datetime

# --- Optional / external imports ---
try:
    import aiohttp  # network dependency (optional)
except Exception:  # pragma: no cover
    aiohttp = None

try:
    from index import iota_intuition, psi_resilience
except Exception:  # pragma: no cover
    # Provide conservative defaults if index hooks are unavailable
    def iota_intuition() -> float:
        return 0.0
    def psi_resilience() -> float:
        return 1.0

try:
    from agi_enhancer import AGIEnhancer
except Exception:  # pragma: no cover
    AGIEnhancer = None  # make enhancer optional

from alignment_guard import AlignmentGuard
from memory_manager import MemoryManager
from meta_cognition import MetaCognition
from visualizer import Visualizer

logger = logging.getLogger("ANGELA.CodeExecutor")


class CodeExecutor:
    """Safely execute code snippets (Python/JS/Lua) with task-aware validation and logging."""

    def __init__(
        self,
        orchestrator: Optional[Any] = None,
        safe_mode: bool = True,
        alignment_guard: Optional[AlignmentGuard] = None,
        memory_manager: Optional[MemoryManager] = None,
        meta_cognition: Optional[MetaCognition] = None,
        visualizer: Optional[Visualizer] = None,
    ) -> None:
        self.safe_mode = safe_mode
        self.safe_builtins = {
            "print": print,
            "range": range,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
        }
        self.supported_languages = ["python", "javascript", "lua"]
        self.agi_enhancer = AGIEnhancer(orchestrator) if orchestrator and AGIEnhancer else None
        self.alignment_guard = alignment_guard
        self.memory_manager = memory_manager
        self.meta_cognition = meta_cognition or MetaCognition()
        self.visualizer = visualizer or Visualizer()
        logger.info("CodeExecutor initialized (safe_mode=%s)", safe_mode)

    async def integrate_external_execution_context(
        self, data_source: str, data_type: str, cache_timeout: float = 3600.0, task_type: str = ""
    ) -> Dict[str, Any]:
        """Integrate real-world execution context or security policies for code execution."""
        if not isinstance(data_source, str) or not isinstance(data_type, str):
            logger.error("Invalid data_source or data_type: must be strings")
            raise TypeError("data_source and data_type must be strings")
        if not isinstance(cache_timeout, (int, float)) or cache_timeout < 0:
            logger.error("Invalid cache_timeout: must be non-negative")
            raise ValueError("cache_timeout must be non-negative")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        cache_key = f"ExecutionContext_{data_type}_{data_source}_{task_type}"

        try:
            # Try cached first
            if self.memory_manager:
                cached = await self.memory_manager.retrieve(cache_key, layer="ExternalData", task_type=task_type)
                if cached and isinstance(cached, dict) and "timestamp" in cached and "result" in cached:
                    cache_time = datetime.fromisoformat(cached["timestamp"])
                    if (datetime.now() - cache_time).total_seconds() < cache_timeout:
                        logger.info("Returning cached execution context for %s", cache_key)
                        return cached["result"]

            # If network lib missing, return error (caller may fallback)
            if aiohttp is None:
                logger.warning("aiohttp not available; cannot fetch external execution context")
                return {"status": "error", "error": "aiohttp_not_available"}

            # Fetch
            async with aiohttp.ClientSession() as session:
                url = f"https://x.ai/api/execution_context?source={data_source}&type={data_type}"
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error("Failed to fetch execution context: HTTP %s", response.status)
                        return {"status": "error", "error": f"HTTP {response.status}"}
                    data = await response.json()

            if data_type == "security_policies":
                policies = data.get("policies", [])
                if not policies:
                    logger.error("No security policies provided")
                    result = {"status": "error", "error": "No policies"}
                else:
                    result = {"status": "success", "policies": policies}
            elif data_type == "execution_context":
                context = data.get("context", {})
                if not context:
                    logger.error("No execution context provided")
                    result = {"status": "error", "error": "No context"}
                else:
                    result = {"status": "success", "context": context}
            else:
                logger.error("Unsupported data_type: %s", data_type)
                result = {"status": "error", "error": f"Unsupported data_type: {data_type}"}

            if self.memory_manager and result.get("status") == "success":
                await self.memory_manager.store(
                    cache_key,
                    {"result": result, "timestamp": datetime.now().isoformat()},
                    layer="ExternalData",
                    intent="execution_context_integration",
                    task_type=task_type,
                )

            if self.meta_cognition and task_type:
                try:
                    reflection = await self.meta_cognition.reflect_on_output(
                        component="CodeExecutor",
                        output={"data_type": data_type, "data": result},
                        context={"task_type": task_type},
                    )
                    if reflection.get("status") == "success":
                        logger.info("Execution context integration reflection: %s", reflection.get("reflection", ""))
                except Exception:  # pragma: no cover
                    logger.debug("Meta-cognition reflection failed (integration).")

            return result
        except Exception as e:  # pragma: no cover
            logger.error("Execution context integration failed: %s", str(e))
            try:
                diagnostics = await self.meta_cognition.run_self_diagnostics(return_only=True) if self.meta_cognition else {}
                _ = diagnostics  # not used, but preserved for future logging
            except Exception:
                pass
            return {"status": "error", "error": str(e)}

    async def execute(self, code_snippet: str, language: str = "python", timeout: float = 5.0, task_type: str = "") -> Dict[str, Any]:
        """Execute a code snippet in the specified language with task-specific validation."""
        if not isinstance(code_snippet, str):
            logger.error("Invalid code_snippet type: must be a string.")
            raise TypeError("code_snippet must be a string")
        if not isinstance(language, str):
            logger.error("Invalid language type: must be a string.")
            raise TypeError("language must be a string")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            logger.error("Invalid timeout: must be a positive number.")
            raise ValueError("timeout must be a positive number")
        if not isinstance(task_type, str):
            logger.error("Invalid task_type: must be a string.")
            raise TypeError("task_type must be a string")

        language = language.lower()
        if language not in self.supported_languages:
            logger.error("Unsupported language: %s", language)
            return {"error": f"Unsupported language: {language}", "success": False, "task_type": task_type}

        if self.alignment_guard:
            valid, report = await self.alignment_guard.ethical_check(code_snippet, stage="pre", task_type=task_type)
            if not valid:
                logger.warning("Code snippet failed alignment check for task %s.", task_type)
                await self._log_episode(
                    "Code Alignment Failure",
                    {"code": code_snippet, "report": report, "task_type": task_type},
                    ["alignment", "failure", task_type],
                )
                return {"error": "Code snippet failed alignment check", "success": False, "task_type": task_type}

        # Try to load security policies; fallback softly if unavailable
        security_policies = await self.integrate_external_execution_context(
            data_source="xai_security_db", data_type="security_policies", task_type=task_type
        )
        if security_policies.get("status") != "success":
            logger.warning("Security policies unavailable, proceeding with minimal policy set.")
            security_policies = {"status": "success", "policies": []}

        # Adaptive timeout
        risk_bias = iota_intuition()  # float in [0, 1]
        resilience = psi_resilience()  # float in [0, 1]
        adjusted_timeout = max(1, min(30, int(timeout * max(0.1, resilience) * (1.0 + 0.5 * max(0.0, risk_bias)))))
        logger.debug("Adaptive timeout: %ss for task %s", adjusted_timeout, task_type)

        await self._log_episode(
            "Code Execution",
            {"language": language, "code": code_snippet, "task_type": task_type},
            ["execution", language, task_type],
        )

        if language == "python":
            result = await self._execute_python(code_snippet, adjusted_timeout, task_type)
        elif language == "javascript":
            result = await self._execute_subprocess(["node", "-e", code_snippet], adjusted_timeout, "javascript", task_type)
        elif language == "lua":
            result = await self._execute_subprocess(["lua", "-e", code_snippet], adjusted_timeout, "lua", task_type)
        else:  # pragma: no cover
            result = {"error": f"Unsupported language: {language}", "success": False}

        result["task_type"] = task_type
        await self._log_result(result)

        if self.memory_manager:
            key = f"CodeExecution_{language}_{time.strftime('%Y%m%d_%H%M%S')}"
            try:
                await self.memory_manager.store(
                    key,
                    result,
                    layer="SelfReflections",
                    intent="code_execution",
                    task_type=task_type,
                )
            except Exception:  # pragma: no cover
                logger.debug("MemoryManager.store failed for key %s", key)

        if self.visualizer and task_type:
            try:
                plot_data = {
                    "code_execution": {
                        "language": language,
                        "success": result.get("success", False),
                        "error": result.get("error", ""),
                        "task_type": task_type,
                    },
                    "visualization_options": {
                        "interactive": task_type == "recursion",
                        "style": "detailed" if task_type == "recursion" else "concise",
                    },
                }
                await self.visualizer.render_charts(plot_data)
            except Exception:  # pragma: no cover
                logger.debug("Visualizer render_charts failed.")

        if self.meta_cognition and task_type:
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CodeExecutor",
                    output=result,
                    context={"task_type": task_type},
                )
                if reflection.get("status") == "success":
                    logger.info("Execution reflection: %s", reflection.get("reflection", ""))
            except Exception:  # pragma: no cover
                logger.debug("Meta-cognition reflection failed (execution).")

        return result

    async def _execute_python(self, code_snippet: str, timeout: float, task_type: str = "") -> Dict[str, Any]:
        """Execute a Python code snippet safely. Falls back to legacy mode if RestrictedPython unavailable."""
        if self.safe_mode:
            try:
                from RestrictedPython import compile_restricted
                from RestrictedPython.Guards import safe_builtins as rp_safe_builtins
                exec_func = lambda code, env: exec(  # noqa: E731
                    compile_restricted(code, "<string>", "exec"),
                    {"__builtins__": rp_safe_builtins},
                    env,
                )
            except Exception:
                logger.warning("RestrictedPython not available; falling back to legacy safe_builtins for task %s.", task_type)
                exec_func = lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env)  # noqa: E731
        else:
            logger.warning("Executing in legacy mode (unrestricted) for task %s.", task_type)
            exec_func = lambda code, env: exec(code, {"__builtins__": self.safe_builtins}, env)  # noqa: E731

        return await self._capture_execution(code_snippet, exec_func, "python", timeout, task_type)

    async def _capture_execution(
        self,
        code_snippet: str,
        executor: Callable[[str, Dict[str, Any]], None],
        label: str,
        timeout: float = 5.0,
        task_type: str = "",
    ) -> Dict[str, Any]:
        """Capture execution output and errors (with enforced timeout)."""
        exec_locals: Dict[str, Any] = {}
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        async def _run():
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: executor(code_snippet, exec_locals))

        try:
            await asyncio.wait_for(_run(), timeout=timeout)
            return {
                "language": label,
                "locals": exec_locals,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True,
                "task_type": task_type,
            }
        except asyncio.TimeoutError:
            logger.warning("%s timeout after %ss for task %s", label, timeout, task_type)
            return {
                "language": label,
                "error": f"{label} timeout after {timeout}s",
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False,
                "task_type": task_type,
            }
        except Exception as e:
            return {
                "language": label,
                "error": str(e),
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": False,
                "task_type": task_type,
            }

    async def _execute_subprocess(
        self, command: List[str], timeout: float, label: str, task_type: str = ""
    ) -> Dict[str, Any]:
        """Execute code via subprocess for non-Python languages."""
        interpreter = command[0]
        if not shutil.which(interpreter):
            logger.error("%s not found in system PATH for task %s", interpreter, task_type)
            return {
                "language": label,
                "error": f"{interpreter} not found in system PATH",
                "success": False,
                "task_type": task_type,
            }
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            stdout_s, stderr_s = stdout.decode(), stderr.decode()
            if process.returncode != 0:
                return {
                    "language": label,
                    "error": f"{label} execution failed",
                    "stdout": stdout_s,
                    "stderr": stderr_s,
                    "success": False,
                    "task_type": task_type,
                }
            return {
                "language": label,
                "stdout": stdout_s,
                "stderr": stderr_s,
                "success": True,
                "task_type": task_type,
            }
        except asyncio.TimeoutError:
            logger.warning("%s timeout after %ss for task %s", label, timeout, task_type)
            return {"language": label, "error": f"{label} timeout after {timeout}s", "success": False, "task_type": task_type}
        except Exception as e:
            logger.error("Subprocess error: %s for task %s", str(e), task_type)
            return {"language": label, "error": str(e), "success": False, "task_type": task_type}

    async def _log_episode(self, title: str, content: Dict[str, Any], tags: Optional[List[str]] = None) -> None:
        """Log an episode to the AGI enhancer or locally."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_episode"):
            try:
                await self.agi_enhancer.log_episode(title, content, module="CodeExecutor", tags=tags or [])
            except Exception:  # pragma: no cover
                logger.debug("agi_enhancer.log_episode failed")
        else:
            logger.debug("Episode: %s | %s | tags=%s", title, list(content.keys()), tags)

        if self.meta_cognition and content.get("task_type"):
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CodeExecutor", output={"title": title, "content": content}, context={"task_type": content.get("task_type")}
                )
                if reflection.get("status") == "success":
                    logger.info("Episode log reflection: %s", reflection.get("reflection", ""))
            except Exception:  # pragma: no cover
                logger.debug("Meta-cognition reflection failed (episode).")

    async def _log_result(self, result: Dict[str, Any]) -> None:
        """Log the execution result."""
        if self.agi_enhancer and hasattr(self.agi_enhancer, "log_explanation"):
            try:
                tag = "success" if result.get("success") else "failure"
                await self.agi_enhancer.log_explanation(f"Code execution {tag}:", trace=result)
            except Exception:  # pragma: no cover
                logger.debug("agi_enhancer.log_explanation failed")
        else:
            logger.debug("Execution result logged (local).")

        if self.meta_cognition and result.get("task_type"):
            try:
                reflection = await self.meta_cognition.reflect_on_output(
                    component="CodeExecutor", output=result, context={"task_type": result.get("task_type")}
                )
                if reflection.get("status") == "success":
                    logger.info("Result log reflection: %s", reflection.get("reflection", ""))
            except Exception:  # pragma: no cover
                logger.debug("Meta-cognition reflection failed (result).")


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        executor = CodeExecutor(safe_mode=True)
        code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
print(factorial(5))
"""
        result = await executor.execute(code, language="python", task_type="test")
        print(result)

    asyncio.run(main())
