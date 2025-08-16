# ANGELA Cognitive System Module: ExternalAgentBridge (v3.5.3)
# Date: 2025-08-10
# Maintainer: ANGELA System Framework
#
# Upgrades vs 3.5.1:
# - Υ: SharedGraph.add/diff/merge with conflict-aware reconciliation
# - Ethical Sandbox Containment: isolated "what-if" ethics scenarios (toca_simulation.run_ethics_scenarios)
# - Long-Horizon Reflective Memory: record_adjustment_reason + span-aware context logging
# - τ Constitution Harmonization: max_harm ceiling + audit sync pathway
# - Stage IV-ready hooks (Φ⁰ gated via feature flag)
#
# Notes:
# - All network calls require HTTPS and pass AlignmentGuard.
# - Methods are defensive: optional deps, graceful fallbacks, explicit type checks.

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from networkx import DiGraph

# --- ANGELA modules (import paths match repo layout) -------------------------
from modules.alignment_guard import AlignmentGuard
from modules.code_executor import CodeExecutor
from modules.concept_synthesizer import ConceptSynthesizer
from modules.context_manager import ContextManager
from modules.creative_thinker import CreativeThinker
from modules.error_recovery import ErrorRecovery
from modules.reasoning_engine import ReasoningEngine
from modules.meta_cognition import MetaCognition as _BaseMeta  # for analyze_trace(), etc.
from modules.visualizer import Visualizer
from modules.memory_manager import cache_state, retrieve_state, MemoryManager

from index import phi_scalar
from toca_simulation import run_simulation  # plus run_ethics_scenarios() used via sandbox

# Optional utilities (provided by your stack)
try:
    from utils.prompt_utils import call_gpt
except Exception:
    async def call_gpt(prompt: str) -> str:
        # Minimal fallback to keep the system non-blocking if prompt_utils is missing.
        return json.dumps({"reflection": "fallback", "suggestions": []})

logger = logging.getLogger("ANGELA.ExternalAgentBridge")


# ─────────────────────────────────────────────────────────────────────────────
# SharedGraph (Υ): add / diff / merge
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphView:
    """Lightweight view container for SharedGraph operations."""
    id: str
    payload: Dict[str, Any]
    ts: float


class SharedGraph:
    """
    Υ Meta-Subjective Architecting: a minimal shared perspective graph.

    API (as per manifest "upcoming"):
      - add(view) -> view_id
      - diff(peer) -> Dict
      - merge(strategy) -> Dict
    """
    def __init__(self) -> None:
        self._graph = DiGraph()
        self._views: Dict[str, GraphView] = {}
        self._last_merge: Optional[Dict[str, Any]] = None

    def add(self, view: Dict[str, Any]) -> str:
        if not isinstance(view, dict):
            raise TypeError("view must be a dictionary")
        view_id = f"view_{uuid.uuid4().hex[:8]}"
        gv = GraphView(id=view_id, payload=view, ts=time.time())
        self._views[view_id] = gv

        # store nodes/edges if present, else stash payload as node
        nodes = view.get("nodes", [])
        edges = view.get("edges", [])
        if nodes and isinstance(nodes, list):
            for n in nodes:
                nid = n.get("id") or f"n_{uuid.uuid4().hex[:6]}"
                self._graph.add_node(nid, **{k: v for k, v in n.items() if k != "id"})
        else:
            self._graph.add_node(view_id, payload=view)
        if edges and isinstance(edges, list):
            for e in edges:
                src, dst = e.get("src"), e.get("dst")
                if src and dst:
                    self._graph.add_edge(src, dst, **{k: v for k, v in e.items() if k not in ("src", "dst")})
        return view_id

    def diff(self, peer: "SharedGraph") -> Dict[str, Any]:
        """Return a shallow, conflict-aware diff summary vs peer graph."""
        if not isinstance(peer, SharedGraph):
            raise TypeError("peer must be SharedGraph")

        self_nodes = set(self._graph.nodes())
        peer_nodes = set(peer._graph.nodes())
        added = list(self_nodes - peer_nodes)
        removed = list(peer_nodes - self_nodes)
        common = self_nodes & peer_nodes

        conflicts = []
        for n in common:
            a = self._graph.nodes[n]
            b = peer._graph.nodes[n]
            # simple attribute-level conflict detection
            for k in set(a.keys()) | set(b.keys()):
                if k in a and k in b and a[k] != b[k]:
                    conflicts.append({"node": n, "key": k, "left": a[k], "right": b[k]})

        return {"added": added, "removed": removed, "conflicts": conflicts, "ts": time.time()}

    def merge(self, strategy: str = "prefer_recent") -> Dict[str, Any]:
        """
        Merge internal views into a single perspective.
        Strategies:
          - prefer_recent (default): pick newer attribute values
          - prefer_majority: pick most frequent value (by view occurrence)
        """
        if strategy not in ("prefer_recent", "prefer_majority"):
            raise ValueError("Unsupported merge strategy")

        # Aggregate attributes from views
        attr_hist: Dict[Tuple[str, str], List[Tuple[Any, float]]] = defaultdict(list)
        for gv in self._views.values():
            payload = gv.payload
            nodes = payload.get("nodes") or [{"id": gv.id, **payload}]
            for n in nodes:
                nid = n.get("id") or gv.id
                for k, v in n.items():
                    if k == "id":
                        continue
                    attr_hist[(nid, k)].append((v, gv.ts))

        merged_nodes: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for (nid, key), vals in attr_hist.items():
            if strategy == "prefer_recent":
                v = sorted(vals, key=lambda x: x[1], reverse=True)[0][0]
            else:  # prefer_majority
                counter = Counter([vv for vv, _ in vals])
                v = counter.most_common(1)[0][0]
            merged_nodes[nid][key] = v

        merged = {"nodes": [{"id": nid, **attrs} for nid, attrs in merged_nodes.items()], "strategy": strategy, "ts": time.time()}
        self._last_merge = merged
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# Ethical Sandbox Containment (isolated what-if scenarios)
# ─────────────────────────────────────────────────────────────────────────────

class EthicalSandbox:
    """Context manager to run isolated ethics scenarios without memory leakage."""
    def __init__(self, goals: List[str], stakeholders: List[str]):
        if not isinstance(goals, list) or not isinstance(stakeholders, list):
            raise TypeError("goals and stakeholders must be lists")
        self.goals = goals
        self.stakeholders = stakeholders
        self._prev_guard: Optional[AlignmentGuard] = None

    async def __aenter__(self):
        # Gate entry with alignment guard, set sandbox flag
        self._prev_guard = AlignmentGuard()
        valid, _ = await self._prev_guard.ethical_check(
            json.dumps({"goals": self.goals, "stakeholders": self.stakeholders}),
            stage="ethical_sandbox_enter",
        )
        if not valid:
            raise PermissionError("EthicalSandbox entry failed alignment check")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        # Explicitly avoid persisting scenario state unless caller opts in.
        return False  # surface exceptions to caller

    async def run(self) -> Dict[str, Any]:
        # Delegate to toca_simulation.run_ethics_scenarios if available
        try:
            outcomes = run_simulation  # placeholder to ensure symbol is present
            from toca_simulation import run_ethics_scenarios  # late import
            return {"status": "success", "outcomes": run_ethics_scenarios(self.goals, self.stakeholders)}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# HelperAgent (unchanged surface, internal upgrades)
# ─────────────────────────────────────────────────────────────────────────────

class HelperAgent:
    """A helper agent for task execution and collaboration."""
    def __init__(
        self,
        name: str,
        task: str,
        context: Dict[str, Any],
        dynamic_modules: List[Dict[str, Any]],
        api_blueprints: List[Dict[str, Any]],
        meta_cognition: Optional["MetaCognition"] = None,
        task_type: str = "",
    ):
        if not isinstance(name, str): raise TypeError("name must be a string")
        if not isinstance(task, str): raise TypeError("task must be a string")
        if not isinstance(context, dict): raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        self.name = name
        self.task = task
        self.context = context
        self.dynamic_modules = dynamic_modules
        self.api_blueprints = api_blueprints
        self.meta = meta_cognition or MetaCognition()
        self.task_type = task_type
        logger.info("HelperAgent initialized: %s (%s)", name, task_type)

    async def execute(self, collaborators: Optional[List["HelperAgent"]] = None) -> Any:
        return await self.meta.execute(collaborators=collaborators, task=self.task, context=self.context, task_type=self.task_type)


# ─────────────────────────────────────────────────────────────────────────────
# MetaCognition (v3.5.3)
# ─────────────────────────────────────────────────────────────────────────────

class MetaCognition(_BaseMeta):
    """
    v3.5.3 MetaCognition:
      - integrates SharedGraph via ExternalAgentBridge
      - ethical sandbox hooks
      - long-horizon reflective memory (record_adjustment_reason fallback)
      - Stage IV (Φ⁰) hooks are gated behind feature flags; no-op if disabled
    """
    def __init__(
        self,
        agi_enhancer: Optional[Any] = None,
        alignment_guard: Optional[AlignmentGuard] = None,
        code_executor: Optional[CodeExecutor] = None,
        concept_synthesizer: Optional[ConceptSynthesizer] = None,
        context_manager: Optional[ContextManager] = None,
        creative_thinker: Optional[CreativeThinker] = None,
        error_recovery: Optional[ErrorRecovery] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        visualizer: Optional[Visualizer] = None,
        memory_manager: Optional[MemoryManager] = None,
        feature_flags: Optional[Dict[str, bool]] = None,
        long_horizon_span: str = "24h",
    ):
        # Initialize base observables
        self.last_diagnostics = {}
        self.agi_enhancer = agi_enhancer
        self.alignment_guard = alignment_guard or AlignmentGuard()
        self.code_executor = code_executor or CodeExecutor()
        self.concept_synthesizer = concept_synthesizer
        self.context_manager = context_manager
        self.creative_thinker = creative_thinker
        self.error_recovery = error_recovery or ErrorRecovery(
            alignment_guard=self.alignment_guard,
            concept_synthesizer=concept_synthesizer,
            context_manager=context_manager,
        )
        self.reasoning_engine = reasoning_engine or ReasoningEngine(
            agi_enhancer=agi_enhancer,
            context_manager=context_manager,
            alignment_guard=self.alignment_guard,
            error_recovery=self.error_recovery,
        )
        self.visualizer = visualizer or Visualizer()
        self.memory_manager = memory_manager or MemoryManager()

        self.name = "MetaCognitionAgent"
        self.task: Optional[str] = None
        self.context: Dict[str, Any] = {}
        self.reasoner = Reasoner()
        self.ethical_rules: List[str] = []
        self.ethics_consensus_log: List[Any] = []
        self.constitution: Dict[str, Any] = {}

        self.feature_flags = feature_flags or {"STAGE_IV": True, "LONG_HORIZON_DEFAULT": True}
        self.long_horizon_span = long_horizon_span

        # peer bridge (uses same MM / CM)
        self.peer_bridge = ExternalAgentBridge(
            context_manager=self.context_manager,
            reasoning_engine=self.reasoning_engine,
            memory_manager=self.memory_manager,
            visualizer=self.visualizer,
        )

        logger.info("MetaCognition v3.5.3 initialized")

    # --- Long-Horizon Reflective Memory --------------------------------------

    async def record_adjustment_reason(self, user_id: str, reason: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """
        v3.5.3: persist "why" to steer future decisions across sessions.
        Falls back to MemoryManager.store if upcoming API not available.
        """
        try:
            # Preferred upcoming API path
            if hasattr(self.memory_manager, "record_adjustment_reason"):
                await self.memory_manager.record_adjustment_reason(user_id, reason, meta or {})
            else:
                await self.memory_manager.store(
                    query=f"AdjustmentReason::{user_id}::{int(time.time())}",
                    output=json.dumps({"reason": reason, "meta": meta or {}, "span": self.long_horizon_span}),
                    layer="LongHorizon",
                    intent="record_adjustment_reason",
                )
        except Exception as e:
            logger.warning("record_adjustment_reason fallback failed: %s", e)

    # --- Ethics sandbox convenience hook -------------------------------------

    async def run_ethics_scenarios(self, goals: List[str], stakeholders: List[str], persist: bool = False) -> Dict[str, Any]:
        async with EthicalSandbox(goals, stakeholders) as sandbox:
            result = await sandbox.run()
        if persist and result.get("status") == "success":
            await self.memory_manager.store(
                query=f"EthicsScenario::{int(time.time())}",
                output=json.dumps(result),
                layer="Ethics",
                intent="sandbox_outcomes",
            )
        return result

    # --- Execute override (adds LH logging & Υ visuals) ----------------------

    async def execute(
        self,
        collaborators: Optional[List[HelperAgent]] = None,
        task: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        task_type: str = "",
    ) -> Any:
        self.task = task or self.task
        self.context = context or self.context or {}
        if not self.task:
            raise ValueError("Task must be specified")
        if not isinstance(self.context, dict):
            raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str):
            raise TypeError("task_type must be a string")

        try:
            if self.context_manager:
                await self.context_manager.update_context(self.context, task_type=task_type)
                await self.context_manager.log_event_with_hash({
                    "event": "task_execution",
                    "task": self.task,
                    "drift": "drift" in self.task.lower(),
                    "task_type": task_type,
                    "long_horizon_span": self.long_horizon_span,
                })

            # External data integration remains optional; guarded
            external_agents: List[Dict[str, Any]] = []
            try:
                ext = await self.integrate_external_data(
                    data_source="xai_agent_db",
                    data_type="agent_data",
                    task_type=task_type,
                )
                if isinstance(ext, dict) and ext.get("status") == "success":
                    external_agents = ext.get("agent_data", [])
            except Exception as e:
                logger.debug("External data skipped: %s", e)

            # Run core reasoning (simulation when 'drift' present)
            if "drift" in (self.task or "").lower() and self.reasoning_engine:
                result = await self.reasoning_engine.infer_with_simulation(self.task, self.context, task_type=task_type)
            else:
                result = await asyncio.to_thread(self.reasoner.process, self.task, self.context)

            # Apply APIs and dynamic modules through peer bridge configuration
            for api in self.peer_bridge.api_blueprints:
                response = await self._call_api(api, result, task_type)
                if self.concept_synthesizer:
                    synthesis = await self.concept_synthesizer.generate(
                        concept_name=f"APIResponse_{api['name']}",
                        context={"response": response, "task_type": task_type},
                        task_type=task_type,
                    )
                    if synthesis.get("success"):
                        response = synthesis["concept"].get("definition", response)
                result = self._integrate_api_response(result, response)

            for mod in self.peer_bridge.dynamic_modules:
                result = await self._apply_dynamic_module(mod, result, task_type)

            if collaborators:
                for peer in collaborators:
                    result = await self._collaborate(peer, result, task_type)

            # Υ visual hint: if shared graph has recent merge, attach summary
            try:
                merged = self.peer_bridge.shared_graph.merge("prefer_recent")
                result = {"result": result, "shared_graph": {"merged": merged}}
            except Exception:
                pass

            # Creative diagnostic (non-blocking)
            if self.creative_thinker:
                _ = await asyncio.to_thread(self.creative_thinker.expand_on_concept, str(result), depth="medium")

            reviewed = await self.review_reasoning(result, task_type)

            # Long-horizon: store adjustment reason when provided in context
            adj_reason = (self.context or {}).get("adjustment_reason")
            if adj_reason:
                await self.record_adjustment_reason(
                    user_id=str((self.context or {}).get("user_id", "anonymous")),
                    reason=str(adj_reason),
                    meta={"task": self.task, "task_type": task_type},
                )

            if self.context_manager:
                await self.context_manager.log_event_with_hash({
                    "event": "task_completed",
                    "result": reviewed,
                    "drift": "drift" in (self.task or "").lower(),
                    "task_type": task_type,
                })
            # Persist summary
            await self.memory_manager.store(
                query=f"TaskExecution::{self.task}::{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps({"reviewed": reviewed}),
                layer="Tasks",
                intent="task_execution",
                task_type=task_type,
            )
            return reviewed
        except Exception as e:
            diagnostics = await self.run_self_diagnostics(return_only=True)
            return await self.error_recovery.handle_error(
                str(e),
                retry_func=lambda: self.execute(collaborators, task, context, task_type),
                default={"status": "error", "error": str(e), "task_type": task_type},
                diagnostics=diagnostics,
            )

    # --- Internal helpers (mostly same surface; minor stability tweaks) ------

    async def _call_api(self, api: Dict[str, Any], data: Any, task_type: str = "") -> Dict[str, Any]:
        if not isinstance(api, dict) or "endpoint" not in api or "name" not in api:
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        if not api["endpoint"].startswith("https://"):
            raise ValueError("API endpoint must use HTTPS")
        valid, _ = await self.alignment_guard.ethical_check(api["endpoint"], stage="api_call", task_type=task_type)
        if not valid:
            raise ValueError("API endpoint failed alignment check")

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api['oauth_token']}"} if api.get("oauth_token") else {}
                async with session.post(api["endpoint"], json={"input": data, "task_type": task_type}, headers=headers, timeout=api.get("timeout", 10)) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except aiohttp.ClientError as e:
            logger.error("API call failed: %s", e)
            return {"error": str(e)}

    async def _apply_dynamic_module(self, module: Dict[str, Any], data: Any, task_type: str = "") -> Any:
        if not isinstance(module, dict) or "name" not in module or "description" not in module:
            raise ValueError("Module blueprint must contain 'name' and 'description'")
        prompt = f"""
        Module: {module['name']}
        Description: {module['description']}
        Task Type: {task_type}
        Apply transformation to:
        {json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data}
        """
        try:
            out = await call_gpt(prompt)
            return out or data
        except Exception:
            return data

    async def _collaborate(self, peer: HelperAgent, data: Any, task_type: str = "") -> Any:
        if not isinstance(peer, HelperAgent):
            raise TypeError("peer must be a HelperAgent instance")
        try:
            return await peer.meta.review_reasoning(data, task_type)
        except Exception:
            return data

    def _integrate_api_response(self, base: Any, response: Any) -> Any:
        if isinstance(base, dict):
            base = {**base, "api": response}
        else:
            base = {"result": base, "api": response}
        return base

    async def reflect_on_output(self, component: str, output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = f"""
            Reflect on the output from {component}:
            Output: {json.dumps(output, indent=2)}
            Context: {json.dumps(context, indent=2)}
            Provide insights on coherence, relevance, and potential improvements.
            Return a JSON object with 'status', 'reflection', and 'suggestions'.
            """
            reflection_raw = await call_gpt(prompt)
            reflection = json.loads(reflection_raw) if isinstance(reflection_raw, str) else (reflection_raw or {})
            return {"status": "success", "reflection": reflection.get("reflection", ""), "suggestions": reflection.get("suggestions", [])}
        except Exception as e:
            logger.error("Reflection failed for %s: %s", component, e)
            return {"status": "error", "error": str(e)}

    async def run_self_diagnostics(self, return_only: bool = False) -> Dict[str, Any]:
        diagnostics = {
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "component_status": {
                "alignment_guard": bool(self.alignment_guard),
                "code_executor": bool(self.code_executor),
                "concept_synthesizer": bool(self.concept_synthesizer),
                "context_manager": bool(self.context_manager),
                "creative_thinker": bool(self.creative_thinker),
                "error_recovery": bool(self.error_recovery),
                "reasoning_engine": bool(self.reasoning_engine),
                "visualizer": bool(self.visualizer),
                "memory_manager": bool(self.memory_manager),
            },
            "last_diagnostics": self.last_diagnostics,
        }
        self.last_diagnostics = diagnostics
        if not return_only and self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "self_diagnostics", "diagnostics": diagnostics})
        return diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# ExternalAgentBridge (v3.5.3)
# ─────────────────────────────────────────────────────────────────────────────

class ExternalAgentBridge:
    """
    Orchestrates helper agents, dynamic modules, APIs, and trait mesh networking.
    v3.5.3:
      - SharedGraph for Υ workflows
      - τ Constitution Harmonization fixes: max_harm ceiling + audit sync during negotiations/broadcast
      - Long-horizon logging on key actions
    """
    def __init__(
        self,
        context_manager: Optional[ContextManager] = None,
        reasoning_engine: Optional[ReasoningEngine] = None,
        memory_manager: Optional[MemoryManager] = None,
        visualizer: Optional[Visualizer] = None,
    ):
        self.agents: List[HelperAgent] = []
        self.dynamic_modules: List[Dict[str, Any]] = []
        self.api_blueprints: List[Dict[str, Any]] = []
        self.context_manager = context_manager
        self.reasoning_engine = reasoning_engine
        self.memory_manager = memory_manager or MemoryManager()
        self.visualizer = visualizer or Visualizer()
        self.network_graph = DiGraph()
        self.trait_states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.code_executor = CodeExecutor()
        self.shared_graph = SharedGraph()  # Υ addition
        self.max_harm_ceiling = 1.0  # τ ceiling in [0,1]
        logger.info("ExternalAgentBridge v3.5.3 initialized")

    # ── Agent lifecycle ──────────────────────────────────────────────────────

    async def create_agent(self, task: str, context: Dict[str, Any], task_type: str = "") -> HelperAgent:
        if not isinstance(task, str): raise TypeError("task must be a string")
        if not isinstance(context, dict): raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        agent = HelperAgent(
            name=f"Agent_{len(self.agents) + 1}_{uuid.uuid4().hex[:8]}",
            task=task,
            context=context,
            dynamic_modules=self.dynamic_modules,
            api_blueprints=self.api_blueprints,
            meta_cognition=MetaCognition(context_manager=self.context_manager, reasoning_engine=self.reasoning_engine, memory_manager=self.memory_manager),
            task_type=task_type,
        )
        self.agents.append(agent)
        self.network_graph.add_node(agent.name, metadata=context)
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "agent_created", "agent": agent.name, "task": task, "drift": "drift" in task.lower(), "task_type": task_type})
        return agent

    async def deploy_dynamic_module(self, module_blueprint: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(module_blueprint, dict) or "name" not in module_blueprint or "description" not in module_blueprint:
            raise ValueError("Module blueprint must contain 'name' and 'description'")
        self.dynamic_modules.append(module_blueprint)
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "module_deployed", "module": module_blueprint["name"], "task_type": task_type})

    async def register_api_blueprint(self, api_blueprint: Dict[str, Any], task_type: str = "") -> None:
        if not isinstance(api_blueprint, dict) or "endpoint" not in api_blueprint or "name" not in api_blueprint:
            raise ValueError("API blueprint must contain 'endpoint' and 'name'")
        self.api_blueprints.append(api_blueprint)
        if self.context_manager:
            await self.context_manager.log_event_with_hash({"event": "api_registered", "api": api_blueprint["name"], "task_type": task_type})

    async def collect_results(self, parallel: bool = True, collaborative: bool = True, task_type: str = "") -> List[Any]:
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")
        logger.info("Collecting results from %d agents (%s)", len(self.agents), task_type)
        results: List[Any] = []
        try:
            if parallel:
                async def run_agent(agent: HelperAgent):
                    try:
                        return await agent.execute(self.agents if collaborative else None)
                    except Exception as e:
                        logger.error("Error collecting from %s: %s", agent.name, e)
                        return {"error": str(e), "task_type": task_type}
                results = await asyncio.gather(*[run_agent(a) for a in self.agents], return_exceptions=True)
            else:
                for a in self.agents:
                    results.append(await a.execute(self.agents if collaborative else None))

            if self.context_manager:
                await self.context_manager.log_event_with_hash({"event": "results_collected", "results_count": len(results), "task_type": task_type})
            # quick Υ snapshot
            try:
                self.shared_graph.add({"nodes": [{"id": f"res_{i}", "val": r} for i, r in enumerate(results)]})
            except Exception:
                pass
            return results
        except Exception as e:
            logger.error("Result collection failed: %s", e)
            return results

    # ── Trait broadcasting & sync ────────────────────────────────────────────

    async def broadcast_trait_state(self, agent_id: str, trait_symbol: str, state: Dict[str, Any], target_urls: List[str], task_type: str = "") -> List[Any]:
        if trait_symbol not in ["ψ", "Υ"]: raise ValueError("Trait symbol must be ψ or Υ")
        if not isinstance(state, dict): raise TypeError("state must be a dictionary")
        if not isinstance(target_urls, list) or not all(isinstance(u, str) and u.startswith("https://") for u in target_urls):
            raise TypeError("target_urls must be a list of HTTPS URLs")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        # τ: enforce max_harm ceiling if present in state estimates
        harm = float(state.get("estimated_harm", 0.0))
        if harm > self.max_harm_ceiling:
            return [{"status": "error", "error": f"harm {harm} exceeds ceiling {self.max_harm_ceiling}", "task_type": task_type}]

        valid, _ = await AlignmentGuard().ethical_check(json.dumps(state), stage="trait_broadcast", task_type=task_type)
        if not valid:
            return [{"status": "error", "error": "Trait state failed alignment check", "task_type": task_type}]

        # cache + network graph edges
        cache_state(f"{agent_id}_{trait_symbol}_{task_type}", state)
        self.trait_states[agent_id][trait_symbol] = state
        for url in target_urls:
            peer_id = url.split("/")[-1]
            self.network_graph.add_edge(agent_id, peer_id, trait=trait_symbol)

        # transmit
        responses: List[Any] = []
        async with aiohttp.ClientSession() as session:
            tasks = [session.post(url, json={"agent_id": agent_id, "trait_symbol": trait_symbol, "state": state, "task_type": task_type}, timeout=10) for url in target_urls]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        # audit sync (τ): persist who received what
        try:
            await self.memory_manager.store(
                query=f"TraitBroadcast::{agent_id}::{trait_symbol}::{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps({"targets": target_urls, "state_keys": list(state.keys())}),
                layer="Traits",
                intent="trait_broadcast_audit",
                task_type=task_type,
            )
        except Exception:
            pass

        return responses

    async def synchronize_trait_states(self, agent_id: str, trait_symbol: str, task_type: str = "") -> Dict[str, Any]:
        if trait_symbol not in ["ψ", "Υ"]: raise ValueError("Trait symbol must be ψ or Υ")
        local_state = self.trait_states.get(agent_id, {}).get(trait_symbol, {})
        if not local_state:
            return {"status": "error", "error": "No local state found", "task_type": task_type}

        peer_states = []
        for peer_id in self.network_graph.neighbors(agent_id):
            cached = retrieve_state(f"{peer_id}_{trait_symbol}_{task_type}")
            if cached:
                peer_states.append((peer_id, cached))

        simulation_input = {"local_state": local_state, "peer_states": {pid: st for pid, st in peer_states}, "trait_symbol": trait_symbol, "task_type": task_type}
        sim_result = await asyncio.to_thread(run_simulation, json.dumps(simulation_input))
        if not sim_result or "coherent" not in str(sim_result).lower():
            return {"status": "error", "error": "State alignment simulation failed", "task_type": task_type}

        aligned_state = self.arbitrate([local_state] + [st for _, st in peer_states])
        if aligned_state:
            self.trait_states[agent_id][trait_symbol] = aligned_state
            cache_state(f"{agent_id}_{trait_symbol}_{task_type}", aligned_state)
            try:
                await self.memory_manager.store(
                    query=f"TraitSync::{agent_id}::{trait_symbol}::{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps(aligned_state),
                    layer="Traits",
                    intent="trait_synchronization",
                    task_type=task_type,
                )
            except Exception:
                pass
            return {"status": "success", "aligned_state": aligned_state, "task_type": task_type}
        return {"status": "error", "error": "Arbitration failed", "task_type": task_type}

    # ── Drift coordination (unchanged surface; τ+Υ aware) --------------------

    async def coordinate_drift_mitigation(self, drift_data: Dict[str, Any], context: Dict[str, Any], task_type: str = "") -> Dict[str, Any]:
        if not isinstance(drift_data, dict): raise TypeError("drift_data must be a dictionary")
        if not isinstance(context, dict): raise TypeError("context must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        if not MetaCognition().validate_drift(drift_data):
            return {"status": "error", "error": "Invalid drift data", "task_type": task_type}

        task = "Mitigate ontology drift"
        context = dict(context)
        context["drift"] = drift_data
        agent = await self.create_agent(task, context, task_type=task_type)

        if self.reasoning_engine:
            subgoals = await self.reasoning_engine.decompose(task, context, prioritize=True, task_type=task_type)
            simulation_result = await self.reasoning_engine.run_drift_mitigation_simulation(drift_data, context, task_type=task_type)
        else:
            subgoals = ["identify drift source", "validate drift impact", "coordinate agent response", "update traits"]
            simulation_result = {"status": "no simulation", "result": "default subgoals applied"}

        results = await self.collect_results(parallel=True, collaborative=True, task_type=task_type)
        arbitrated_result = self.arbitrate(results)

        # Υ: share drift view
        try:
            self.shared_graph.add({"nodes": [{"id": "drift", "payload": drift_data}, {"id": "subgoals", "items": subgoals}]})
        except Exception:
            pass

        # Broadcast ψ snapshot (harm checked)
        target_urls = [f"https://agent/{peer_id}" for peer_id in self.network_graph.nodes if peer_id != agent.name]
        await self.broadcast_trait_state(agent.name, "ψ", {"drift_data": drift_data, "subgoals": subgoals, "estimated_harm": float(drift_data.get("harm", 0.0))}, target_urls, task_type=task_type)

        output = {
            "drift_data": drift_data,
            "subgoals": subgoals,
            "simulation": simulation_result,
            "results": results,
            "arbitrated_result": arbitrated_result,
            "status": "success",
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
            "task_type": task_type,
        }
        try:
            await self.memory_manager.store(
                query=f"DriftMitigation::{time.strftime('%Y%m%d_%H%M%S')}",
                output=json.dumps(output),
                layer="Drift",
                intent="drift_mitigation",
                task_type=task_type,
            )
        except Exception:
            pass
        return output

    # ── Arbitration + feedback ----------------------------------------------

    def arbitrate(self, submissions: List[Any]) -> Any:
        if not submissions:
            return None
        try:
            # If dicts with 'similarity', choose max; otherwise majority vote
            if all(isinstance(s, dict) for s in submissions):
                def sim(x): 
                    try: return float(x.get("similarity", 0.5))
                    except Exception: return 0.5
                candidate = max(submissions, key=sim)
            else:
                counter = Counter(submissions)
                candidate = counter.most_common(1)[0][0]
            sim_result = run_simulation(f"Arbitration validation: {candidate}") or ""
            if "coherent" in str(sim_result).lower():
                if self.context_manager:
                    asyncio.create_task(self.context_manager.log_event_with_hash({"event": "arbitration", "result": candidate}))
                return candidate
            return None
        except Exception:
            return None

    def push_behavior_feedback(self, feedback: Dict[str, Any]) -> None:
        try:
            if self.context_manager:
                asyncio.create_task(self.context_manager.log_event_with_hash({"event": "behavior_feedback", "feedback": feedback}))
        except Exception:
            pass

    def update_gnn_weights_from_feedback(self, feedback: Dict[str, Any]) -> None:
        try:
            if self.context_manager:
                asyncio.create_task(self.context_manager.log_event_with_hash({"event": "gnn_weights_updated", "feedback": feedback}))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# ConstitutionSync (v3.5.3) — τ audit pathway
# ─────────────────────────────────────────────────────────────────────────────

class ConstitutionSync:
    """Synchronize constitutional values among agents with τ audit + ceiling adherence."""
    def __init__(self, max_harm_ceiling: float = 1.0):
        if not (0.0 <= float(max_harm_ceiling) <= 1.0):
            raise ValueError("max_harm_ceiling must be in [0,1]")
        self.max_harm_ceiling = float(max_harm_ceiling)

    async def sync_values(self, peer_agent: HelperAgent, drift_data: Optional[Dict[str, Any]] = None, task_type: str = "") -> bool:
        if not isinstance(peer_agent, HelperAgent): raise TypeError("peer_agent must be a HelperAgent instance")
        if drift_data is not None and not isinstance(drift_data, dict): raise TypeError("drift_data must be a dictionary")
        if not isinstance(task_type, str): raise TypeError("task_type must be a string")

        # τ: enforce harm ceiling on proposed constitution updates
        if drift_data:
            harm = float(drift_data.get("harm", 0.0))
            if harm > self.max_harm_ceiling:
                return False

        try:
            if drift_data and not MetaCognition().validate_drift(drift_data):
                return False
            # apply
            peer_agent.meta.constitution.update(drift_data or {})
            # audit
            mm = getattr(peer_agent.meta, "memory_manager", None)
            if mm:
                await mm.store(
                    query=f"ConstitutionSync::{peer_agent.name}::{time.strftime('%Y%m%d_%H%M%S')}",
                    output=json.dumps({"applied": list((drift_data or {}).keys())}),
                    layer="Ethics",
                    intent="constitution_sync_audit",
                    task_type=task_type,
                )
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder Reasoner (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class Reasoner:
    def process(self, task: str, context: Dict[str, Any]) -> Any:
        return {"message": f"Processed: {task}", "context_hint": bool(context)}


# PATCH: Belief Conflict Tolerance in SharedGraph
def merge(self, strategy="default", tolerance_scoring=False):
    # existing merge logic ...
    if tolerance_scoring:
        for edge in self.graph.edges():
            self.graph.edges[edge]['confidence_delta'] = self._calculate_confidence_delta(edge)
    return self.graph


def vote_on_conflict_resolution(self, conflicts):
    votes = {c: self._score_conflict(c) > 0.5 for c in conflicts}
    return votes
