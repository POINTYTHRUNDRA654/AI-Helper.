"""Agent — goal-directed tool orchestration.

The :class:`Agent` is the brain of AI Helper.  Give it any free-text goal
and it will:

1. **Plan** — decide which tools to call, in what order, using what arguments.
2. **Act** — invoke those tools via the :class:`~ai_helper.tools.ToolRegistry`.
3. **Observe** — incorporate each result into its context.
4. **Repeat** — keep going until the goal is achieved or ``max_steps`` is reached.

Two planning modes
------------------
* **Ollama mode** — when Ollama is running locally, the agent uses the
  configured model as its reasoning engine (ReAct-style JSON loop).  This
  gives natural language understanding, nuanced tool selection and
  coherent multi-step plans.

* **Rule-based mode** — when no LLM is available, a lightweight keyword
  router maps common goals to the right tools.  Useful in headless / offline
  environments or as an instant fallback.

Usage
-----
::

    from ai_helper.agent import Agent
    agent = Agent()

    # Simple one-shot
    result = agent.execute("What is my CPU usage right now?")
    print(result.answer)

    # Multi-step
    result = agent.execute("Find all Python files in my home directory, "
                           "then show me the largest one.")
    for step in result.steps:
        print(step)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AgentStep:
    """One reasoning/action step taken by the agent."""
    step_number: int
    thought: str
    tool_name: str
    tool_args: Dict[str, Any]
    result: ToolResult
    elapsed_s: float = 0.0

    def __str__(self) -> str:
        status = "✓" if self.result.success else "✗"
        output_preview = self.result.output[:200].replace("\n", " ")
        return (
            f"Step {self.step_number} [{self.elapsed_s:.1f}s] {status}\n"
            f"  Thought : {self.thought}\n"
            f"  Tool    : {self.tool_name}({self.tool_args})\n"
            f"  Result  : {output_preview}"
        )


@dataclass
class AgentResult:
    """The outcome of :meth:`Agent.execute`."""
    goal: str
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    success: bool = True
    total_elapsed_s: float = 0.0

    def __str__(self) -> str:
        steps_str = "\n".join(str(s) for s in self.steps)
        return (
            f"Goal  : {self.goal}\n"
            f"Steps : {len(self.steps)}\n"
            f"{steps_str}\n"
            f"Answer: {self.answer}"
        )


# ---------------------------------------------------------------------------
# System prompt for Ollama planning
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are AI Helper, an intelligent desktop assistant with access to the \
user's computer files, programs and system information.

{tool_catalogue}

Your job is to accomplish the user's goal step by step.

Rules:
- At each step, decide which ONE tool to call next.
- Reply with ONLY a JSON object, no extra text:
  {{"thought": "<why this tool>", "tool": "<tool_name>", "args": {{"<param>": "<value>"}}}}
- When the goal is fully accomplished, reply with:
  {{"thought": "<summary>", "tool": "finish", "args": {{"answer": "<final answer to user>"}}}}
- If a tool fails, try a different approach.
- Keep thoughts concise (one sentence).
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """Goal-directed agent that uses tools to help the user.

    Parameters
    ----------
    registry:
        :class:`~ai_helper.tools.ToolRegistry` to use.  Defaults to one
        with all built-in tools registered.
    ollama_model:
        Ollama model to use for planning (default ``"llama3"``).
    ollama_url:
        Ollama server URL (default ``"http://localhost:11434"``).
    max_steps:
        Maximum tool-call iterations before giving up (default ``10``).
    """

    _FINISH = "finish"

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        ollama_model: str = "llama3",
        ollama_url: str = "http://localhost:11434",
        max_steps: int = 10,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, goal: str) -> AgentResult:
        """Execute *goal* using available tools.

        Tries Ollama planning first; falls back to rule-based planning if
        Ollama is unavailable or returns unparseable output.
        """
        t0 = time.monotonic()

        # Decide planning mode
        use_ollama = self._ollama_available()
        if use_ollama:
            logger.info("Agent: using Ollama (%s) for planning", self.ollama_model)
        else:
            logger.info("Agent: using rule-based planner (Ollama not available)")

        result = AgentResult(goal=goal, answer="", steps=[])

        if use_ollama:
            self._run_ollama_loop(goal, result)
        else:
            self._run_rule_based(goal, result)

        result.total_elapsed_s = time.monotonic() - t0
        result.success = bool(result.answer)
        return result

    # ------------------------------------------------------------------
    # Ollama planning loop (ReAct)
    # ------------------------------------------------------------------

    def _run_ollama_loop(self, goal: str, result: AgentResult) -> None:
        from .ai_integrations import OllamaClient  # noqa: PLC0415
        client = OllamaClient(base_url=self.ollama_url)

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            tool_catalogue=self.registry.describe_all()
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": goal},
        ]

        for step_num in range(1, self.max_steps + 1):
            t_step = time.monotonic()
            llm_result = client.chat(self.ollama_model, messages, timeout=60.0)

            if llm_result.error:
                result.answer = f"Ollama error: {llm_result.error}"
                break

            raw = llm_result.response.strip()
            action = self._parse_action(raw)
            if action is None:
                # Can't parse → treat whole reply as final answer
                result.answer = raw
                break

            thought = action.get("thought", "")
            tool_name = action.get("tool", "")
            tool_args: Dict[str, Any] = action.get("args", {})

            if tool_name == self._FINISH:
                result.answer = tool_args.get("answer", raw)
                break

            tool_result = self.registry.invoke(tool_name, **tool_args)
            elapsed = time.monotonic() - t_step

            step = AgentStep(
                step_number=step_num,
                thought=thought,
                tool_name=tool_name,
                tool_args=tool_args,
                result=tool_result,
                elapsed_s=elapsed,
            )
            result.steps.append(step)

            # Add tool result to conversation
            messages.append({"role": "assistant", "content": raw})
            observation = tool_result.output if tool_result.success else f"Error: {tool_result.error}"
            messages.append({"role": "user", "content": f"Tool result:\n{observation}"})

        if not result.answer:
            result.answer = (
                result.steps[-1].result.output if result.steps else
                "I could not complete the goal within the step limit."
            )

    # ------------------------------------------------------------------
    # Rule-based fallback planner
    # ------------------------------------------------------------------

    _RULES: List[tuple[str, str, Dict[str, Any]]] = [
        # pattern, tool_name, args_template (may use {goal} substitution)
        # NOTE: more-specific patterns must come before less-specific ones.
        (r"\bread\b.+file|open.+file|contents?\s+of",
         "read_file", {}),
        (r"\bwrite\b.+to\b|save.+file|create.+file",
         "write_file", {}),
        (r"\bsearch\b.+files?|find.+files?|look for.+files?",
         "search_files", {}),
        (r"\blist\b.+(files?|folder|dir)|what.+in.+dir|ls\b",
         "list_directory", {}),
        (r"\brun\b.+program|execute|launch|start\s+\w",
         "run_program", {}),
        (r"\bgpu\b|vram|nvidia|graphics\s+card",
         "gpu_stats", {}),
        # ai apps checked BEFORE generic "running" pattern
        (r"\bai\s+apps?\b|comfyui|lm\s+studio|stable\s+diff",
         "list_ai_apps", {}),
        (r"\bollama.*model|model.*ollama|list.*model",
         "list_ollama_models", {}),
        (r"\bollama\b|local\s+llm|local\s+model|ask.*model",
         "ask_ollama", {}),
        (r"\bcpu\b|\bmemory\b|\bdisk\b|system\s+stat|resource",
         "system_snapshot", {}),
        (r"\bprocesses?\b|running programs?|what.+running",
         "list_programs", {}),
    ]

    def _run_rule_based(self, goal: str, result: AgentResult) -> None:
        goal_lower = goal.lower()
        matched = False

        for pattern, tool_name, _ in self._RULES:
            if re.search(pattern, goal_lower):
                matched = True
                args = self._extract_args(tool_name, goal)
                t0 = time.monotonic()
                tool_result = self.registry.invoke(tool_name, **args)
                step = AgentStep(
                    step_number=1,
                    thought=f"Goal matches pattern for {tool_name!r}",
                    tool_name=tool_name,
                    tool_args=args,
                    result=tool_result,
                    elapsed_s=time.monotonic() - t0,
                )
                result.steps.append(step)
                result.answer = tool_result.output if tool_result.success else str(tool_result)
                break

        if not matched:
            # Generic fallback: show system status
            t0 = time.monotonic()
            tool_result = self.registry.invoke("system_snapshot")
            step = AgentStep(
                step_number=1,
                thought="No specific tool matched; showing system status",
                tool_name="system_snapshot",
                tool_args={},
                result=tool_result,
                elapsed_s=time.monotonic() - t0,
            )
            result.steps.append(step)
            result.answer = (
                f"I'm not sure how to handle that specific request yet, but here is "
                f"the current system status:\n\n{tool_result.output}"
            )

    def _extract_args(self, tool_name: str, goal: str) -> Dict[str, Any]:
        """Best-effort argument extraction from free text for rule-based planner."""
        # Extract quoted strings as primary argument values
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', goal)
        first_quoted = next((a or b for a, b in quoted), "")

        # Extract file paths (heuristic: look for path separators)
        path_match = re.search(r"([A-Za-z]:[\\\/][^\s,]+|\/[^\s,]+|~\/[^\s,]+)", goal)
        path = path_match.group(1) if path_match else ""

        if tool_name == "read_file":
            return {"path": path or first_quoted or goal}
        if tool_name == "write_file":
            return {"path": path or first_quoted or "output.txt", "content": ""}
        if tool_name == "search_files":
            # Extract a pattern — look for *.ext or quoted terms
            pattern_m = re.search(r"\*\.\w+|\w+\.\w+", goal)
            return {"query": pattern_m.group(0) if pattern_m else first_quoted or "*",
                    "root": path or ""}
        if tool_name == "list_directory":
            return {"path": path or first_quoted or ""}
        if tool_name == "run_program":
            cmd_m = re.search(r"run\s+(\S+)|execute\s+(\S+)|launch\s+(\S+)", goal.lower())
            cmd = next((g for g in (cmd_m.groups() if cmd_m else []) if g), first_quoted or "")
            return {"command": cmd}
        if tool_name == "list_programs":
            name_m = re.search(r"named?\s+(\S+)|called\s+(\S+)", goal.lower())
            name = next((g for g in (name_m.groups() if name_m else []) if g), "")
            return {"name_filter": name}
        if tool_name == "ask_ollama":
            return {"prompt": goal, "model": self.ollama_model}
        return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ollama_available(self) -> bool:
        try:
            from .ai_integrations import OllamaClient  # noqa: PLC0415
            return OllamaClient(base_url=self.ollama_url).is_running()
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _parse_action(text: str) -> Optional[Dict[str, Any]]:
        """Extract the first JSON object from *text*, or None."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        # Find first {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
