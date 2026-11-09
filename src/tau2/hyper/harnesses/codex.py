"""Codex app-server integration for the Hyper-τ construction sandbox."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tau2.hyper.sandbox.builder import BuildStep
from tau2.hyper.sandbox.native_builder import NativeSandboxBuilder
from tau2.hyper.sandbox.native_runtime import NativeProcessEvent

# A run on an image with another Codex release names it here so the
# build metadata records the version that actually ran.
CODEX_HARNESS_VERSION = os.environ.get("TAU2_CODEX_HARNESS_VERSION", "0.144.6")


class CodexSandboxBuilder(NativeSandboxBuilder):
    """Run the pinned real Codex app-server inside the outer Docker boundary."""

    harness_name = "codex"
    harness_version = CODEX_HARNESS_VERSION
    # TAU2_CODEX_GATEWAY_PROVIDER names a model_routing.toml provider
    # for the Developer seat only (e.g. the ChatGPT subscription backend), so
    # providers.openai can keep serving the inner loop.
    model_gateway_provider = os.environ.get("TAU2_CODEX_GATEWAY_PROVIDER", "openai")
    runtime_config_path = "/runtime-home/.codex/config.toml"
    # TAU2_CODEX_MODEL_CATALOG names a host JSON model catalog that
    # replaces Codex's bundled one. The app-server drops a turn's reasoning
    # effort when its catalog does not list the model, so a Developer model
    # newer than this Codex release would otherwise run at the default effort.
    model_catalog_host_path = os.environ.get("TAU2_CODEX_MODEL_CATALOG")
    model_catalog_path = "/runtime-home/.codex/model_catalog.json"

    def harness_config_metadata(self) -> dict:
        return {
            "interface": "app-server-stdio",
            "approval_policy": "never",
            "inner_sandbox": "danger-full-access",
            "web_search": "disabled",
            "apps": False,
            "multi_agent": False,
            "memory": False,
            "history_persistence": "none",
            "model_gateway": "provider-only/per-run/model-scoped",
            "gateway_token_inherited_by_shell": False,
            "mcp_servers": ["hyper_tau"],
        }

    def runtime_environment(self, broker) -> dict[str, str]:
        environment = super().runtime_environment(broker)
        environment["TAU2_DEVELOPER_MODEL"] = self.llm
        reasoning_effort = self.llm_args.get("reasoning_effort")
        if reasoning_effort and reasoning_effort != "none":
            environment["TAU2_DEVELOPER_REASONING_EFFORT"] = str(reasoning_effort)
        return environment

    def render_runtime_config(
        self,
        *,
        include_client_tool: bool,
        include_live_experiment_tool: bool = False,
        include_sample_scenarios_tool: bool = False,
    ) -> str:
        enabled_tools = ["run_local_test", "submit"]
        if include_sample_scenarios_tool:
            enabled_tools.insert(1, "run_sample_scenarios")
        if include_live_experiment_tool:
            enabled_tools.insert(1, "run_live_experiment")
        if include_client_tool:
            enabled_tools.insert(1, "talk_to_client")
        tools = ", ".join(json.dumps(tool) for tool in enabled_tools)
        reasoning_effort = self.llm_args.get("reasoning_effort")
        reasoning_line = (
            f"model_reasoning_effort = {json.dumps(str(reasoning_effort))}\n"
            if reasoning_effort and reasoning_effort != "none"
            else ""
        )
        catalog_line = (
            f"model_catalog_json = {json.dumps(self.model_catalog_path)}\n"
            if self.model_catalog_host_path
            else ""
        )
        return (
            f"model = {json.dumps(self.llm)}\n"
            'model_provider = "tau2_gateway"\n'
            f"{reasoning_line}"
            f"{catalog_line}"
            'approval_policy = "never"\n'
            'sandbox_mode = "danger-full-access"\n'
            "allow_login_shell = false\n"
            'web_search = "disabled"\n'
            'file_opener = "none"\n'
            "\n[history]\n"
            'persistence = "none"\n'
            "\n[memories]\n"
            "generate_memories = false\n"
            "use_memories = false\n"
            "\n[shell_environment_policy]\n"
            'inherit = "all"\n'
            "ignore_default_excludes = false\n"
            'exclude = ["TAU2_MODEL_GATEWAY_TOKEN"]\n'
            "\n[model_providers.tau2_gateway]\n"
            'name = "Hyper-tau model gateway"\n'
            f'base_url = "http://tau2-model-gateway:8143/{self.model_gateway_provider}/v1"\n'
            'env_key = "TAU2_MODEL_GATEWAY_TOKEN"\n'
            'wire_api = "responses"\n'
            # Ride out provider token-rate-limit windows (Codex
            # backs off 0.2 s x 2^n; 12 retries wait up to ~14 min in total)
            # instead of failing the turn after the default 5 quick retries.
            "request_max_retries = 12\n"
            "stream_max_retries = 12\n"
            "supports_websockets = false\n"
            "supports_standalone_web_search = false\n"
            "\n[features]\n"
            "apps = false\n"
            "remote_plugin = false\n"
            "multi_agent = false\n"
            "goals = false\n"
            "hooks = false\n"
            "memories = false\n"
            "network_proxy = false\n"
            "skill_mcp_dependency_install = false\n"
            "shell_tool = true\n"
            "\n[feedback]\n"
            "enabled = false\n"
            "\n[mcp_servers.hyper_tau]\n"
            'command = "/opt/tau2/.venv/bin/python"\n'
            'args = ["-m", "tau2.hyper.sandbox.callback_mcp"]\n'
            "required = true\n"
            "startup_timeout_sec = 30\n"
            "tool_timeout_sec = 28800\n"
            'default_tools_approval_mode = "auto"\n'
            f"enabled_tools = [{tools}]\n"
            'env_vars = ["TAU2_CALLBACK_DIR", "TAU2_CALLBACK_TOKEN", '
            '"TAU2_CALLBACK_TIMEOUT_SECONDS", "TAU2_CLIENT_TOOL_ENABLED", '
            '"TAU2_LIVE_EXPERIMENT_TOOL_ENABLED", '
            '"TAU2_SAMPLE_SCENARIOS_TOOL_ENABLED", '
            '"PATH", "PYTHONPATH", "TAU2_DATA_DIR"]\n'
        )

    def runtime_files(self, **kwargs) -> dict[str, str]:
        files = super().runtime_files(**kwargs)
        if self.model_catalog_host_path:
            files[self.model_catalog_path] = Path(
                self.model_catalog_host_path
            ).read_text()
        return files

    def harness_command(self) -> list[str]:
        return ["python", "-m", "tau2.hyper.harnesses.codex_driver"]

    def normalize_event(self, event: NativeProcessEvent) -> list[BuildStep]:
        if event.channel != "stdout":
            return []
        try:
            message = json.loads(event.text)
        except json.JSONDecodeError:
            return []
        if message.get("method") != "item/completed":
            return []
        item = message.get("params", {}).get("item", {})
        item_type = item.get("type")
        if item_type == "agentMessage":
            return [BuildStep(step_idx=0, role="assistant", content=item.get("text"))]
        if item_type == "reasoning":
            summary = item.get("summary") or []
            if isinstance(summary, str):
                content = summary
            else:
                content = "\n".join(
                    part if isinstance(part, str) else str(part.get("text", ""))
                    for part in summary
                )
            return [BuildStep(step_idx=0, role="assistant", content=content)]
        if item_type == "commandExecution":
            return [
                BuildStep(
                    step_idx=0,
                    role="tool",
                    tool_calls=[
                        {
                            "id": item.get("id", ""),
                            "name": "shell",
                            "arguments": {
                                "command": item.get("command"),
                                "cwd": item.get("cwd"),
                            },
                        }
                    ],
                    tool_results=[
                        {
                            "id": item.get("id", ""),
                            "name": "shell",
                            "result": item.get("aggregatedOutput", ""),
                            "exit_code": item.get("exitCode"),
                            "status": item.get("status"),
                        }
                    ],
                )
            ]
        if item_type == "fileChange":
            return [
                BuildStep(
                    step_idx=0,
                    role="tool",
                    tool_calls=[
                        {
                            "id": item.get("id", ""),
                            "name": "file_change",
                            "arguments": {"changes": item.get("changes", [])},
                        }
                    ],
                    tool_results=[
                        {
                            "id": item.get("id", ""),
                            "name": "file_change",
                            "result": item.get("status", ""),
                        }
                    ],
                )
            ]
        if item_type == "mcpToolCall":
            tool_name = f"mcp:{item.get('server', '')}/{item.get('tool', '')}"
            return [
                BuildStep(
                    step_idx=0,
                    role="tool",
                    tool_calls=[
                        {
                            "id": item.get("id", ""),
                            "name": tool_name,
                            "arguments": item.get("arguments", {}),
                        }
                    ],
                    tool_results=[
                        {
                            "id": item.get("id", ""),
                            "name": tool_name,
                            "result": item.get("result") or item.get("error") or "",
                            "status": item.get("status"),
                        }
                    ],
                )
            ]
        return []
