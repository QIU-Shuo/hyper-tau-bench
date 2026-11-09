"""τ^τ-Verified settings. On by default; ``TAU2_VERIFIED=0`` turns it off.

τ^τ-Verified is the original benchmark with its confirmed errors fixed: two
checks that fail correct agents and framework documents that misstate the
runtime. The design (``run_local_test``, the Client, the menu and the score
formula) and the held-out test set are unchanged. All of it runs on the host,
so the construction image is the original one.
"""

from __future__ import annotations

import os

from tau2.utils.utils import DATA_DIR

VERIFIED_ENV = "TAU2_VERIFIED"
RUNTIME_NOTES_PATH = (
    DATA_DIR / "tau2" / "hyper" / "framework_reference" / "verified_runtime_notes.md"
)


def verified_enabled() -> bool:
    """Whether this run is τ^τ-Verified rather than the original benchmark.

    On unless ``TAU2_VERIFIED=0``.
    """
    return os.environ.get(VERIFIED_ENV, "1") != "0"


# The original contract says tool results always arrive as a MultiToolMessage.
_TOOL_RESULT_SENTENCE = (
    "or a `MultiToolMessage` (results from tool calls the agent made\n"
    "on the previous turn)"
)
_TOOL_RESULT_REPLACEMENT = (
    "or the results of the tool calls the agent made on the previous\n"
    "turn (see *Tool results* below)"
)


def verified_agent_contract(text: str) -> str:
    """The kit's ``agent_contract.md`` with the runtime facts it omits."""
    if _TOOL_RESULT_SENTENCE not in text:
        raise ValueError("agent_contract.md no longer has the tool-result sentence")
    text = text.replace(_TOOL_RESULT_SENTENCE, _TOOL_RESULT_REPLACEMENT)
    return text.rstrip("\n") + "\n" + RUNTIME_NOTES_PATH.read_text()
