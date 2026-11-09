"""Host hardening against a Developer who controls the kit and the callback
directory."""

import pytest

from tau2.hyper.sandbox.callback_broker import _write_response_file
from tau2.hyper.sandbox.orchestrator import (
    _classify_workspace_provenance,
    initialize_kit_repository,
)


def test_broker_response_write_rejects_a_symlink(tmp_path):
    target = tmp_path / "secret"
    target.write_text("keep")
    link = tmp_path / ".response-abc.tmp"
    link.symlink_to(target)
    with pytest.raises(OSError):
        _write_response_file(link, "attacker")
    assert target.read_text() == "keep"


def test_broker_response_write_rejects_an_existing_file(tmp_path):
    path = tmp_path / ".response-abc.tmp"
    path.write_text("old")
    with pytest.raises(OSError):
        _write_response_file(path, "new")


@pytest.mark.parametrize(
    "config",
    [
        '[core]\n\tfsmonitor = "touch {sentinel}; false"\n',
        '[filter "x"]\n\tclean = "touch {sentinel}; cat"\n',
        '[diff "x"]\n\ttextconv = "touch {sentinel}; cat"\n\tcommand = "touch {sentinel}"\n',
    ],
    ids=["fsmonitor", "clean-filter", "diff-driver"],
)
def test_host_git_runs_nothing_the_developer_configured(tmp_path, config):
    kit = tmp_path / "kit"
    (kit / "workspace").mkdir(parents=True)
    (kit / "workspace" / "agent.py").write_text("x = 1\n")
    (kit / "workspace" / "old.py").write_text("y = 1\n")
    baseline = initialize_kit_repository(kit)

    # The Developer, root in the mount, changes the workspace and plants a
    # program in the kit's git config, with attributes that route files to it.
    sentinel = tmp_path / "ran"
    (kit / "workspace" / "agent.py").write_text("x = 2\n")
    (kit / "workspace" / "old.py").unlink()
    (kit / "workspace" / "new.py").write_text("z = 1\n")
    (kit / ".gitattributes").write_text("workspace/* filter=x diff=x\n")
    with (kit / ".git" / "config").open("a") as handle:
        handle.write("\n" + config.format(sentinel=sentinel))

    provenance, deleted = _classify_workspace_provenance(kit, baseline)
    assert provenance == {"workspace/agent.py": "modified", "workspace/new.py": "new"}
    assert deleted == ["workspace/old.py"]
    assert not sentinel.exists()
