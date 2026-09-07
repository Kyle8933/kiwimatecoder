from __future__ import annotations

from typing import Any

import pytest
from rich.console import Console

from kiwimatecoder.permissions import PermissionMode
from kiwimatecoder.session import Session


@pytest.fixture
def session(tmp_path):
    """A session rooted at a temp workspace."""
    return Session(
        provider_id="openrouter",
        model="test-model",
        mode=PermissionMode.ASK,
        workspace_root=tmp_path,
    )


def track_console(console: Console) -> list[tuple[str, str]]:
    """Record status start/stop and non-empty prints on a real Console."""
    log: list[tuple[str, str]] = []
    real_status = console.status
    real_print = console.print

    def status(message: object, **kwargs: Any) -> Any:
        handle = real_status(message, **kwargs)
        text = str(message)
        orig_start = handle.start
        orig_stop = handle.stop

        def start() -> None:
            log.append(("start", text))
            orig_start()

        def stop() -> None:
            log.append(("stop", text))
            orig_stop()

        handle.start = start
        handle.stop = stop
        return handle

    def printer(*args: object, **kwargs: Any) -> Any:
        if args:
            rendered = str(args[0])
            if rendered:
                log.append(("print", rendered))
        return real_print(*args, **kwargs)

    console.status = status  # type: ignore[method-assign]
    console.print = printer  # type: ignore[method-assign]
    return log
