import pytest

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
