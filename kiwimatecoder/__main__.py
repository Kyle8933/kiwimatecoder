"""Run the CLI with ``python -m kiwimatecoder``.

Background jobs spawn headless agent runs this way so they do not depend on the
``kiwimatecoder`` console script being on PATH.
"""

from __future__ import annotations

from kiwimatecoder.main import app

if __name__ == "__main__":
    app()
