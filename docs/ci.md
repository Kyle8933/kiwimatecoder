# CI and automation

KiwiMateCoder's headless `-p/--print` mode is designed to run in CI. The process
never prompts: anything that would need approval is denied unless `--yes` or
`--mode auto-accept` is passed. Assistant output goes to stdout, tool/progress
lines to stderr, so `--output-format json` keeps stdout machine-readable.

## GitHub Actions: review a pull request

Copy this workflow into `.github/workflows/review.yml` and store your provider
key as a repository secret (for example `OPENROUTER_API_KEY`):

```yaml
name: AI review

on:
  pull_request:

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install
        run: pip install kiwimatecoder
      - name: Review the staged diff
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          git diff "origin/${{ github.base_ref }}...HEAD" > review.diff
          kiwimatecoder -p "Review the staged diff for bugs" \
            --output-format json --max-turns 10 > review.json
          cat review.json
```

Swap `OPENROUTER_API_KEY` for the environment variable of whichever provider you
use (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, ...). `pip
install "kiwimatecoder[browser]"` adds the optional Playwright browser tool.

The same step works with any shell: pipe a diff or file through stdin with
`kiwimatecoder -p -`.

## Exit codes

`-p/--print` returns a precise exit code so a CI step can gate on it:

| Code | Meaning |
| --- | --- |
| `0` | Success: the run finished and produced a final answer. |
| `1` | Runtime failure: missing/invalid API key, provider error, or the tool loop hit `--max-turns` without an answer. |
| `2` | Invalid usage: unknown `--output-format`/`--mode`/`--provider`, empty prompt, or a missing workspace. |

A failed *review* is still exit `0` — the model answered. Use the JSON
`success` field or the assistant text for policy decisions, and reserve the
exit code for whether the agent itself ran.

## Pre-commit

The repository ships a `.pre-commit-config.yaml` with a `ruff check` hook on
commit and a local `pytest -q` hook on push:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

Run every hook manually with `pre-commit run --all-files` (add
`--hook-stage pre-push` for the test hook).

## Repository CI

`.github/workflows/ci.yml` runs the full suite on Ubuntu and macOS across Python
3.10–3.12, plus `ruff check kiwimatecoder tests` and `mypy kiwimatecoder`.
