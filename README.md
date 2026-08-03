# kiwimatecoder

An agentic AI coding assistant for your terminal. Run one command and drop into an
interactive session that can read, edit, and run code in your project — across the
model provider of your choice. Bring your own API key.

## Installation

```bash
pip install -e .
```

For a regular (non-editable) install:

```bash
pip install .
```

## Quick start

Add an API key for at least one provider, then launch the interactive session:

```bash
kiwimatecoder config set-key --provider openrouter <YOUR_KEY>
kiwimatecoder
```

Running `kiwimatecoder` with no arguments opens the interactive REPL. It keeps running
until you exit. Type a request in plain language and KiwiMateCoder will read files,
search the codebase, propose edits, and run commands to carry it out. For larger or
ambiguous tasks, it starts with a short plan and gives you a few concrete options
so you can choose the scope and tradeoffs before it proceeds.

```
kiwi (openrouter:anthropic/claude-sonnet-5 · ask) › add a docstring to main.py
```

- **Ctrl-C** cancels the current turn and returns you to the prompt.
- **Ctrl-D** exits the session.
- Type `/` to open the slash-command menu. It filters as you keep typing.
- Run `/model`, `/provider`, or `/mode` without an argument to open a
  keyboard-driven selector. Arrow keys move, Enter selects, and Ctrl-C returns to
  the prompt without changing anything.
- Opening `/model` checks the provider for models released since you last looked,
  and drops any it has retired. `/model refresh` forces the check.
- Use `/mode plan` when you only want investigation and options, with no edits or
  shell commands.

## Permission modes

The agent can edit files and run shell commands. A permission mode controls how much
it can do without asking, and you can switch it at any time with `/mode`:

| Mode | Behavior |
|------|----------|
| `ask` (default) | Reads run freely; every file write/edit or shell command shows a preview/diff and waits for your approval. |
| `auto-accept` | Actions run without prompting. |
| `plan` | Read-only: the agent can inspect and explain, but cannot write or run anything. |

Reads, writes, edits, listings and searches are sandboxed to the workspace root (via symlink-aware path resolution). `run_bash` commands execute with the workspace root as their cwd but are otherwise unrestricted (subject to approval/mode); use them for git, tests, builds, etc.

## Slash commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands. |
| `/exit`, `/quit` | Leave the session. |
| `/clear` | Clear the conversation history. |
| `/model [name\|refresh\|list]` | Interactively choose a model (the list is refreshed from the provider), set one by name, or refresh/show the list. |
| `/provider [id]` | Interactively choose a provider, or switch by id. |
| `/mode [ask\|auto-accept\|plan]` | Interactively choose, or directly set, the permission mode. |
| `/tools` | List available tools. |
| `/files` | List files changed this session. |
| `/context [list\|add\|remove\|clear]` | Pin files to include as context on every turn. |
| `/config` | Show or change providers, API keys, model defaults, and model filters. |
| `/cost` | Show token usage for this session. |

Examples:

```text
/context add README.md kiwimatecoder/*.py
/context
/context remove README.md
/context clear
```

Config examples:

```text
/config
/config provider add local "Local Models" http://localhost:1234/v1 local-code LOCAL_API_KEY
/config key set local <YOUR_KEY>
/config provider use local
/config models allow local-code local-fast
/config models deny noisy-model
/config models refresh
/config provider remove local
```

## Tools

The assistant has these capabilities, all scoped to the workspace:

- `read_file`, `list_dir`, `search` (grep + glob) — read-only, always allowed.
- `write_file`, `edit_file` — create/modify files (approval-gated).
- `run_bash` — run shell commands (approval-gated).

## Providers

KiwiMateCoder ships a built-in registry of providers. Switch live with `/provider`,
or set persistent defaults with `/config provider use`, `config set-provider`, and
`config set-model`. A key can be supplied via `/config key set <provider> <key>`,
`config set-key --provider <id>`, or the provider's environment variable. You can
also add OpenAI-compatible custom providers with `/config provider add`.

| Provider id | Default model | Key env var |
|-------------|---------------|-------------|
| `openai` | `gpt-5.6-sol` | `OPENAI_API_KEY` |
| `anthropic` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| `google` | `gemini-3.5-flash` | `GEMINI_API_KEY` |
| `xai` | `grok-4.5` | `XAI_API_KEY` |
| `mistral` | `mistral-medium-3.5` | `MISTRAL_API_KEY` |
| `deepseek` | `deepseek-v4-pro` | `DEEPSEEK_API_KEY` |
| `qwen` | `qwen3.7-max` | `DASHSCOPE_API_KEY` |
| `moonshot` | `kimi-k2.7-code` | `MOONSHOT_API_KEY` |
| `openrouter` | `anthropic/claude-sonnet-5` | `OPENROUTER_API_KEY` |

These defaults are a starting point; the live catalog below is what `/model`
actually offers once a provider is in use.

Note: The `anthropic` provider id is kept for key configuration and future expansion.
Its current base URL points at Anthropic's native API; the client assumes
OpenAI-compatible chat+tools streaming for all providers today. For Anthropic
models, routing via `openrouter` (or another gateway) is the most reliable path
until native support is added. (Model listing already speaks Anthropic's native
`/v1/models` scheme, so `/model` works there today.)

## Model catalogs

Model ids drift constantly, so the list `/model` offers is fetched from the
provider itself rather than being frozen into the release.

- **Newest first.** Opening `/model` (or running `/model refresh`) calls the
  provider's `/models` endpoint and offers what it serves today, ordered by
  release date, with the provider default pinned first.
- **Deprecated ids disappear.** Anything the provider no longer lists is dropped
  from the catalog, and `/model refresh` prints what was removed. If your current
  model is one of them, it says so.
- **Only usable models.** Embedding, image, audio, moderation, and non
  tool-calling models are filtered out — the agent loop needs text chat plus tool
  calls. The newest 60 are offered; any other id still works if you type it.
- **Cached for a day.** Results are stored in `~/.kiwimatecoder/model_cache.json`
  and reused for 24 hours, so the selector stays instant and tab completion never
  touches the network. A failed fetch falls back to the cached list, then to the
  built-in one, and is not retried automatically for 30 minutes.
- **Needs a key.** A provider is only queried once it has an API key configured
  (local providers on `localhost` are queried without one).

From the shell, `kiwimatecoder config models [--provider <id>] [--refresh]` shows
or refreshes the same catalog.

The curated tuples in `kiwimatecoder/providers.py` remain the offline fallback,
and custom providers can list theirs via a `"models"` array in
`~/.kiwimatecoder/config.json`. `/config models allow|deny` reshapes what is
offered on top of whichever catalog is in use, and `/model <name>` or
`config set-model <name>` accepts any id, listed or not.

## One-shot mode

For a quick question without entering the REPL:

```bash
kiwimatecoder ask "how do I reverse a list in python?"
kiwimatecoder ask "review this file" --file app.py --provider openai
```

## Update

Update the CLI from the same Python environment:

```bash
kiwimatecoder -update
```

You can also run `kiwimatecoder update`. Check the installed version before/after
with:

```bash
kiwimatecoder --version
```

When KiwiMateCoder is running from a Git checkout, the updater first fetches
`origin` and compares the local `HEAD` to `origin/<branch>`:

- **Already up to date** — prints `Already on the latest version (commit <sha>).`
  and exits without pulling or reinstalling.
- **Behind** — prints `Updating from <old-sha> (N commit(s) behind origin/<branch>)…`,
  runs `git pull --ff-only`, reinstalls the checkout with
  `pip install --upgrade -e <path>`, and reports `Updated <old-sha> → <new-sha>.`

KiwiMateCoder is not published to PyPI, so for packaged (non-Git) installs the
fallback runs
`pip install --upgrade --force-reinstall git+https://github.com/Kyle8933/kiwimatecoder.git`.


## Configuration

Settings live in `~/.kiwimatecoder/config.json` (provider keys, default
provider/model, default mode). The original single-key `~/.kiwimatecoder/config`
format is read automatically, so existing setups keep working.

Fetched model catalogs are cached separately in
`~/.kiwimatecoder/model_cache.json`. It holds no secrets and can be deleted at
any time; the next `/model` rebuilds it.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

- Media generation (images/video) — a clean extension point exists in
  `kiwimatecoder/media.py` but is not yet implemented.
