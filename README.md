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
search the codebase, propose edits, and run commands to carry it out.

```
kiwi (openrouter:mistralai/devstral-2512 · ask) › add a docstring to main.py
```

- **Ctrl-C** cancels the current turn and returns you to the prompt.
- **Ctrl-D** exits the session.

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
| `/model [name]` | Show or set the model. |
| `/provider [id]` | List providers or switch the active one. |
| `/mode [ask\|auto-accept\|plan]` | Show or set the permission mode. |
| `/tools` | List available tools. |
| `/files` | List files changed this session. |
| `/cost` | Show token usage for this session. |

## Tools

The assistant has these capabilities, all scoped to the workspace:

- `read_file`, `list_dir`, `search` (grep + glob) — read-only, always allowed.
- `write_file`, `edit_file` — create/modify files (approval-gated).
- `run_bash` — run shell commands (approval-gated).

## Providers

KiwiMateCoder ships a built-in registry of providers. Switch live with `/provider`,
or set defaults with `config set-provider` / `config set-model`. A key can be supplied
either via `config set-key --provider <id>` or the provider's environment variable.

| Provider id | Default model | Key env var |
|-------------|---------------|-------------|
| `openai` | `gpt-5.5` | `OPENAI_API_KEY` |
| `anthropic` | `claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| `google` | `gemini-3.5-flash` | `GEMINI_API_KEY` |
| `xai` | `grok-build-0.1` | `XAI_API_KEY` |
| `mistral` | `mistral-medium-3.5` | `MISTRAL_API_KEY` |
| `deepseek` | `deepseek-v4-pro` | `DEEPSEEK_API_KEY` |
| `qwen` | `qwen3.7-max` | `DASHSCOPE_API_KEY` |
| `moonshot` | `kimi-k2.7-code` | `MOONSHOT_API_KEY` |
| `openrouter` | `mistralai/devstral-2512` | `OPENROUTER_API_KEY` |

Default model ids reflect what was current at the time of writing; they drift, so
override them with `/model <name>` or `config set-model <name>` as providers update.

Note: The `anthropic` provider id is kept for key configuration and future expansion.
Its current base URL points at Anthropic's native API; the client assumes
OpenAI-compatible chat+tools streaming for all providers today. For Anthropic
models, routing via `openrouter` (or another gateway) is the most reliable path
until native support is added.

## One-shot mode

For a quick question without entering the REPL:

```bash
kiwimatecoder ask "how do I reverse a list in python?"
kiwimatecoder ask "review this file" --file app.py --provider openai
```

## Configuration

Settings live in `~/.kiwimatecoder/config.json` (provider keys, default
provider/model, default mode). The original single-key `~/.kiwimatecoder/config`
format is read automatically, so existing setups keep working.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Roadmap

- Media generation (images/video) — a clean extension point exists in
  `kiwimatecoder/media.py` but is not yet implemented.
