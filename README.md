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
kiwimatecoder setup
kiwimatecoder
```

`kiwimatecoder setup` walks you through choosing a provider and saving its API
key. You can also pass the arguments directly:

```bash
kiwimatecoder setup --provider openrouter
kiwimatecoder setup --provider openai --key sk-...   # non-interactive
```

Running `kiwimatecoder` with no arguments opens the interactive REPL. It keeps
running until you exit. Type a request in plain language and KiwiMateCoder will
read files, search the codebase, propose edits, and run commands to carry it
out. For larger or ambiguous tasks, it starts with a short plan and gives you a
few concrete options so you can choose the scope and tradeoffs before it
proceeds.

```
kiwi (openrouter:anthropic/claude-sonnet-5 · ask) › add a docstring to main.py
```

- **Ctrl-C** cancels the current turn and returns you to the prompt; whatever
  the model had already streamed is kept in the conversation so you can continue
  from it.
- **Ctrl-D** exits the session.
- While the agent is working you can type ahead: press Enter to send a steering
  message and the model picks it up on its next step, or type a slash command
  (e.g. `/undo`) and it runs as soon as the turn finishes. Ctrl-C during the
  turn cancels it.
- Your prompt history is saved; the up-arrow recalls commands from previous
  sessions.
- Every session is auto-saved when you exit. Pick it back up with
  `kiwimatecoder --continue` (same as `--resume last`), or from inside the REPL
  with `/load last`.
- Type `/` to open the slash-command menu. It filters as you keep typing.
- Run `/model`, `/provider`, or `/mode` without an argument to open a
  keyboard-driven selector. `/provider` is a checklist: check every provider you
  want on the failover roster. The first checked provider is the primary; the
  rest are tried in order if the primary fails. `/provider <id>` replaces the
  roster with that single provider. Arrow keys move, Enter selects, and Ctrl-C
  returns to the prompt without changing anything.
- Opening `/model` checks the provider for models released since you last looked,
  and drops any it has retired. `/model refresh` forces the check. The model you
  pick is saved as the default for the next session (`/config model reset`
  restores the provider default). Switching the primary provider with
  `/provider` clears that saved model so a vendor-specific id cannot leak.
- `/model search <term>` searches the provider's full catalog by name and opens
  the selector on the matches — useful when a model is older than the newest few
  that `/model` lists.
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

At an approval prompt, answer `a` (always) to approve that tool for the rest of
the session **and future sessions**. Persisted approvals survive restarts and
provider switches; manage them with `/config permissions` or
`config permissions list|remove|clear`. When a diff spans multiple hunks, answer
`h` to choose which hunks to apply (e.g. `1,3`, `1-2`, `all`, or `none`).

Every mutating action is checkpointed before it runs. Restore the last change
with `/undo`, rewind several steps with `/undo 3`, and list them with
`/checkpoints`.

Reads, writes, edits, listings and searches are sandboxed to the workspace root (via symlink-aware path resolution). `run_bash` commands execute with the workspace root as their cwd but are otherwise unrestricted (subject to approval/mode); use them for git, tests, builds, etc.

### Command rules

Auto-approve trusted commands and hard-block dangerous ones with regex rules
(deny rules block even in `auto-accept` mode):

```text
/config commands allow ^pytest\b
/config commands deny rm\s+-rf
/config commands list
```

### Dry run and trusted workspace

`/dry-run` previews every mutating tool call (diffs and shell commands) without
executing anything or prompting. `/config trust on` lets read-only tools read
paths outside the workspace root; writes stay sandboxed.

## Slash commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands. |
| `/exit`, `/quit` | Leave the session. |
| `/clear` | Clear the conversation history. |
| `/model [name\|refresh\|list\|search <term>]` | Interactively choose a model (the list is refreshed from the provider), set one by name, refresh/show the list, or search the full catalog by name. The choice is remembered for the next session. |
| `/provider [id]` | Choose a failover roster (checklist), or replace it with one provider by id. |
| `/mode [ask\|auto-accept\|plan]` | Interactively choose, or directly set, the permission mode. |
| `/dry-run [on\|off\|toggle]` | Preview mutating actions without running them. |
| `/undo [count]`, `/checkpoints` | Restore files changed by recent tools, or list checkpoints. |
| `/todos` | Show the agent's task list. |
| `/templates` | List custom prompt templates discovered in the workspace and user command directories. |
| `/compact [budget]` | Trim older history to fit a token budget. |
| `/save`, `/load`, `/sessions`, `/fork [name]`, `/export [path]` | Save, resume, branch, or export sessions as Markdown. |
| `/tools` | List available tools. |
| `/files` | List files changed this session. |
| `/context [list\|add\|remove\|clear]` | Pin files to include as context on every turn. |
| `/config` | Show or change providers, keys, models, filters, permissions, command rules, sampling, styles, verify, and budgets. |
| `/cost` | Show token usage, context gauge, and estimated USD cost for this session (per-model pricing). |
| `/doctor` | Run environment, config, provider, and workspace diagnostics. |

Examples:

```text
/context add README.md kiwimatecoder/*.py
/undo 2
/export review.md
/fork experiment
```

Config examples:

```text
/config
/config provider add local "Local Models" http://localhost:1234/v1 local-code LOCAL_API_KEY
/config provider edit local name="Local Models 2" default_model=local-fast
/config key set local <YOUR_KEY>
/config provider use local
/config models allow local-code local-fast
/config models deny noisy-model
/config models refresh
/config mode set plan
/config permissions list
/config permissions remove run_bash
/config commands allow ^pytest\b
/config commands deny rm\s+-rf
/config sampling set temperature=0.2 max_tokens=4096
/config sampling reset
/config style set concise
/config prompt set "Prefer functional style; never use classes."
/config prompt clear
/config verify set "pytest -q"
/config budget tokens 500000
/config budget cost 5.00
/config trust on
/config provider remove local
```

Running bare `/config` opens an interactive menu: pick **Keys**, choose a
provider, then set (type a new key) or remove it. Every entry also works as a
typed command as shown above, and `/config key set` / `/config key list` always
show which file or environment variable the active key comes from.

## Tools

The assistant has these capabilities, all scoped to the workspace:

- `read_file`, `list_dir`, `search` (grep + glob) — read-only, always allowed;
  batches of read-only calls run in parallel.
- `write_file`, `edit_file` — create/modify files (approval-gated; each is
  checkpointed first so `/undo` can restore it).
- `run_bash` — run shell commands (approval-gated, subject to command rules).
- `update_todos` — keep the visible task list in sync with multi-step work.
- `ask_user` — ask a clarifying question with optional choices (interactive
  sessions only).

## Providers

KiwiMateCoder ships a built-in registry of providers. Switch live with `/provider`,
or set persistent defaults from the shell with `config provider use <id>` and
`config model set <id>`. An API key can be saved interactively with
`kiwimatecoder setup`, directly with `config key set <provider> <key>`, or via the
provider's environment variable. You can also add OpenAI-compatible custom
providers with `config provider add`.

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
| `ollama` | *(from server)* | `OLLAMA_API_KEY` (optional) |
| `lmstudio` | *(from server)* | `LMSTUDIO_API_KEY` (optional) |
| `unsloth` | *(from server)* | `UNSLOTH_API_KEY` (required) |

These defaults are a starting point; the live catalog below is what `/model`
actually offers once a provider is in use.

The `anthropic` provider talks to Anthropic's native Messages API, including
streaming tool calls and native model listing. Custom providers can opt into the
same path with `config provider add ... --compat anthropic` (or
`/config provider edit <id> compat=anthropic`); everything else speaks the
OpenAI-compatible chat+tools protocol.

### Local providers (Ollama, LM Studio & Unsloth)

The `ollama`, `lmstudio`, and `unsloth` providers talk to model servers on your
own machine. Make sure the server is running ([Ollama](https://ollama.com) on
`http://localhost:11434`, [LM Studio](https://lmstudio.ai)'s local server on
`http://localhost:1234`, or [Unsloth](https://unsloth.ai)'s Studio API on
`http://localhost:8888`), then:

```text
/provider ollama
```

That's it for Ollama and LM Studio — no API key, no setup. The session model is
resolved live from whatever the server has loaded or pulled, and `/model` lists
the server's actual models. `kiwimatecoder setup` detects which local servers
are running and marks them in the picker. Tool calling depends on the model you
pick, so choose a tool-capable family (Llama 3.1+, Qwen 3, DeepSeek, ...).

`unsloth` works the same way but **requires a key**: Unsloth enforces auth even
on localhost. Create one in Unsloth under Settings → API (it starts with
`sk-unsloth-`), run `kiwimatecoder setup --provider unsloth` to save it, and
load a GGUF model in Unsloth before chatting. Tip: start Unsloth's server with
`--disable-tools` when an external agent drives it — otherwise Unsloth's own
server-side tools swallow the agent's tool calls.

A key only matters if your server enforces auth (e.g. LM Studio's server token,
or Ollama behind an authenticating proxy) — set `OLLAMA_API_KEY` /
`LMSTUDIO_API_KEY` or `config key set ollama <key>` in that case. Running on a
different port or another machine? Add it as a custom provider instead:
`/config provider add mybox "My Box" http://mybox.local:11434/v1 any-model` —
any `localhost`/`*.local` provider is treated as keyless automatically.

## Model catalogs

Model ids drift constantly, so the list `/model` offers is fetched from the
provider itself rather than being frozen into the release.

- **Newest first.** Opening `/model` (or running `/model refresh`) calls the
  provider's `/models` endpoint and offers what it serves today, ordered by
  release date, with the provider default pinned first. Search
  (`/model search <term>`) bypasses this newest-first cap and scans the whole
  catalog by name.
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
- **Needs a key.** A cloud provider is only queried once it has an API key
  configured; keyless local servers (`ollama`, `lmstudio`, and any custom
  provider on `localhost` or a `*.local` host) are queried without one —
  `unsloth`, which enforces auth locally, is queried once its key is set.

From the shell, `kiwimatecoder config models show [--provider <id>]` and
`kiwimatecoder config models refresh [--provider <id>]` show or refresh the same
catalog.

The curated tuples in `kiwimatecoder/providers.py` remain the offline fallback,
and custom providers can list theirs via a `"models"` array in
`~/.kiwimatecoder/config.json`. `/config models allow|deny` reshapes what is
offered on top of whichever catalog is in use, and `/model <name>` or
`config model set <name>` accepts any id, listed or not.

## One-shot mode

For a quick question without entering the REPL:

```bash
kiwimatecoder ask "how do I reverse a list in python?"
kiwimatecoder ask "review this file" --file app.py --provider openai
```

## Update

Update the CLI from the same Python environment:

```bash
kiwimatecoder update
```

or with a flag:

```bash
kiwimatecoder --update
```

(The older `-update` form still works but is deprecated.) Check the installed
version before/after with:

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


## Project instructions

Drop an `AGENTS.md` (or `CLAUDE.md`, `.kiwimatecoder/AGENTS.md`,
`.kiwimatecoder/instructions.md`) at the workspace root and its contents are
loaded into the system prompt on every turn — build commands, style rules,
testing conventions. Files are capped at 32KB each and 48KB total.

## Custom commands and prompt templates

Drop a Markdown file in `<workspace>/.kiwimatecoder/commands/` (or
`~/.kiwimatecoder/commands/`) and its filename becomes a slash command. The
body is the prompt sent as a normal user turn; `$ARGUMENTS` is replaced with
whatever you type after the command, and the arguments are appended when the
token is absent. Workspace templates win over user templates with the same
name, and built-in commands always win over templates. `/templates` lists what
was discovered; `/help` adds a Custom commands group.

```text
# .kiwimatecoder/commands/review.md
# Review the current diff
Review $ARGUMENTS for bugs, security issues, and missing tests.

/review src/app.py
```

## Agent Skills

Skills are Markdown instruction bundles the model loads only when a task
matches, so they cost no tokens on turns that do not use them. Each skill is a
directory containing `SKILL.md`:

```text
<workspace>/.kiwimatecoder/skills/pdf/SKILL.md   # project skill
~/.kiwimatecoder/skills/pdf/SKILL.md             # personal skill
```

The system prompt lists each skill's name and first line only; the model calls
the read-only `load_skill` tool to pull the full body (capped at 32KB).
Workspace skills shadow user skills with the same name.

## Plugins

Plugins are Python files with a `register(api)` entry point that can add tools,
slash commands, and event subscribers without forking the CLI:

```python
from kiwimatecoder.tools.base import FunctionTool, ToolResult

def register(api):
    api.register_tool(FunctionTool(
        name="word_count",
        description="Count words in a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        func=lambda args, session: ToolResult(content="..."),
    ))
    api.register_command("hello", lambda arg, session, console: "continue")
    api.subscribe("post_tool", lambda event: None)
```

User plugins load from `~/.kiwimatecoder/plugins/*.py`. Project plugins in
`<workspace>/.kiwimatecoder/plugins/*.py` are **disabled by default** — cloning
a repository must never execute code just because the CLI started in it. Opt in
with `"plugins": {"allow_project": true}` in `~/.kiwimatecoder/config.json`, and
skip individual plugins with `"plugins": {"disabled": ["name"]}`. A plugin that
fails to import or register is reported as a dim warning and skipped; it never
prevents startup.

## Sampling and output styles

Sampling parameters apply to every provider and are sent when set:

```bash
kiwimatecoder config sampling set temperature=0.2 top_p=0.9 max_tokens=4096 reasoning_effort=medium
kiwimatecoder config sampling show
kiwimatecoder config sampling reset
```

Output styles shape the agent's replies: `default`, `concise`, `explanatory`,
or `code`. Set one with `config style set <name>` (or `/config style set`). For
a fully custom instruction, add a system-prompt suffix with
`config prompt set "<text>"` and remove it with `config prompt clear`.

## Auto-verify

Point the agent at your test/lint command and it runs automatically after any
successful file edit, feeding the output back so it can fix failures:

```bash
kiwimatecoder config verify set "pytest -q"
kiwimatecoder config verify show
kiwimatecoder config verify clear
```

## Budgets

Cap a session's spend by tokens, cost, or both. The agent warns at 80% and
stops starting new turns once a limit is reached:

```bash
kiwimatecoder config budget tokens 500000
kiwimatecoder config budget cost 5.00
kiwimatecoder config budget show
kiwimatecoder config budget clear
```

## Audit log

Every tool decision (allowed, denied, dry-run, auto-verify) is appended to
`~/.kiwimatecoder/audit.log` as one JSON line per action. Secrets are redacted
before writing; the file is owner-only and safe to delete.

A `pre_tool` hook that exits non-zero blocks the action and is recorded with
the `hook_blocked` decision.

## Lifecycle hooks

Run your own shell commands when the agent reaches a lifecycle event. Hooks
live under the `hooks` key in `~/.kiwimatecoder/config.json` (a project
`.kiwimatecoder.json` may add them too). Each event maps to a list of commands
run with the session workspace as their working directory:

```json
{
  "hooks": {
    "session_start": ["echo session started in $KIWI_WORKSPACE"],
    "session_end": ["echo session ended"],
    "pre_tool": ["test \"$KIWI_TOOL_NAME\" != \"run_bash\" || ./scripts/safety-check.sh"],
    "post_tool": ["echo \"$KIWI_TOOL_NAME\" exited ok=$KIWI_TOOL_OK\" >> .kiwi-hooks.log"]
  }
}
```

Events are `session_start`, `session_end`, `pre_tool`, and `post_tool`. Hooks
receive `KIWI_EVENT` and `KIWI_WORKSPACE`, plus `KIWI_TOOL_NAME`,
`KIWI_TOOL_OK` (`true`/`false`), `KIWI_TOOL_ARGS` (JSON; secrets redacted), and
`KIWI_TOOL_DURATION_MS` for tool events. A non-zero exit from a `pre_tool` hook
blocks the tool call; every hook times out after 60 seconds and never crashes
the agent. `/config` does not manage hooks yet — edit the JSON directly.

## Configuration

Global settings live in `~/.kiwimatecoder/config.json` (provider keys, default
provider/model, default mode, sampling, output style, custom prompt, and
persisted tool approvals). The original single-key `~/.kiwimatecoder/config`
format is read automatically, so existing setups keep working.

A project can pin its own settings in `.kiwimatecoder.json` at the repository
root (or wherever `KIWIMATECODER_PROJECT_CONFIG` points). Project values
override global config — useful for pinning a provider, model, mode, sampling,
or permissions per repository. API keys are never read from project files.

Fetched model catalogs are cached separately in
`~/.kiwimatecoder/model_cache.json`. It holds no secrets and can be deleted at
any time; the next `/model` rebuilds it.

Prompt history lives in `~/.kiwimatecoder/history` and session autosaves in
`~/.kiwimatecoder/sessions/last.json`. Both can be deleted at any time.

Run `kiwimatecoder doctor` (or `/doctor`) to check config paths, key status,
provider reachability, the model catalog, and the workspace.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check kiwimatecoder tests
mypy kiwimatecoder
```

## Roadmap

The full prioritized plan lives in [ROADMAP.md](ROADMAP.md). P0 and P1 are
implemented (checkpoints/undo, command rules, dry-run, redacted audit log,
hunk-level approvals, todos, ask-user, parallel reads, compaction, auto-verify,
budgets, trusted workspace, and message steering with interrupt recovery), and
P2 has begun (event bus + hooks, the extensible tool registry, custom commands
and prompt templates, on-demand Agent Skills, and a plugin system). P2 continues
with MCP.
