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
| `/sync [status\|push\|pull]` | Show or sync saved sessions across machines through a shared folder (opt-in). |
| `/tools` | List available tools. |
| `/files` | List files changed this session. |
| `/context [list\|add\|remove\|clear]` | Pin files to include as context on every turn. |
| `/memory [list\|add\|add-user\|clear]` | Show or edit persistent project and user memory. |
| `/index [status\|build\|clear]` | Show, refresh, or delete the local codebase index used by semantic search. |
| `/jobs [list\|show <id>\|cancel <id>\|tick\|run <prompt>]` | List or manage detached background agent jobs; `tick` starts due scheduled runs. |
| `/config` | Show or change providers, keys, models, filters, permissions, command rules, sampling, styles, themes, output modes, accessibility, verify, and budgets. |
| `/mcp [list\|reload]` | List configured MCP servers, their status, and registered tools; `reload` reconnects every server. |
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
/config ui theme ocean
/config ui output compact
/config ui ascii on
/config prompt set "Prefer functional style; never use classes."
/config prompt clear
/config verify set "pytest -q"
/config budget tokens 500000
/config budget cost 5.00
/config trust on
/config remote host devbox
/config remote user kiwi
/config remote workspace /srv/app
/config remote enable on
/config browser enable on
/config browser headless off
/config browser timeout 20000
/config provider remove local
```

Running bare `/config` opens an interactive menu: pick **Keys**, choose a
provider, then set (type a new key) or remove it. Every entry also works as a
typed command as shown above, and `/config key set` / `/config key list` always
show which file or environment variable the active key comes from.

## Tools

The assistant has these capabilities, all scoped to the workspace:

- `read_file`, `list_dir`, `search` (grep + glob + semantic) — read-only,
  always allowed; batches of read-only calls run in parallel.
- `view_image` — attach a workspace image (`.png`, `.jpg`, `.jpeg`, `.gif`,
  `.webp`) so a vision-capable model can see it (see below).
- `write_file`, `edit_file` — create/modify files (approval-gated; each is
  checkpointed first so `/undo` can restore it).
- `run_bash` — run shell commands (approval-gated, subject to command rules).
- `shell` — run a command in a persistent shell that remembers `cd` and exported
  environment state between calls (approval-gated; see below).
- `shell_jobs` — start, list, read, or kill detached background commands
  (approval-gated; see below).
- `update_todos` — keep the visible task list in sync with multi-step work.
- `task` — delegate a focused investigation to a subagent with its own context
  (approval-gated; see below).
- `ask_user` — ask a clarifying question with optional choices (interactive
  sessions only).
- `web_fetch`, `web_search` — read a page or search the web; read-only, and
  local/private addresses are blocked by default (see below).
- `browser` — drive an optional headless Chromium page (screenshot, click,
  type, evaluate) when Playwright is installed and browser automation is
  enabled (approval-gated; see below).
- `remember`, `recall` — persist and read durable facts across sessions (see
  below).
- `git`, `git_write` — inspect status/diff/log and stage, unstage, commit, or
  create a branch (see below).
- `forge`, `forge_write` — list/view or create pull requests/merge requests
  and issues through the `gh`/`glab` CLI (see below).
- `read_clipboard`, `write_clipboard` — read the system clipboard or copy text
  to it (see below).
- `http_get`, `http_request` — exercise an HTTP API and inspect the response
  (see below).
- `lsp_diagnostics`, `lsp_definition`, `lsp_references` — language-server
  diagnostics and navigation, when enabled (see below).

## Persistent shell

`run_bash` starts a fresh process per call, so `cd` and exported environment
variables never survive between commands. The `shell` tool instead drives one
long-lived shell per session (`/bin/sh` on POSIX, `cmd.exe` on Windows) so state
sticks:

```text
shell: cd backend
shell: source .venv/bin/activate
shell: pytest -q        # runs inside backend with the venv active
```

- The shell starts in the session workspace root. Output (stdout + stderr
  combined) is capped at 30 KB with a truncation note, and every call reports
  the exit code.
- A command that exceeds the timeout (default 120 s) is killed and the shell is
  restarted, so a hung command can never wedge the session.
- `shell_jobs` runs commands detached from the shell and captures their output
  under `~/.kiwimatecoder/jobs/shell/`. `action: list` shows tracked jobs,
  `action: output` reads one, and `action: kill` terminates it. At most
  `shell.max_jobs` (default 8) run at once.
- Settings: `/config shell [show|persistent on|off|timeout <s>|max-jobs <n>]`
  in the REPL, or `kiwimatecoder config shell ...` from the shell.
- Exiting the session terminates the persistent shell and every tracked
  background job. `run_bash` is unchanged and still the right tool for
  one-shot commands.

## Command sandboxing

On macOS and Linux you can execute `run_bash` (and the persistent shell and its
background jobs) through an OS-level sandbox. It is **off by default**:

```bash
kiwimatecoder config sandbox show
kiwimatecoder config sandbox enable on
kiwimatecoder config sandbox network off          # block network access
kiwimatecoder config sandbox add-path ~/shared    # extra writable directory
```

The REPL equivalent is `/config sandbox [show|enable on|off|network on|off|
add-path <path>|remove-path <path>|clear-paths]`.

- **macOS** uses seatbelt via `sandbox-exec`: `process*` is allowed, reads are
  broadly allowed, and writes are limited to the workspace, `/tmp`, and
  `extra_writable`.
- **Linux** uses bubblewrap (`bwrap`): `/` is mounted read-only, the workspace
  is read-write, `/tmp` is a private tmpfs, and `--unshare-net` is added when
  the network toggle is off. Install the `bubblewrap` package to use it.
- Windows has no supported backend; enabling it there falls back to running
  unsandboxed with a warning in the tool output.
- When the sandbox is enabled, the approval preview shows the wrapped command
  and the network mode. If the backend is missing or fails to start, the tool
  warns and runs the command unsandboxed rather than failing.
- Config lives in the `sandbox` section:
  `{"enabled": false, "network": true, "extra_writable": []}`.

> **Limitations:** the sandbox is a damage-limiter for ordinary commands, not a
> security boundary against a hostile model or malicious code. Reads stay
> broad, the sandbox runs with your user's privileges (no user namespaces or
> credential isolation on macOS), and an unavailable backend silently degrades
> to an unsandboxed shell with only a warning. Read access to API keys stored
> on disk is not blocked, so use `--mode plan` or command rules when that
> matters.

## Remote and devcontainer execution

Shell commands can run on another machine over SSH or inside a
devcontainer/Docker container. `run_bash`, the persistent `shell`, and
`shell_jobs` all execute remotely when it is configured; the approval preview
shows the wrapper and tool output includes a `[remote]` note. It is **off by
default**:

```bash
kiwimatecoder config remote show
kiwimatecoder config remote host devbox                  # ssh [user@]host
kiwimatecoder config remote user kiwi                    # optional
kiwimatecoder config remote port 2222                    # optional (default 22)
kiwimatecoder config remote identity ~/.ssh/id_ed25519   # optional
kiwimatecoder config remote workspace /srv/app           # cd before running
kiwimatecoder config remote devcontainer auto            # auto|off|container name
kiwimatecoder config remote enable on
```

The REPL equivalent is `/config remote [show|enable on|off|host <h>|user <u>|
port <n>|identity <path>|workspace <path>|devcontainer <auto|off|name>]`.

When `devcontainer` is `auto` (the default), the first usable route wins:

1. a `.devcontainer/devcontainer.json` in the workspace **and** the
   `devcontainer` CLI: `devcontainer exec --workspace-folder <ws> sh -lc ...`;
2. that detected container through `docker exec -w <workspaceFolder>
   <name> ...`;
3. SSH when `remote.host` is set: `ssh -o BatchMode=yes [-p <port>]
   [-i <identity>] [user@]host -- sh -lc ...`, with
   `cd <remote.workspace> && <command>` when a remote workspace is set;
4. local execution with a warning in the tool output.

Set `devcontainer off` to skip detection, or name a container explicitly to
force `docker exec`. Remote wrapping happens first and a remote command is not
additionally wrapped in a local sandbox. If `ssh`, `docker`, or `devcontainer`
is missing, the command runs locally and the output carries the warning.

- Config lives in the `remote` section:
  `{"enabled": false, "host": "", "user": "", "port": 22, "identity": "",
  "workspace": "", "devcontainer": "auto"}`. `identity` must point at an
  existing key file, and enabling remote execution requires a host (or an
  explicit container name).
- The persistent shell starts through the same wrapper, so `cd` and exported
  variables persist on the remote side. SSH uses agent/key auth only
  (`BatchMode=yes`); there is no password prompt.

> **Limitations:** only commands run remotely. File tools (`read_file`,
> `write_file`, `edit_file`, `search`, ...) still read the **locally**
> accessible workspace, so point them at the remote tree with sshfs, a bind
> mount, or a synced checkout. A full SFTP-backed file layer is deliberately
> not shipped: a half-working remote filesystem would silently split reads and
> writes between two machines. Detached agent jobs (`/jobs`) also run locally
> against that same local view. SSH host keys and identity files are the
> user's responsibility; remote command approval and command rules work
> exactly as they do locally.

## Web access

`web_search` searches the web with the DuckDuckGo HTML endpoint (no API key)
and returns titles, URLs, and snippets. `web_fetch` downloads a page,
follows redirects, converts HTML to readable text, and summarizes binary
responses instead of dumping them. Both are read-only tools, so they run
without approval and stay available in plan mode, and both descriptions tell
the model to cite the URLs it relies on.

- Response size is capped with a truncation note; defaults are 50,000
  characters and a 20-second timeout.
- `localhost`, loopback/private/link-local IPs, and `*.local` hosts are
  refused by default. Enable them with `/config web allow-local on` (or
  `kiwimatecoder config web allow-local on`) when you are working against a
  local dev server.
- Settings live under `/config web show|max-chars <n>|timeout <s>|allow-local <on|off>`
  in the REPL, or `kiwimatecoder config web ...` from the shell.
- DuckDuckGo parses the public HTML page, so result extraction is best-effort
  and may need updating if that page's markup changes; `web_fetch` is
  unaffected.

## Proxy, custom CA, and offline mode

Every outbound request — chat streaming, model catalogs, `web_fetch`/
`web_search`, and embeddings — honors the `network` settings in
`~/.kiwimatecoder/config.json`:

```json
{
  "network": {
    "proxy": "http://proxy.corp:8080",
    "ca_bundle": "/etc/ssl/certs/corp-ca.pem",
    "offline": false
  }
}
```

Set them from the REPL with `/config network ...` or from the shell with
`kiwimatecoder config network ...`:

```bash
kiwimatecoder config network show
kiwimatecoder config network proxy http://proxy.corp:8080   # or: clear
kiwimatecoder config network ca /etc/ssl/certs/corp-ca.pem  # or: clear
kiwimatecoder config network offline on                     # or: off
```

- `proxy` routes all requests through an HTTP(S) proxy (`proxy`/`verify` are
  passed to the underlying `httpx` clients).
- `ca_bundle` points at a PEM file used instead of the system trust store for
  TLS verification — for corporate MITM proxies or self-signed CAs. The file
  must exist when set.
- `offline on` is air-gapped mode: cloud chat, model catalogs, web tools, and
  embeddings are refused with an "offline mode is enabled" error before any
  request is made. **Local providers keep working** — `ollama`, `lmstudio`,
  `unsloth`, and custom providers on `localhost`/`*.local` are still reachable
  for chat, catalog, and embedding calls. `web_fetch`/`web_search` always
  refuse offline because they only reach the public internet.

## Browser automation

The `browser` tool drives a headless Chromium page through Playwright when you
opt in. Playwright is not installed by default, and the tool stays hidden until
browser automation is enabled:

```bash
pip install 'kiwimatecoder[browser]'
playwright install chromium
kiwimatecoder config browser enable on
```

Without the extra, the tool returns a clear install hint instead of failing;
the base install keeps the same dependency set.

- The tool is approval-gated because a page can submit forms or trigger
  downloads. In `ask` mode the preview shows `browser <action> <url|selector>`
  (scripts are redacted and capped). It is not advertised while disabled, and
  plan mode only advertises read-only tools.
- `open` accepts only `http(s)` URLs and refuses `localhost`, loopback/private
  IPs, and `*.local` hosts unless `/config web allow-local on` is set. It
  reports the final URL after redirects and the page title.
- `text` returns visible page text capped at ~30,000 characters with a
  truncation note. `click`/`type` take a CSS selector, and `evaluate` runs
  JavaScript and returns a capped JSON-ish result.
- `screenshot` saves a PNG to
  `<workspace>/.kiwimatecoder/screenshots/<timestamp>.png` and attaches it to
  the next request, so a vision-capable model sees it exactly like a
  `view_image` attachment. The `vision` per-turn count and size limits apply.
- One page is shared by successive tool calls in a session; `close` (and
  leaving the REPL) shuts Chromium down. `timeout_ms` can override the
  configured action timeout per call.
- Settings live under `/config browser [show|enable on|off|headless on|off|timeout <ms>]`
  in the REPL, or `kiwimatecoder config browser ...` from the shell (defaults:
  off, headless on, 15,000 ms; timeout range 1s–120s). Changing a setting
  closes the live page so the next call starts with the new values.

```text
Open https://example.com/docs and summarize the visible text.
Screenshot the pricing page and tell me what stands out.
Fill the search box with "playwright" and click Search.
```

## Git and forge tools

The agent can inspect and change git state through dedicated tools instead of
dropping to `run_bash`:

- `git` (read-only) — `status`, `diff` (staged or not, optionally against a
  ref), `log` (1–100 commits), `show <ref>`, and `branches`.
- `git_write` (approval-gated) — `stage`, `unstage`, `commit` (a non-empty
  message is required), and `checkout_branch` (creates a new branch). The
  approval prompt shows the exact `git ...` command line.
- `forge` (read-only) — `pr_list`, `pr_view`, `issue_list`, and `issue_view`
  for the forge detected from the `origin` remote.
- `forge_write` (approval-gated) — `pr_create` and `issue_create`.

Forge integration goes through the official CLI (`gh` for GitHub, `glab` for
GitLab), so your existing CLI login is used and no token is read or stored by
KiwiMateCoder. The matching CLI must be installed and authenticated; on
GitLab, pull requests are handled as merge requests (`glab mr ...`).

Intentionally out of scope: `push`, `reset`, `clean`, and any forge
merge/close/delete command. Those stay manual, or go through `run_bash` where
the command rules and approval prompt apply.

## Clipboard and HTTP tools

Clipboard access uses the platform's standard tool, discovered on `PATH` and
run without a shell: `pbpaste`/`pbcopy` on macOS, `wl-paste`/`wl-copy`
(preferred) or `xclip` on Linux, and PowerShell's `Get-Clipboard`/
`Set-Clipboard` on Windows. If none is installed the tool returns a clear
install hint instead of failing.

- `read_clipboard` is read-only, so it runs without approval.
- `write_clipboard` is approval-gated and the preview shows the redacted text
  (clipped to ~200 characters).

`http_get` and `http_request` exercise HTTP APIs and print the status line,
selected response headers, and the body (JSON is pretty-printed, bodies are
truncated with a note). 4xx/5xx responses are returned for inspection rather
than raised, and ``httpx`` errors are converted to friendly messages.

- `http_get(url, headers?)` is read-only.
- `http_request(method, url, headers?, body?, json?, timeout?)` is
  approval-gated because it can change remote state. Methods are GET, POST,
  PUT, PATCH, and DELETE; `body` (raw text) and `json` are mutually exclusive.
- Both reuse the `web` settings: response size and timeout come from
  `/config web max-chars` / `/config web timeout`, and local/private addresses
  stay blocked unless `/config web allow-local on`.

```text
Read what's on my clipboard and save it to notes.txt.
Copy the test command to my clipboard.
GET https://api.github.com/repos/psf/requests and summarize the JSON.
POST https://httpbin.org/post with json {"hello": "world"}.
```

## Images

Attach images to a message three ways, all sharing the same size and count
limits:

- Type `@path/to/image.png` in a prompt. Existing image files under the
  workspace are attached and removed from the text (a lone `@image.png` sends
  "Please analyze the attached image(s)."). Paths that are missing, outside the
  workspace, or not images are left in the text untouched.
- Paste or drag an image into the terminal. Most terminals insert the file path
  at the cursor; `@`-mention extraction then attaches it on send.
- Let the model call the `view_image` tool when it needs to look at a file.

Each image is validated (extension and magic bytes), base64-encoded, and sent
to the model as an image part on the next request. Both OpenAI-compatible and
native Anthropic providers are supported. Limits live under
`/config vision [show|max-bytes <n>|max-images <n>]` (defaults: 5 MB per image,
4 images per turn); oversized or invalid images produce a clear message instead
of a failed request.

## Notebooks, PDFs, and docx

`read_file` extracts structured documents to text automatically, so the model
can read them without extra tooling:

- **Jupyter notebooks** (`.ipynb`) render as `## Cell N (type)` sections with
  cell sources plus text and stream outputs; image outputs are summarized as
  `[image/png output, N bytes]`.
- **Word documents** (`.docx`) are unzipped and `word/document.xml` is stripped
  to text with paragraph and line breaks preserved.
- **PDFs** (`.pdf`) use the `pdftotext` CLI (from Poppler) when installed for
  reliable extraction. When it is missing, a conservative stdlib fallback
  inflates FlateDecode streams and reads text from `BT`/`ET` blocks; if nothing
  is parseable the result explains how to install `pdftotext` instead of
  failing.

Document output is capped at ~200 KB with a truncation note, and `offset`/
`limit` are ignored for documents (they apply to plain text files only).
Malformed documents always return a clear message rather than raising.

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
| `azure` | `gpt-5.6-sol` *(deployment)* | `AZURE_OPENAI_API_KEY` |
| `bedrock` | `openai.gpt-5.6-sol` | `AWS_BEARER_TOKEN_BEDROCK` |
| `groq` | `llama-4.1-70b-versatile` | `GROQ_API_KEY` |
| `together` | `meta-llama/Llama-4.1-70B-Instruct-Turbo` | `TOGETHER_API_KEY` |
| `fireworks` | `accounts/fireworks/models/llama-v4-70b-instruct` | `FIREWORKS_API_KEY` |
| `cerebras` | `llama-4.1-70b` | `CEREBRAS_API_KEY` |
| `deepinfra` | `meta-llama/Llama-4.1-70B-Instruct` | `DEEPINFRA_API_KEY` |
| `ollama` | *(from server)* | `OLLAMA_API_KEY` (optional) |
| `lmstudio` | *(from server)* | `LMSTUDIO_API_KEY` (optional) |
| `unsloth` | *(from server)* | `UNSLOTH_API_KEY` (required) |

Groq, Together, Fireworks, Cerebras, and DeepInfra work out of the box once
their API key is set — they speak the standard OpenAI-compatible protocol.

`azure` and `bedrock` ship **placeholder** endpoints because the endpoint is
account-specific (`https://<resource>.openai.azure.com/openai/v1` and
`https://bedrock-runtime.<region>.amazonaws.com/openai/v1`). Point them at your
own resource by adding a custom provider (built-in entries cannot be edited):

```bash
kiwimatecoder config provider add my-azure "My Azure" \
  https://my-resource.openai.azure.com/openai/v1 my-deployment \
  --key-env AZURE_OPENAI_API_KEY --key-header api-key --key-prefix "" \
  --api-version 2024-10-21
kiwimatecoder config key set my-azure <key>

kiwimatecoder config provider add bedrock-eu "Bedrock (eu-central-1)" \
  https://bedrock-runtime.eu-central-1.amazonaws.com/openai/v1 \
  openai.gpt-5.6-sol --key-env AWS_BEARER_TOKEN_BEDROCK
```

Azure OpenAI authenticates with an `api-key` header (no `Bearer` prefix) and
versions its API with `?api-version=`; both are configurable per custom provider
(`--key-header`, `--key-prefix`, `--api-version`, or the slash equivalents).
AWS Bedrock's OpenAI-compatible endpoint accepts a bearer token only: SigV4/IAM
signing, Azure managed identity, Google Vertex AI (project-specific endpoint,
OAuth2-only auth), and interactive OAuth/device-code flows are not implemented
(deferred; see ROADMAP).

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

## Headless mode

Run one agentic turn without entering the REPL — useful for scripts and CI:

```bash
kiwimatecoder -p "Summarize the changes in the working tree."
echo "Explain src/app.py" | kiwimatecoder -p -
```

The prompt is the value of `-p/--print`; `-p -` reads it from stdin. Flags:

| Flag | Meaning |
| --- | --- |
| `--output-format text\|json\|stream-json` | `text` streams assistant text to stdout (default); `json` prints one JSON object at the end; `stream-json` prints one compact JSON object per event, ending with a `result` line. |
| `--mode ask\|auto-accept\|plan` | Permission mode for this run (default: the configured mode). |
| `--yes` | Treat approvals as granted (auto-accept) — never prompts. |
| `--max-turns N` | Hard cap on tool-loop iterations (default 30). |
| `--provider`, `--model` | Provider/model overrides for this run. |
| `--workspace PATH` | Workspace root (default: current directory). |
| `--quiet` | Suppress tool/progress lines on stderr in text mode. |

Tool and progress lines go to **stderr**; only assistant text (text mode) or
JSON (json/stream-json modes) goes to **stdout**. Headless runs never prompt:
actions that would need approval are denied with a note on stderr unless
`--yes` or `--mode auto-accept` is used.

Exit codes: `0` success, `1` runtime failure (missing key, provider error,
turn cap without a final answer), `2` invalid usage (bad format, mode, or
workspace).

CI example:

```yaml
- name: Review the diff
  run: |
    kiwimatecoder -p "Review the staged diff and list risks." \
      --output-format json --max-turns 10 > review.json
    cat review.json
```

More recipes (ready-made GitHub Actions workflows, exit-code gating, and
pre-commit setup) live in [docs/ci.md](docs/ci.md).

## Shell completions and man page

Typer ships shell completion for bash, zsh, fish, and PowerShell:

```bash
kiwimatecoder --install-completion          # detect and install for your shell
kiwimatecoder --show-completion bash        # print the script to stdout instead
```

A troff man page is committed at `docs/kiwimatecoder.1`; view it with `man
./docs/kiwimatecoder.1` (on Linux, `man -l docs/kiwimatecoder.1`). Regenerate it
from the Typer command tree after changing the CLI:

```bash
python scripts/generate_man.py           # rewrite the man page
python scripts/generate_man.py --check   # fail when the committed file is stale
```

Interactive sessions need a real terminal. When stdin is not a TTY (a pipe, a
CI job, cron), the bare command — and `--resume`/`--continue` — exits with code
`2` and a note pointing at headless mode instead of starting prompt_toolkit.
Use `kiwimatecoder -p "..."` or `echo "..." | kiwimatecoder -p -`.

## Python SDK

Embed the agent in a Python program with `kiwimatecoder.sdk`:

```python
from kiwimatecoder.sdk import run_agent_sync

result = run_agent_sync(
    "Summarize README.md in three bullets.",
    workspace=".",
    mode="plan",
)

if result.success:
    print(result.text)
print(result.usage, result.cost_usd)  # tokens and estimated USD (or None)
```

`RunResult` carries `text`, `usage` (`prompt_tokens`/`completion_tokens`),
`cost_usd`, `provider`, `model`, `mode`, `tools_used`, `messages`, `success`,
and `error`. Pass `on_event` to observe the same events as stream-JSON
(`text_delta`, `tool_start`, `tool_end`, `usage`, `done`), `confirm` to
approve actions, and `console` to receive the agent's tool/progress output.
Approvals are denied by default. From async code use `await run_agent(...)`;
`run_agent_sync` raises a clear error if called inside a running loop.

## Editor integration (ACP)

KiwiMateCoder can run as an [Agent Client Protocol](https://agentclientprotocol.com)
(ACP) agent over stdio, so an ACP-capable editor can drive it:

```bash
kiwimatecoder acp [--workspace PATH]
```

A Zed-style agent configuration (`settings.json`) looks like:

```json
{
  "agent": {
    "command": "kiwimatecoder",
    "args": ["acp"]
  }
}
```

The server speaks JSON-RPC 2.0 over newline-delimited JSON on stdin/stdout;
the only thing ever written to stdout is protocol traffic. Implemented subset:

- `initialize` handshake with agent capabilities and version info.
- `session/new` creates one conversation rooted at the client's `cwd`,
  using the **configured** permission mode (never an assumed auto-accept).
- `session/prompt` runs one agent turn, streaming assistant text chunks and
  tool-call progress as `session/update` notifications; the response carries
  `end_turn`, `cancelled`, `refusal`, or `max_tokens`.
- `session/cancel` cancels the running turn; partial assistant text is kept.
- `session/request_permission` bridges every write/command approval to the
  editor when the session runs in `ask` mode. Approving with "always"
  persists the tool allowlist; a cancelled or unanswered request (default
  timeout 300s, `config acp timeout <s>`) is denied exactly like a normal
  denial.

This is a focused subset: terminal and MCP client capabilities advertised by
the editor are not wired through (MCP servers still come from KiwiMateCoder
config), `loadSession` is not supported, image prompt parts are ignored with a
note to the model, and the file tools operate on the local filesystem. When
the client advertises `fs.readTextFile`/`fs.writeTextFile`, the server exposes
`read_text_file`/`write_text_file` helpers that delegate to
`fs/read_text_file`/`fs/write_text_file`; otherwise those helpers use the
local filesystem.

## Background jobs

Run an agent task detached from your terminal, then check on it later:

```bash
kiwimatecoder jobs run "Add type hints to kiwimatecoder/jobs.py and run mypy."
kiwimatecoder jobs list
kiwimatecoder jobs show <id>
kiwimatecoder jobs output <id>
kiwimatecoder jobs cancel <id>
kiwimatecoder jobs clear          # finished jobs only; --all includes running
```

Each job is one headless run (`python -m kiwimatecoder -p ... --output-format
json`) recorded as JSON under `~/.kiwimatecoder/jobs/<id>.json` with its combined
output at `~/.kiwimatecoder/jobs/<id>.log`. Records survive restarts; `jobs list`
refreshes running jobs from their process and parsed result, and corruption in
one record never hides the others.

> **Warning:** `jobs run` defaults to `--mode auto-accept`, so an unattended job
> can edit files and run commands without asking. Pass `--mode ask` (approvals
> are then denied, since nothing can prompt) or `--mode plan` (read-only) to
> constrain a job, and only point jobs at workspaces you trust.

Inside a session, `/jobs [list|show <id>|cancel <id>|tick]` does the same, and
`/jobs run <prompt>` starts a job in the session's workspace (still
`auto-accept`). Jobs are marked `running`, `succeeded`, `failed`, or
`cancelled`; `cancelled` covers an explicit `jobs cancel`.

### Scheduling

Scheduling is pull-based — there is no daemon and no platform cron integration.
`jobs schedule "<prompt>" --every <seconds>` starts a run immediately and stores
the interval plus a `next_run_at` timestamp; every later invocation of
`kiwimatecoder jobs tick` (or `/jobs tick`) starts any run that is due. A job
that is still running is skipped, so runs never overlap. Wire `jobs tick` to
your own scheduler, for example:

```cron
*/5 * * * * kiwimatecoder jobs tick
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

## Memory

The agent can remember durable facts about you and your projects across
sessions. Memory is plain Markdown, one `- fact` bullet per line, in two
scopes:

- **Project** — `<workspace>/.kiwimatecoder/memory.md`; travels with the
  repository and applies only there.
- **User** — `~/.kiwimatecoder/memory.md`; shared across every workspace.

The model writes with the approval-gated `remember` tool (`scope=project` or
`scope=user`) and reads with the read-only `recall` tool. Active facts are
loaded into the system prompt on every turn, capped at 16KB by default so a
pathological file cannot crowd out the conversation.

Manage memory directly with `/memory`:

```text
/memory                       # show both files and their facts
/memory add The build uses uv, not pip
/memory add-user I prefer concise answers
/memory clear project
/memory clear user
```

Settings live under `kiwimatecoder config memory show|max-bytes <n>|enable on|off`.
Never store secrets or API keys in memory — the prompt section says so
explicitly, and the `remember` preview runs the same secret redaction as audit
output.

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

## MCP servers

MCP (Model Context Protocol) servers extend the agent with tools it discovers
at session start. Configure them under `mcp_servers` in
`~/.kiwimatecoder/config.json` — each entry is either a stdio subprocess or an
HTTP endpoint:

```json
{
  "mcp_servers": {
    "files": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {"SOME_TOKEN": "..."},
      "disabled": false
    },
    "remote": {
      "url": "https://mcp.example.com/mcp",
      "headers": {"Authorization": "Bearer sk-..."},
      "disabled": false
    }
  }
}
```

From the shell, `kiwimatecoder config mcp add <name>` writes the same entries
(set exactly one of `--command` or `--url`):

```bash
kiwimatecoder config mcp add files --command npx --args "-y @modelcontextprotocol/server-filesystem /tmp"
kiwimatecoder config mcp add remote --url https://mcp.example.com/mcp -H "Authorization: Bearer sk-..."
kiwimatecoder config mcp list
kiwimatecoder config mcp remove remote
```

In the REPL, `/mcp list` shows every configured server, its status (connected,
disabled, or the failure reason), its resources count, and the registered tools.
`/mcp reload` reconnects each enabled server and re-registers its tools without
restarting the session.

Each server tool is registered as `mcp__<server>__<tool>` (both parts sanitized
to `[A-Za-z0-9_]`), with the server's JSON Schema used for arguments. Tools that
declare `annotations.readOnlyHint` run without approval; every other MCP tool
prompts for approval in `ask` mode, exactly like `write_file` or `run_bash`.
A server that fails to start or initialize is reported as a dim warning and
skipped — it never blocks startup. Connections close when the session exits.

Authentication is whatever headers (HTTP) or environment variables (stdio) you
configure. Interactive OAuth is not implemented yet; see ROADMAP.md.

## Language server diagnostics

When enabled, KiwiMateCoder can talk to language servers for compiler-grade
diagnostics and navigation. It is **opt-in** and off by default; no server
process starts until the feature is enabled and a matching file is queried.

- `lsp_diagnostics` — errors and warnings for one file, or for the five most
  recently touched files when no path is given.
- `lsp_definition` / `lsp_references` — where a symbol is defined, and every
  place it is used. Positions are 1-based line and column numbers.

Built-in presets start automatically when their command is on `PATH`:

| Language | Command | Extensions |
|----------|---------|------------|
| python | `pyright-langserver --stdio` | `.py`, `.pyi` |
| typescript | `typescript-language-server --stdio` | `.ts`, `.tsx`, `.js`, `.jsx` |
| go | `gopls` | `.go` |
| rust | `rust-analyzer` | `.rs` |
| c/cpp | `clangd` | `.c`, `.h`, `.cpp`, `.hpp` |

After a successful `write_file`/`edit_file`, diagnostics for the edited file
are appended to the tool result as `LSP:` lines (capped at ten) when
`diagnostics_after_edits` is on. A timeout or missing server appends nothing,
so LSP can never fail a turn or delay it beyond the configured timeout.

Enable it in-session with `/lsp on`, or from the shell:

```bash
kiwimatecoder config lsp enable on            # on | off
kiwimatecoder config lsp after-edits on       # on | off
kiwimatecoder config lsp show
```

Use `/lsp status` to see whether LSP is on, which server commands are
installed, and which clients are running; `/lsp restart` stops running servers
so they start fresh on next use. Settings live under `lsp` in
`~/.kiwimatecoder/config.json`; add your own server or override a preset:

```json
{
  "lsp": {
    "enabled": true,
    "timeout": 10.0,
    "diagnostics_after_edits": true,
    "servers": {
      "python": {
        "command": "basedpyright-langserver",
        "args": ["--stdio"],
        "extensions": [".py", ".pyi"]
      }
    }
  }
}
```

## Codebase index and semantic search

`search` gains a third mode: `mode='semantic'` ranks files against a
natural-language query using a local, persistent index instead of a regex, and
returns paths, line numbers, and source snippets. The agent chooses it for
questions like "where is authentication handled?", and grep/glob keep working
exactly as before.

The index is built automatically the first time semantic search runs, then
refreshed incrementally: only files whose size or modification time changed
are re-tokenized, deleted files are dropped, and `.gitignore` plus the default
skip directories (`node_modules`, `venv`, `__pycache__`, …) are respected.
Binary files, files larger than the byte cap, and anything beyond the file cap
are skipped. It is a cache: the store lives at
`~/.kiwimatecoder/index/<hash-of-workspace>.json` and `clear` simply deletes it.

```text
/index                 # status: files, terms, store size, stale count, embeddings
/index build           # refresh incrementally; prints added/updated/removed
/index clear           # delete the store for this workspace
```

Ranking uses BM25 over file term frequencies, so there are no API calls and
semantic search works out of the box. Settings live under the `index` key:

```json
{
  "index": {
    "enabled": true,
    "max_files": 5000,
    "max_file_bytes": 262144,
    "embeddings": { "provider": "", "model": "", "batch_size": 32 }
  }
}
```

Embeddings are optional and **OpenAI-compatible only**: set a provider id and
an embedding model (the provider must serve `POST /embeddings` and its API key
is read from the same config/env as chat) to additionally store chunk vectors.
When vectors exist, search blends the lexical and vector scores; if the
embedding provider is unavailable or errors for any reason, search silently
falls back to lexical ranking. Embeddings are never required to use the
feature.

```bash
kiwimatecoder config index show
kiwimatecoder config index embed-provider openai
kiwimatecoder config index embed-model text-embedding-3-small
kiwimatecoder config index embed-provider none    # turn embeddings off
kiwimatecoder config index clear
```

## Subagents

The `task` tool delegates a focused, self-contained investigation to a
subagent that runs in its own context with the same workspace, provider, and
tools. The subagent's intermediate tool output stays out of the parent
conversation; it returns a final report (plus a step/token count) as the tool
result. Use it for broad searches or self-contained questions that would
otherwise flood the main context.

- Spawning a subagent is approval-gated like a command because the subagent can
  write files and run commands. Its own actions still go through the normal
  permission gate (including plan mode and command rules), and its edits are
  checkpointed so `/undo` covers them.
- Subagents cannot spawn nested subagents and cannot use `ask_user`; put every
  decision they need in the prompt.
- A subagent shares the parent's remaining token budget and stops at the
  configured step limit, returning a partial report with a note when either cap
  is hit. While subagents are disabled, `task` is not advertised at all.

```json
{
  "subagents": {
    "enabled": true,
    "max_steps": 20,
    "model": ""
  }
}
```

```bash
kiwimatecoder config subagents show
kiwimatecoder config subagents enable on
kiwimatecoder config subagents max-steps 10
kiwimatecoder config subagents model gpt-5-mini
```

Or in the REPL: `/config subagents [show|enable on|off|max-steps <n>]`.

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

## Themes, output modes, and accessibility

Control color independently of your terminal, pick a theme, tune how much the
agent prints, and switch to ASCII glyphs for screen readers and limited
terminals:

```bash
kiwimatecoder config ui show
kiwimatecoder config ui color never        # auto | always | never
kiwimatecoder config ui theme ocean        # default | ocean | magenta | mono
kiwimatecoder config ui output compact     # normal | compact | verbose
kiwimatecoder config ui ascii on           # on | off
```

The same settings are available in-session with `/config ui ...`. Values live
under the `ui` key in `~/.kiwimatecoder/config.json`:

```json
{
  "ui": {
    "theme": "ocean",
    "color": "auto",
    "output_mode": "normal",
    "ascii": false
  }
}
```

Color precedence: an explicit `ui.color` of `always`/`never` wins; otherwise
`NO_COLOR` set to anything non-empty disables color, then `FORCE_COLOR` set to
anything non-empty enables it. With `auto` and neither variable exported, Rich
still detects a pipe or log file and stays plain.

Output modes shape the tool log. `normal` prints a line per tool with a success
or failure marker. `compact` hides successful tool lines while keeping failures,
the thinking status, and the final answer. `verbose` adds a redacted argument
block (secrets replaced, capped at ~500 characters) before each tool runs and
the result size after.

ASCII mode replaces the check, cross, blocked, folder, and bullet glyphs with
plain text (`[ok]`, `[fail]`, `[blocked]`, no folder prefix, `-`) and the CLI
confirmations follow suit. Color, theme, ASCII, and output-mode changes apply
to the next session.

## Model routing

Route short, plain requests to a cheaper model while keeping the session model
for real work. Configure it under `model_routing` in
`~/.kiwimatecoder/config.json`:

```json
{
  "model_routing": {
    "enabled": true,
    "simple_model": "gpt-5-mini",
    "simple_max_chars": 200,
    "exclude_keywords": ["refactor", "implement", "debug", "test"]
  }
}
```

A turn is routed when routing is enabled, `simple_model` is set, the message is
at most `simple_max_chars` long, and it contains no code fence, file path,
slash command, or exclude keyword (whole-word, case-insensitive). The override
applies to the primary provider only — fallback providers keep their own
models — and the session's model is never changed.

## Prompt caching

Native Anthropic providers support prompt caching. Enable it with
`kiwimatecoder config cache on` (or `/config cache on`): requests then send the
system prompt as a cacheable content block and mark the last tool definition as
an ephemeral cache breakpoint, cutting latency and input cost when the same
system prompt and tools repeat across turns. The toggle only changes native
Anthropic payloads — OpenAI-compatible providers cache automatically, so their
requests are byte-for-byte identical either way.

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
provider/model, default mode, sampling, output style, theme/color/output/ASCII
preferences, custom prompt, and persisted tool approvals). The original
single-key `~/.kiwimatecoder/config` format is read automatically, so existing
setups keep working.

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

## Profiles

Profiles are named presets of global settings: provider, model, mode, sampling,
output style, custom system prompt, verify command, budget, trusted workspace,
command rules, and persisted `always` approvals. Capture the current effective
settings, then re-apply them anywhere:

```bash
kiwimatecoder config profile save work
kiwimatecoder config profile list
kiwimatecoder config profile show work
kiwimatecoder config profile use work      # writes the preset into config
kiwimatecoder config profile remove work
```

Launch once with a profile without persisting it with `kiwimatecoder --profile
work`. On a fresh session the provider, model, mode, and other session-level
settings come from the profile; with `--resume`/`--continue` only the mode and
model are overlaid so the restored conversation keeps its provider. Sampling
and budget are global settings, so use `config profile use` to apply those.
The REPL equivalent is `/config profile list|show|save|use|remove`.

## Cross-machine session sync

Sync saved sessions across machines through a folder you already share (Dropbox,
iCloud Drive, a network mount, or a git checkout). There is no server component
and nothing is sent anywhere: the feature is off by default and only reads and
writes the folder you point it at.

Enable it with an existing directory:

```bash
kiwimatecoder sync enable ~/Dropbox/kiwimatecoder
```

or edit `~/.kiwimatecoder/config.json` directly:

```json
{
  "sync": {
    "enabled": true,
    "path": "~/Dropbox/kiwimatecoder",
    "machine": "laptop",
    "include_autosave": false
  }
}
```

The folder gets a `kiwimatecoder-sessions/` subdirectory holding the session
JSON files plus a `manifest.json` index. `machine` defaults to a slug of the
hostname; set it explicitly when two machines would otherwise share a name.

```bash
kiwimatecoder sync status    # local/remote counts, pending changes, conflicts
kiwimatecoder sync push      # copy local sessions into the shared folder
kiwimatecoder sync pull      # copy newer shared sessions into the local store
kiwimatecoder sync disable   # opt out; the shared folder is left alone
```

The same actions are available in the REPL as `/sync [status|push|pull]`.
Everything mirrors the local session store except the implicit `last.json`
autosave, which is skipped by default (`include_autosave: true` opts in) because
it changes on every exit. Unreadable or corrupt files are skipped and reported,
never fatal.

**Conflicts.** Each machine remembers what it last pushed or pulled. On the next
`push`, a shared file that changed elsewhere *and* whose local copy also changed
is kept as both: the shared copy stays put and the local copy is uploaded as
`<name>__<machine>.json`. Every other difference is last-write-wins by the
session's `saved_at` timestamp. `pull` uses the same rule but renames the
incoming file to `<name>__<remote-machine>.json` instead of overwriting the
local one. Pass `--force` to `push`/`pull` to make the local copy (push) or the
shared copy (pull) win unconditionally.

## Config validation

`kiwimatecoder config validate` checks the stored config and prints a table of
issues, exiting non-zero when any error-level problem is found. It surfaces
exactly what the tolerant getters would silently drop: unknown top-level keys
(warning), malformed provider/model-filter/sampling/budget/hook/command-rule/
profile/MCP/plugin entries, invalid regexes, unknown hook events, bad network
(proxy/CA/offline) values, bad ACP permission timeouts, and bad default mode,
output style, UI (theme/color/ASCII/output mode), or workspace-flag values.

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
P2 is complete (event bus + hooks, the extensible tool registry, custom
commands and prompt templates, on-demand Agent Skills, a plugin system, the MCP
client, config profiles with schema validation, per-turn model routing,
Anthropic prompt caching, and themes/output modes/NO_COLOR accessibility).
