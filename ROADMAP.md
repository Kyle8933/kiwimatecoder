# KiwiMateCoder Roadmap

Prioritized delivery plan for every feature a coding CLI user might want.
Effort is solo-dev: S ≤ 1 day, M = 2–4 days, L = 1–2 weeks, XL = 2–4 weeks.

Sequencing logic: fix credibility (tiny) → durability and safety (P0/P1) →
extensibility (P2) → reach/intelligence (P3) → platform/automation (P4) →
ecosystem polish (P5).

## P0 — Foundation and quick wins (1–2 weeks)

Goal: nothing users touch gets lost, config is future-proof, and docs are accurate.

- [x] 0.1 Fix stale Anthropic comments/docs (`providers.py`, README)
- [x] 0.2 Persistent prompt history (`FileHistory`)
- [x] 0.3 Auto-save sessions + `--continue` / `--resume last`
- [x] 0.4 Accurate token counting + per-model pricing table
- [x] 0.5 Config v2: schema version, project `.kiwimatecoder.json`, precedence
- [x] 0.6 Persisted always-allow rules (survive provider switch/restart)
- [x] 0.7 `/doctor` diagnostics
- [x] 0.8 Sampling params (temperature/top_p/max_tokens/reasoning effort)
- [x] 0.9 Auto-load project instructions (`AGENTS.md` + scoped rules)
- [x] 0.10 Custom system prompt + built-in output styles

## P1 — Trustworthy agent core (4–6 weeks)

Goal: users can let the agent edit freely because every action is undoable,
reviewable, and bounded.

- [x] 1.1 Session store v2: versioned format, fork/branch, export/import (M)
- [x] 1.2 Checkpoints + `/undo` + `/rewind` (L; depends 1.1)
- [x] 1.3 Command allow/deny rules, e.g. deny `rm -rf` (M; depends 0.5, 0.6)
- [x] 1.4 Hunk-level diff accept/reject in approvals (M)
- [x] 1.5 Secrets scanning/redaction in context and logs (M)
- [x] 1.6 Todo/plan tool (M)
- [x] 1.7 Parallel tool execution (M)
- [x] 1.8 Context compaction/summarize + window gauge (M; depends 0.4)
- [x] 1.9 Auto-verify loop: run tests/lint after edits (M)
- [x] 1.10 Budget limits + alerts per session/day (M; depends 0.4)
- [x] 1.11 Message steering/queueing + resume after interrupt (M)
- [x] 1.12 Structured ask-user tool (S)
- [x] 1.13 Dry-run mode (S)
- [x] 1.14 Audit log of every tool action (S)
- [x] 1.15 Trusted-workspace flag for outside-root reads (S; depends 0.5)

## P2 — Extensibility platform (6–9 weeks)

Goal: the community can add capabilities without touching core.

- [x] 2.1 Event bus + hooks (session/tool/approval lifecycle) (L)
- [x] 2.2 Tool registry v2: dynamic registration, tool sources (M)
- [x] 2.3 Plugin system + script-defined custom tools (L; depends 2.1, 2.2)
- [x] 2.4 Custom slash commands + prompt templates (M; depends 0.5)
- [x] 2.5 Agent Skills (markdown instruction bundles) (M; depends 0.9)
- [x] 2.6 MCP client (stdio + HTTP, tools/resources; bearer/header auth; interactive OAuth deferred) (XL; depends 2.2)
- [x] 2.7 Profiles/presets + config schema validation (M; depends 0.5)
- [x] 2.8 Themes, output modes, `NO_COLOR`, accessibility basics (M)
- [x] 2.9 Model routing per task type (M; depends 0.5)
- [x] 2.10 Prompt caching headers where supported (M; depends 0.4)

**2.6 follow-up:** interactive OAuth is deferred. HTTP MCP servers authenticate
with configured headers (e.g. `Authorization: Bearer ...`) and env vars only.
Resources are exposed through the client API (`list_resources`/`read_resource`)
and a count in `/mcp list`; `prompts/*` is not surfaced as a user command yet.

## P3 — Reach and intelligence (8–12 weeks)

Goal: the agent can see the whole project and the outside world.

- [x] 3.1 Web fetch + web search tools (M)
- [x] 3.2 Git tools + GitHub/GitLab PR & issue integration (L)
- [x] 3.3 Vision/image input (paste, drag, @path) (L; depends 2.8)
- [x] 3.4 LSP diagnostics: post-edit errors, definitions, references (L; depends 2.2)
- [x] 3.5 Memory: project + user-level persisted facts (L; depends 0.9)
- [x] 3.6 Codebase indexing / semantic search (XL; depends 0.4)
- [x] 3.7 Subagents / Task tool (XL; depends 1.7, 2.2)
- [x] 3.8 Notebook (.ipynb), PDF/doc reading (M)
- [x] 3.9 Browser automation (Playwright) (L; depends 2.3)
- [x] 3.10 Clipboard + HTTP/API testing tools (S–M)

## P4 — Platform and automation (10–16 weeks)

Goal: usable in CI, IDEs, and remote environments.

- [x] 4.1 Headless mode: `-p/--print`, JSON/stream-JSON, exit codes, render abstraction (L)
- [ ] 4.2 GitHub Action / CI recipes + pre-commit (M; depends 4.1)
- [ ] 4.3 IDE integration via ACP/LSP server (XL; depends 4.1)
- [ ] 4.4 Remote/SSH workspaces + devcontainers (XL; depends 4.5)
- [x] 4.5 Persistent shell sessions + background process management (M)
- [ ] 4.6 Scheduled/background agent tasks (L; depends 4.1, 2.1)
- [ ] 4.7 Azure/Bedrock/Vertex/gateway providers + OAuth/device flow (L; depends 0.5)
- [ ] 4.8 Proxy/CA config + offline/air-gapped mode (M; depends 0.5)
- [ ] 4.9 Cross-machine session sync (opt-in) (XL; depends 1.1)
- [ ] 4.10 Non-TTY fallback, shell completions (bash/zsh/fish), man page (M)
- [ ] 4.11 OS-level sandbox (seatbelt/landlock) + network policy (L; depends 1.3)
- [x] 4.12 Embedded SDK / headless library API (M; depends 4.1)

## P5 — Ecosystem and polish (ongoing)

Goal: breadth of integrations and long-tail quality.

- [ ] 5.1 Media generation (wire up `media.py` stub) (L; depends 2.3)
- [ ] 5.2 Telemetry/OTel + debug logging + opt-in crash reports (M; depends 2.1)
- [ ] 5.3 Evals/benchmark harness for prompt/tool regressions (L; depends 4.1)
- [ ] 5.4 Team sharing: shared policies, session links, SSO (XL; depends 4.9)
- [ ] 5.5 i18n of CLI strings (L; depends 2.8)
- [ ] 5.6 Vim/Emacs keybindings, notifications, image paste, @-mentions (M; depends 2.8)
- [ ] 5.7 Packaging: Homebrew/curl/Docker, version pin/rollback, SBOM, Windows CI (M–L)
- [ ] 5.8 Full accessibility audit (M; depends 2.8)

## Critical paths

- **Durability:** 0.5 → 0.6 → 1.1 → 1.2 (undo is the highest-value trust feature)
- **Safety:** 1.3 → 1.14 → 4.11
- **Cost:** 0.4 → 1.10 → 0.8/2.10
- **Ecosystem:** 2.2 → 2.3/2.6 → 3.7 (subagents need parallel tools 1.7)
- **Automation:** 4.1 → 4.2/4.3/4.12
- **Context:** 0.9 → 3.5; 0.4 → 3.6

## Verification bar

Every item lands with pytest coverage (`tests/` mirrors modules) and passes
`pytest`, `ruff check`, and `mypy kiwimatecoder`.
