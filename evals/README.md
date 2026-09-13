# Eval cases

Declarative regression checks for prompts and tools. Each case is a small JSON
file with a prompt, an optional starting workspace, and deterministic
expectations about the run. The harness calls the same code path as
`kiwimatecoder -p`, so a case measures the real agent.

## Running

Eval runs call a real model and need a configured provider key
(`kiwimatecoder setup`):

```bash
kiwimatecoder eval list
kiwimatecoder eval run --provider openrouter --model anthropic/claude-sonnet-5
kiwimatecoder eval run --filter create --json --report eval-report.json
```

Exit codes: `0` when every case passes, `1` when any case fails, `2` on invalid
input (missing directory, malformed JSON, unknown provider). `--json` prints
the report and `--report PATH` writes it, so a CI job can archive results.

Nothing is sent anywhere beyond the configured provider: the runner materializes
each case in a throwaway temp workspace, runs `sdk.run_agent_sync` with
`mode="auto-accept"` (no prompts), and evaluates the `RunResult` and workspace
files afterwards.

## Schema

```json
{
  "name": "create-file-from-prompt",
  "description": "Writes a requested file with exact content.",
  "prompt": "Create hello.txt whose contents are exactly 'hello from the eval'.",
  "files": {
    "notes.txt": "Starting workspace file."
  },
  "expect": {
    "text_contains": ["hello"],
    "text_not_contains": ["error"],
    "tools_called": ["write_file"],
    "tools_not_called": ["run_bash"],
    "files": {"hello.txt": ["hello from the eval"]},
    "success": true
  }
}
```

- `name` (required): unique across the cases directory; the harness sorts by it.
- `description`: shown by `eval list`.
- `prompt` (required): the user message.
- `files`: files written into the temp workspace before the run. Paths must be
  relative, forward-slash, and stay inside the workspace.
- `expect` (required): every field is optional.
  - `text_contains` / `text_not_contains`: substrings checked against the final
    assistant text.
  - `tools_called` / `tools_not_called`: tool names checked against the tools
    the run used (for example `read_file`, `write_file`, `edit_file`, `search`,
    `run_bash`, `shell`).
  - `files`: workspace path -> substrings that must appear in the file after
    the run. A missing file fails too.
  - `success`: expected value of the run's own success flag. Without it, a run
    error is still a failure; set `false` to assert that a run fails.

Unknown keys, wrong types, empty lists of names, and unsafe paths are rejected
with the file name and a clear message. Expectations are deterministic: there
is no model-graded scoring yet, so keep cases small and assertions stable
across providers. Prefer checking an exact filename or substring over a full
sentence the model may reword.

To add a case, copy one of the files in `evals/cases/`, edit it, and run
`kiwimatecoder eval list` to confirm it validates.
