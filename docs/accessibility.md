# Accessibility audit

Scope: the interactive REPL (`kiwimatecoder`), the management and headless
CLI, and their Rich/prompt_toolkit rendering. This is an honest survey of what
works, what does not, and what to do next. It makes no claim of WCAG
compliance or assistive-technology certification.

## What works today

- **Color control.** `ui.color auto|always|never`, plus `NO_COLOR` and
  `FORCE_COLOR`; Rich also strips styling when stdout/stderr is a pipe or log
  file. The global `--plain` flag forces color off for one run.
- **ASCII glyphs.** `ui.ascii on` (or `--plain`) replaces the check, cross,
  blocked, folder, and bullet glyphs with `[ok]`, `[fail]`, `[blocked]`, no
  folder prefix, and `-`; command confirmations follow suit.
- **Output volume.** `ui.output_mode normal|compact|verbose`. `compact` drops
  successful tool lines while keeping failures, the thinking status, and the
  final answer.
- **Animated output.** `ui.spinner auto|on|off`. `off` (and `--plain`) skips
  the Rich status spinners entirely, so there is no animation, cursor
  movement, or in-place redraw from the agent.
- **Language.** `ui.locale` / `KIWIMATECODER_LANG` (currently `en`, `de`,
  `es`) for the banner hint, prompt and approval labels, common errors,
  `/help` group titles, and key CLI messages.
- **Keyboard-only navigation.** Slash commands, `choice`/checkbox selectors,
  approvals (`y`/`n`/`always`/`hunks`), Alt+Enter newline, Ctrl-C cancel,
  Ctrl-D exit, and Tab completion. `ui.keybindings emacs|vim` switches the
  prompt's editing model.
- **Bounded output.** Tool results are summarized, web fetches and pinned
  files are byte-capped, and the agent does not stream unbounded dumps.
- **Audio cue.** `ui.notify bell|desktop` can ring the terminal bell or send
  an OSC 9 / `osascript` / `notify-send` notification when a turn takes longer
  than `ui.notify_after_seconds`.
- **Plain preset.** The global `--plain` flag forces ASCII glyphs, no color,
  compact output, and no spinner for a single run without persisting anything:
  `kiwimatecoder --plain` or `kiwimatecoder --plain -p "..."`.

## What does not work today

- **Screen-reader semantics.** Terminal output is a linear stream of styled
  text. Rich tables and panels carry no roles, labels, or landmarks, and there
  is no live-region equivalent: appended tool lines and in-place status
  updates are not announced as such. No testing has been done with
  Orca, VoiceOver, or NVDA.
- **i18n coverage is partial.** Only a representative subset of strings is
  translated; most command help, diagnostics, and long-form messages remain
  English-only. There is no pluralization, no locale-aware number/date
  formatting, and no right-to-left support.
- **Reduced motion is limited.** `ui.spinner off` removes spinners, but there
  is no single setting that also flattens streamed redraws and progress
  output.
- **High contrast is not audited.** The `mono` theme reduces accent color to
  white, but no palette has been checked against WCAG contrast ratios on
  common dark and light terminal themes, and terminal colors are not detected.
- **Notification delivery is best-effort.** OSC 9 support varies by terminal,
  and `osascript`/`notify-send` may not be installed or permitted.
- **Image paste has no clipboard-bytes path.** Terminals that insert a file
  path work through `@`-mention extraction, but there is nothing for
  terminals that only put image bytes on the clipboard.
- **`--plain` is per-process.** It is not a persisted profile, and the
  management CLI still uses Rich tables and panels in plain mode.

## Concrete follow-ups

1. **Screen-reader mode:** a setting that emits strictly line-oriented output
   with no in-place redraws and announces tool boundaries with plain textual
   prefixes (`[tool: read_file] ...`).
2. **Contrast audit:** check and tune the `default`/`mono` themes for at least
   a 4.5:1 text contrast ratio on common dark and light terminal palettes, and
   document recommended terminal settings.
3. **i18n completeness:** route every user-facing string through `t()`, add a
   lint that flags unwrapped strings, and support plurals plus locale-aware
   number/date formatting.
4. **One reduced-motion switch** covering spinners, streamed redraws, and
   progress indicators.
5. **Assistive-technology testing:** run a documented protocol against Orca,
   VoiceOver, and NVDA and record the results here.
6. **`--accessible` preset** (plain + spinner off + bell notification) with a
   smoke test.
7. **Clipboard image fallback** for terminals that do not insert file paths.

## Testing this build

`tests/test_accessibility.py` exercises spinner suppression, the `--plain`
flag, config isolation, and checks that this document stays present and
current. `tests/test_ui.py` and `tests/test_config_v2.py` cover the `ui`
config keys used above.
