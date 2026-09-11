from __future__ import annotations

import difflib

from kiwimatecoder.hunks import apply_selected_hunks, parse_hunk_selection, split_hunks

OLD = "".join(f"line{i}\n" for i in range(1, 21))
NEW = OLD.replace("line1\n", "LINE1\n", 1).replace("line20\n", "LINE20\n", 1)


def make_diff(old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="f.txt",
            tofile="f.txt",
            n=3,
        )
    )


def test_split_hunks_records_metadata():
    hunks = split_hunks(make_diff(OLD, NEW))

    assert [hunk.index for hunk in hunks] == [1, 2]
    assert hunks[0].header == "@@ -1,4 +1,4 @@"
    assert hunks[0].old_start == 1
    assert hunks[0].old_count == 4
    assert hunks[0].new_start == 1
    assert hunks[0].new_count == 4
    assert "-line1\n" in hunks[0].lines
    assert "+LINE1\n" in hunks[0].lines


def test_split_hunks_ignores_creation_preview():
    assert split_hunks("+++ create f.txt\n+alpha\n+beta") == []


def test_apply_single_hunk_keeps_other_hunks_original():
    diff = make_diff(OLD, NEW)

    only_first = apply_selected_hunks(OLD, diff, [1])
    only_second = apply_selected_hunks(OLD, diff, [2])

    assert only_first == OLD.replace("line1\n", "LINE1\n", 1)
    assert only_second == OLD.replace("line20\n", "LINE20\n", 1)


def test_apply_all_and_no_hunks():
    diff = make_diff(OLD, NEW)

    assert apply_selected_hunks(OLD, diff, None) == NEW
    assert apply_selected_hunks(OLD, diff, [1, 2]) == NEW
    assert apply_selected_hunks(OLD, diff, []) == OLD


def test_apply_handles_missing_trailing_newline():
    old = "alpha\nbeta\ngamma"
    new = "ALPHA\nbeta\ngamma"
    diff = make_diff(old, new)

    assert apply_selected_hunks(old, diff, [1]) == new
    assert apply_selected_hunks(old, diff, None) == new
    assert apply_selected_hunks(old, diff, []) == old


def test_apply_handles_added_trailing_newline():
    old = "a\nb"
    new = "a\nb\n"
    diff = make_diff(old, new)

    assert apply_selected_hunks(old, diff, None) == new
    assert apply_selected_hunks(old, diff, []) == old


def test_apply_handles_unterminated_final_hunk():
    old = "".join(f"line{i}\n" for i in range(1, 20)) + "last"
    new = old.replace("line1\n", "LINE1\n", 1).replace("last", "LAST")
    diff = make_diff(old, new)

    assert apply_selected_hunks(old, diff, [2]) == old.replace("last", "LAST")
    assert apply_selected_hunks(old, diff, [1]) == old.replace(
        "line1\n", "LINE1\n", 1
    )


def test_apply_creation_preview():
    preview = "+++ create f.txt\n+alpha\n+beta"

    assert apply_selected_hunks("", preview, [1]) == "alpha\nbeta"
    assert apply_selected_hunks("", preview, None) == "alpha\nbeta"
    assert apply_selected_hunks("", preview, []) == ""


def test_apply_unparseable_preview_keeps_old_text():
    assert apply_selected_hunks("old", "(no changes)", None) == "old"
    assert apply_selected_hunks("old", "(no changes)", []) == "old"


def test_parse_hunk_selection_forms():
    assert parse_hunk_selection("all", 3) == "all"
    assert parse_hunk_selection("none", 3) == "none"
    assert parse_hunk_selection("1,3", 3) == (1, 3)
    assert parse_hunk_selection("1 3", 3) == (1, 3)
    assert parse_hunk_selection("2", 3) == (2,)
    assert parse_hunk_selection("1-2", 3) == (1, 2)
    assert parse_hunk_selection("3-3,1", 3) == (1, 3)


def test_parse_hunk_selection_rejects_invalid():
    assert parse_hunk_selection("", 3) is None
    assert parse_hunk_selection("banana", 3) is None
    assert parse_hunk_selection("0", 3) is None
    assert parse_hunk_selection("4", 3) is None
    assert parse_hunk_selection("2-1", 3) is None
    assert parse_hunk_selection("1-4", 3) is None
