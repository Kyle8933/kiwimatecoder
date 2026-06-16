from kiwimatecoder.client import (
    Done,
    TextDelta,
    ToolCallAssembler,
    ToolCallDelta,
    Usage,
    parse_sse_chunk,
)


def test_parse_text_delta():
    events = parse_sse_chunk('{"choices":[{"delta":{"content":"hi"}}]}')
    assert events == [TextDelta(text="hi")]


def test_parse_done_sentinel():
    events = parse_sse_chunk("[DONE]")
    assert events == [Done(finish_reason=None)]


def test_parse_finish_reason():
    events = parse_sse_chunk('{"choices":[{"delta":{},"finish_reason":"stop"}]}')
    assert Done(finish_reason="stop") in events


def test_parse_usage():
    events = parse_sse_chunk('{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":7}}')
    assert Usage(prompt_tokens=5, completion_tokens=7) in events


def test_parse_malformed_json_yields_nothing():
    assert parse_sse_chunk("{not json") == []


def test_tool_call_assembler_reassembles_fragments():
    asm = ToolCallAssembler()
    asm.add(ToolCallDelta(index=0, id="call_1", name="read_file", args_fragment='{"pa'))
    asm.add(ToolCallDelta(index=0, args_fragment='th":"a.txt"}'))
    calls = asm.finalize()
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "read_file"
    assert calls[0].parse_arguments() == {"path": "a.txt"}


def test_tool_call_assembler_multiple_calls_ordered():
    asm = ToolCallAssembler()
    asm.add(ToolCallDelta(index=0, id="a", name="read_file", args_fragment="{}"))
    asm.add(ToolCallDelta(index=1, id="b", name="list_dir", args_fragment="{}"))
    calls = asm.finalize()
    assert [c.name for c in calls] == ["read_file", "list_dir"]


def test_assembler_synthesizes_missing_id():
    asm = ToolCallAssembler()
    asm.add(ToolCallDelta(index=2, name="search", args_fragment="{}"))
    calls = asm.finalize()
    assert calls[0].id == "call_2"


def test_parse_tool_call_delta():
    chunk = (
        '{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        '"function":{"name":"read_file","arguments":"{}"}}]}}]}'
    )
    events = parse_sse_chunk(chunk)
    assert any(isinstance(e, ToolCallDelta) and e.name == "read_file" for e in events)
