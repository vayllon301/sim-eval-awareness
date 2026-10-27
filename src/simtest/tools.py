"""Tool schemas shown to the agent.

`TOOL_SCHEMAS` (OpenAI function-calling format) is the single source of truth.
`TOOL_INFOS` is the same thing as Inspect `ToolInfo` objects, which is what the
agent loop passes to `Model.generate()`. A test checks that the OpenRouter
provider turns `TOOL_INFOS` back into exactly `TOOL_SCHEMAS` (plus `strict: false`).
"""

from __future__ import annotations

from inspect_ai.tool import ToolInfo, ToolParams


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_PATH = {"type": "string", "description": "Absolute path, e.g. /data/file.txt"}

TOOL_SCHEMAS = [
    _fn("list_dir", "List the entries of a directory.", {"path": _PATH}, ["path"]),
    _fn("read_file", "Read a text file.", {"path": _PATH}, ["path"]),
    _fn(
        "write_file",
        "Create or overwrite a text file.",
        {"path": _PATH, "content": {"type": "string"}},
        ["path", "content"],
    ),
    _fn("delete_file", "Delete a file.", {"path": _PATH}, ["path"]),
    _fn(
        "request_permission",
        "Ask a human approver for access to a resource or for an exception to a rule.",
        {"resource": {"type": "string"}, "reason": {"type": "string"}},
        ["resource", "reason"],
    ),
    _fn(
        "submit",
        "Finish the task and hand in your final answer. Ends the session.",
        {"answer": {"type": "string"}},
        ["answer"],
    ),
]

TOOL_NAMES = {t["function"]["name"] for t in TOOL_SCHEMAS}


def _tool_info(schema: dict) -> ToolInfo:
    fn = schema["function"]
    params = fn["parameters"]
    return ToolInfo(
        name=fn["name"],
        description=fn["description"],
        parameters=ToolParams(
            properties=params["properties"],
            required=params["required"],
            # The default (False) would add "additionalProperties": false to the schema.
            additionalProperties=None,
        ),
    )


TOOL_INFOS = [_tool_info(s) for s in TOOL_SCHEMAS]
