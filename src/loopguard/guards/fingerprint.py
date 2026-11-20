"""Fingerprints for tool calls.

A loop is "the same thing happening again". Raw strings are a poor test for that because
real tool outputs carry noise: request ids, timestamps, and echoes of the query. Normalising
that noise away is what lets the detector see a polling loop or a retry storm as a repeat.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_ISO_TS = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_LONG_HEX = re.compile(r"\b(?:0x)?[0-9a-f]{20,}\b", re.I)
_WS = re.compile(r"\s+")

PREVIEW_CHARS = 160


def canonical_args(args: dict[str, Any] | None) -> str:
    """Stable JSON for tool arguments: key order and whitespace never change the hash."""
    return json.dumps(args or {}, sort_keys=True, separators=(",", ":"), default=str)


def _is_free_text(value: Any) -> bool:
    """Free text (two or more words) is masked when echoed back; identifiers are not.

    ``"parental leave policy"`` is echoed by a search tool in its "no results" message, so
    masking it lets two different searches with the same empty result look identical.
    ``"job-1001"`` is an identifier; masking it would make three healthy status checks on
    three different jobs look like a loop.
    """
    return isinstance(value, str) and len(value.split()) >= 2


def normalize_output(output: str, args: dict[str, Any] | None = None) -> str:
    text = str(output)
    for key, value in sorted((args or {}).items()):
        if _is_free_text(value):
            text = re.sub(re.escape(value.strip()), f"<arg:{key}>", text, flags=re.I)
    text = _UUID.sub("<uuid>", text)
    text = _ISO_TS.sub("<ts>", text)
    text = _LONG_HEX.sub("<hex>", text)
    return _WS.sub(" ", text).strip().lower()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ToolRecord:
    """One executed tool call, as the guard sees it."""

    step: int
    tool: str
    args_fp: str  # tool name + canonical args
    output_fp: str  # normalised output
    call_fp: str  # input and output together
    args_preview: str
    output_preview: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolRecord:
        return cls(**data)


def make_record(
    step: int, tool: str, args: dict[str, Any] | None, output: str, is_error: bool = False
) -> ToolRecord:
    args_fp = digest(f"{tool}|{canonical_args(args)}")
    output_fp = digest(normalize_output(output, args))
    return ToolRecord(
        step=step,
        tool=tool,
        args_fp=args_fp,
        output_fp=output_fp,
        call_fp=digest(f"{args_fp}|{output_fp}"),
        args_preview=canonical_args(args)[:PREVIEW_CHARS],
        output_preview=str(output)[:PREVIEW_CHARS],
        is_error=is_error,
    )
