"""Turn arbitrary SDK payloads into text the sanitizer can process.

JSON output is preferred: the sanitizer walks JSON leaf-by-leaf, so
structured prompts/messages survive PHI replacement without corruption.
"""

from __future__ import annotations

import json
from typing import Any


def to_text(value: Any) -> str:
    """Serialize a payload to a stable string (JSON when possible)."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_fallback)
    except (TypeError, ValueError):
        return repr(value)


def _fallback(value: Any) -> Any:
    # Pydantic models (OpenAI / Anthropic SDK types) expose model_dump.
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001 — never let serialization break capture
            pass
    return repr(value)
