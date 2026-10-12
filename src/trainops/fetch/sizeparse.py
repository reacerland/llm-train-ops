"""Parse human-readable sizes like '500MB' into bytes (1024-based)."""

import re

_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
_PATTERN = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)\s*", re.IGNORECASE)


def parse_size(text: str) -> int:
    """Parse '500MB' / '2GB' / '1024B' into a byte count. Raises ValueError on bad input."""
    match = _PATTERN.fullmatch(text)
    if not match:
        raise ValueError(f"invalid size: {text!r} (expected e.g. '500MB', '2GB')")
    value, unit = match.groups()
    return int(float(value) * _UNITS[unit.upper()])
