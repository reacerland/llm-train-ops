"""Pure field-extraction and value-normalization for cache -> arrow conversion."""

import json


class MissingFieldError(Exception):
    """Raised when a record lacks a required field."""

    def __init__(self, record_index: int, field: str):
        self.record_index = record_index
        self.field = field
        super().__init__(f"record {record_index}: missing field '{field}'")


def parse_fields(spec: str) -> list[str]:
    """'text, title' -> ['text', 'title']; rejects an empty result."""
    fields = [part.strip() for part in spec.split(",") if part.strip()]
    if not fields:
        raise ValueError("field list is empty")
    return fields


def normalize_value(value):
    """str stays str, None stays None, everything else -> JSON string."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def extract_fields(record: dict, fields: list[str], record_index: int = 0) -> dict:
    """Project a raw record onto the field whitelist, normalizing value types."""
    return {f: normalize_value(_require(record, f, record_index)) for f in fields}


def _require(record: dict, field: str, record_index: int):
    if field not in record:
        raise MissingFieldError(record_index, field)
    return record[field]
