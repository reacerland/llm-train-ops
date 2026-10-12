import pytest

from trainops.convert.transform import (
    MissingFieldError,
    extract_fields,
    normalize_value,
    parse_fields,
)


def test_parse_fields_basic():
    assert parse_fields("text, title") == ["text", "title"]


def test_parse_fields_strips_and_drops_empties():
    assert parse_fields(" text ,, ") == ["text"]


def test_parse_fields_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        parse_fields(" , ,")


def test_normalize_value_string_passthrough():
    assert normalize_value("hello") == "hello"


def test_normalize_value_none_passthrough():
    assert normalize_value(None) is None


def test_normalize_value_number_becomes_json_string():
    assert normalize_value(42) == "42"
    assert normalize_value(3.5) == "3.5"


def test_normalize_value_bool_list_dict_become_json_strings():
    assert normalize_value(True) == "true"
    assert normalize_value([1, "二"]) == '[1, "二"]'
    assert normalize_value({"a": 1}) == '{"a": 1}'


def test_normalize_value_unicode_not_escaped():
    assert normalize_value({"t": "中文"}) == '{"t": "中文"}'


def test_extract_fields_projects_and_normalizes():
    record = {"text": "hi", "n": 3, "extra": "x"}
    assert extract_fields(record, ["text", "n"], record_index=0) == {"text": "hi", "n": "3"}


def test_extract_fields_missing_raises_with_index_and_field():
    with pytest.raises(MissingFieldError) as exc_info:
        extract_fields({"a": 1}, ["a", "b"], record_index=7)
    assert exc_info.value.record_index == 7
    assert exc_info.value.field == "b"
    assert "record 7" in str(exc_info.value)
    assert "'b'" in str(exc_info.value)
