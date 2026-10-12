import pytest

from trainops.fetch.sizeparse import parse_size


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1024B", 1024),
        ("512KB", 512 * 1024),
        ("500MB", 500 * 1024**2),
        ("2GB", 2 * 1024**3),
        ("1.5GB", int(1.5 * 1024**3)),
        ("500mb", 500 * 1024**2),
        (" 2 GB ", 2 * 1024**3),
    ],
)
def test_parse_size_valid(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "500", "500TB", "-1GB", "MB"])
def test_parse_size_invalid(text):
    with pytest.raises(ValueError):
        parse_size(text)
