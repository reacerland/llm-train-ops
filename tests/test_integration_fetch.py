import json

import pytest

from trainops.fetch.command import run_fetch

pytestmark = pytest.mark.slow

DATASET = "swift/self-cognition"  # small script-free dataset on ModelScope
TARGET = 10 * 1024  # 10KB (dataset total is ~23KB serialized)


def test_fetch_real_dataset(tmp_path):
    result = run_fetch(DATASET, None, "train", TARGET, tmp_path)
    assert result.bytes >= TARGET or result.dataset_exhausted
    data_file = tmp_path / "datasets" / "swift__self-cognition" / "default" / "train" / "data.jsonl"
    assert data_file.exists()
    first = json.loads(data_file.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(first, dict)

    # Same target again must be a cache hit (no download).
    again = run_fetch(DATASET, None, "train", TARGET, tmp_path)
    assert again.downloaded is False
    assert again.bytes == result.bytes
