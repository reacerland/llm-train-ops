"""Thin wrapper over the ModelScope SDK streaming API."""

from collections.abc import Iterator
from typing import Any


def stream_dataset(dataset: str, subset: str, split: str) -> Iterator[dict[str, Any]]:
    """Yield raw records from a ModelScope dataset in streaming mode.

    Raises (from the SDK) when the dataset does not exist, auth fails, or the
    dataset does not support streaming.
    """
    from modelscope.msdatasets import MsDataset  # lazy: heavy import

    ds = MsDataset.load(dataset, subset_name=subset, split=split, streaming=True)
    for row in ds:
        yield dict(row)
