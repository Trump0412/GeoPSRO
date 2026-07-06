from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from geopsro4d.data.schema import GeoSample, normalize_sample
from geopsro4d.utils.io import iter_jsonl


class RFTDataset(Dataset):
    def __init__(self, samples: list[GeoSample]) -> None:
        self.samples = samples

    @classmethod
    def from_jsonl(cls, path: str | Path, *, dataset_name: str | None = None) -> "RFTDataset":
        samples = [normalize_sample(row, default_dataset=dataset_name or Path(path).stem) for row in iter_jsonl(path)]
        return cls(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> GeoSample:
        return self.samples[index]
