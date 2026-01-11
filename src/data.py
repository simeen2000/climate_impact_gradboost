from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import kagglehub
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    kaggle_id: str = "uom190346a/global-climate-events-and-economic-impact-dataset"


def download_dataset(spec: DatasetSpec) -> Path:
    path = kagglehub.dataset_download(spec.kaggle_id)
    return Path(path)


def load_first_csv(dataset_dir: Path) -> pd.DataFrame:
    csvs = sorted([p for p in dataset_dir.iterdir() if p.suffix.lower() == ".csv"])
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in: {dataset_dir}")
    return pd.read_csv(csvs[0])


def load_dataset(spec: Optional[DatasetSpec] = None) -> pd.DataFrame:
    spec = spec or DatasetSpec()
    d = download_dataset(spec)
    return load_first_csv(d)
