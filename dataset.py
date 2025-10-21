from datetime import datetime, timedelta
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
from helpers import get_time_gap


class MapSequenceDatasetDynamic(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        max_seq_len: int,
        min_seq_len: int,
        num_samples: int,
        max_gap_minutes: int
    ):
        self.df = df.reset_index(drop=True)
        self.map_ids = df["map_id"].tolist()
        self.dates = [datetime.fromisoformat(x) for x in df["date"].tolist()]
        self.max_len = max_seq_len
        self.min_len = min_seq_len
        self.num_samples = num_samples
        self.max_gap = timedelta(minutes=max_gap_minutes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        L = np.random.randint(self.min_len, self.max_len + 1)
        i = np.random.randint(L, len(self.map_ids))  # už ne -1

        seq = self.map_ids[i - L:i]
        dates = self.dates[i - L:i]

        if 0:
            is_gap, _ = get_time_gap(dates, self.max_gap)
            if is_gap:
                seq = self.map_ids[-L - 1:-1]
                target = self.map_ids[-1]
            else:
                target = self.map_ids[i]

        target = self.map_ids[i]

        pad_len = self.max_len - len(seq)
        padded_seq = [0] * pad_len + seq

        return {
            "map_seq": torch.tensor(padded_seq, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long)
        }


class MapSequenceDatasetFixed(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        max_seq_len: int,
        min_seq_len: int,
        num_samples: int,
        max_gap_minutes: int
    ):
        self.df = df.reset_index(drop=True)
        self.map_ids = df["map_id"].tolist()
        self.dates = [datetime.fromisoformat(x) for x in df["date"].tolist()]
        self.max_len = max_seq_len
        self.min_len = min_seq_len
        self.num_samples = num_samples
        self.max_gap = timedelta(minutes=max_gap_minutes)

        self.samples = []
        for _ in range(self.num_samples):
            L = np.random.randint(self.min_len, self.max_len + 1)
            i = np.random.randint(L, len(self.map_ids))

            seq = self.map_ids[i - L:i]
            dates = self.dates[i - L:i]

            # is_gap, _ = get_time_gap(dates, self.max_gap)
            # if is_gap:
            #     continue

            target = self.map_ids[i]
            pad_len = self.max_len - len(seq)
            padded_seq = [0] * pad_len + seq

            self.samples.append((padded_seq, target))

        print(f"Generated {len(self.samples)} fixed samples "
              f"(seq len {self.min_len}–{self.max_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return {
            "map_seq": torch.tensor(seq, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long)
        }


class MapSequenceDatasetOld(Dataset):
    def __init__(self, df: pd.DataFrame, max_sequence_length: int, min_sequence_length: int):
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.map_ids = df["map_id"].tolist()
        self.samples = []

        skipped_count = 0
        for L in range(min_sequence_length, max_sequence_length + 1):
            for i in range(L, len(df)):
                seq = self.map_ids[i - L:i]
                dates = [
                    datetime.fromisoformat(x)
                    for x in df['date'][i - L:i].tolist()
                ]
                is_gap, gap_list = get_time_gap(dates, timedelta(minutes=30))
                # if is_gap:
                #    skipped_count += len(gap_list)
                #    continue
                target = self.map_ids[i]
                self.samples.append((seq, target))

        print(f"Skipped sequences due to time gaps: {skipped_count}")
        print(f"Total usable sequences: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        pad_len = self.max_sequence_length - len(seq)
        padded_seq = [0] * pad_len + seq
        return {
            "map_seq": torch.tensor(padded_seq, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long)
        }
