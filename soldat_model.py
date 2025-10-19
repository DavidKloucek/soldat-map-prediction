from collections import Counter
from datetime import datetime, timedelta
from typing import Iterable
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def most_common_next(words: list[str], target: str, n: int) -> list[tuple[str, int]]:
    next_words = [
        words[i + 1]
        for i in range(len(words) - 1)
        if words[i] == target
    ]
    counter = Counter(next_words)
    return counter.most_common(n)


def format_td(td: timedelta):
    total_seconds = int(td.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"


def get_time_gap(dates: Iterable[datetime], ts: timedelta) -> tuple[bool, list[timedelta]]:
    last = None
    r = False
    wrong_list = []
    for item in dates:
        if last is not None:
            diff = item - last
            if diff > ts:
                r = True
                wrong_list.append(diff)
        last = item
    return r, wrong_list


class MapSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sequence_length: int):
        self.sequence_length = sequence_length
        self.map_ids = df["map_id"].tolist()
        self.samples = []

        skipped_count = 0
        for i in range(sequence_length, len(df)):
            seq = self.map_ids[i-sequence_length:i]
            dates = [datetime.fromisoformat(
                x) for x in df['date'][i-sequence_length:i].tolist()]
            is_gap, gap_list = get_time_gap(dates, timedelta(minutes=30))
            if is_gap:
                for g in gap_list:
                    skipped_count += 1
                    # print(format_td(g))
                continue
            target = self.map_ids[i]
            self.samples.append((seq, target))
        print(f"Skipped sequences: {skipped_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return {
            "map_seq": torch.tensor(seq, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long)
        }


class MapPredictor(nn.Module):
    def __init__(self, map_encoder: LabelEncoder, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.map_encoder = map_encoder
        num_maps = len(map_encoder.classes_)
        self.map_embedding = nn.Embedding(num_maps, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim,
                          batch_first=True, num_layers=1, dropout=0.0)
        self.fc = nn.Linear(hidden_dim, num_maps)

    def forward(self, map_seq):
        map_emb = self.map_embedding(map_seq)
        _, h_n = self.gru(map_emb)
        h_n = h_n[-1]
        logits = self.fc(h_n)
        return logits

    def predict(self, seq: list[str], topk: int) -> tuple[list[str], list[float]]:
        self.eval()
        map_sequence = self.map_encoder.transform(seq)
        sequence_length = len(map_sequence)

        input_seq = map_sequence[-sequence_length:]
        input_tensor = torch.tensor(
            np.array([input_seq]), dtype=torch.long).to(self.get_device())

        with torch.no_grad():
            logits = self(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_idx = torch.topk(probs, topk).indices
        top_probs = probs[top_idx]
        top_names = self.map_encoder.inverse_transform(top_idx.cpu().numpy())
        return top_names.tolist(), top_probs.cpu().numpy().tolist()

    def get_config(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
        }

    def get_device(self):
        return next(self.parameters()).device


def save_model(model: MapPredictor, path: str) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "map_classes": model.map_encoder.classes_.tolist(),
        "config": model.get_config(),
    }, path)


def load_model(path: str, device: str | torch.device) -> MapPredictor:
    data = torch.load(path, map_location=device)

    le = LabelEncoder()
    le.classes_ = np.array(data["map_classes"], dtype=object)

    model = MapPredictor(map_encoder=le, **data['config'])
    model.load_state_dict(data["state_dict"])
    model.to(device)
    return model
