import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder


class SequenceLengthException(ValueError):
    pass


class MapPredictor(nn.Module):
    def __init__(self, max_sequence_length: int, map_encoder: LabelEncoder, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.map_encoder = map_encoder
        num_maps = len(map_encoder.classes_)
        self.dropout = nn.Dropout(0.1)
        self.map_embedding = nn.Embedding(num_maps+1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim,
                          batch_first=True, num_layers=2, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, num_maps+1)

    def forward(self, map_seq):
        map_emb = self.map_embedding(map_seq)
        map_emb = self.dropout(map_emb)
        _, h_n = self.gru(map_emb)
        h_n = h_n[-1]
        logits = self.fc(h_n)
        return logits

    def predict(self, seq: list[str], topk: int) -> tuple[list[str], list[float]]:
        self.eval()

        map_sequence = self.map_encoder.transform(seq)+1
        input_seq = map_sequence.tolist()

        max_len = self.get_config()["max_sequence_length"]
        pad_len = max_len - len(input_seq)
        padded_seq = [0] * pad_len + input_seq

        input_tensor = torch.tensor(
            [padded_seq], dtype=torch.long).to(self.get_device())

        with torch.no_grad():
            logits = self(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_idx = torch.topk(probs, topk).indices
        top_probs = probs[top_idx]
        top_names = self.map_encoder.inverse_transform(top_idx.cpu().numpy()-1)

        return top_names.tolist(), top_probs.cpu().numpy().tolist()

    def get_config(self) -> dict:
        return {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "max_sequence_length": self.max_sequence_length,
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
