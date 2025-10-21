import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model import MapPredictor, save_model
from helpers import create_model_file_path, most_common_next
from dataset import MapSequenceDatasetDynamic, MapSequenceDatasetFixed
import numpy as np

print("\n\n", "█" * 30, "\n\n")

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = "cpu"
#device = "mps"

target_servers = [
    '=AUS7RAL|12 Euro Wars',
    '=AUS7RAL|12 EuroShots',
    '=AUS7RAL|12 EuroShots #2',
    '[CTF] Oneshot Europe'
]
selected_server = target_servers[3]

max_seq_len = 8
num_epochs = 150
batch_size = 64*4
learning_rate = 0.0005
show_prediction_count = 5
embedding_dim = 16*1
hidden_dim = 64*1

df = pd.read_csv("soldat.csv", sep=";")
df = df.sort_values("date").reset_index(drop=True)
df = df[["server_name", "map_name", "date"]].dropna()
df = df[df["server_name"] == selected_server].reset_index(drop=True)

df["prev_server"] = df["server_name"].shift(1)
df["prev_map"] = df["map_name"].shift(1)
mask = ~((df["server_name"] == df["prev_server"])
         & (df["map_name"] == df["prev_map"]))
df = df[mask].drop(columns=["prev_server", "prev_map"]).reset_index(drop=True)

if 0:
    for m in set(df["map_name"]):
        next = most_common_next(df["map_name"].tolist(), m, 3)
        tokens = [f"{s}  {c}" for s, c in next if c > 5]
        print(f"{m} ... {' ... '.join(tokens)}")

    df_server = df[df["server_name"] == selected_server].reset_index(drop=True)

    map_encoder = LabelEncoder()
    df_server["map_id"] = map_encoder.fit_transform(df_server["map_name"]) + 1

    dataset = MapSequenceDatasetDynamic(
        df=df_server,
        min_seq_len=2,
        max_seq_len=max_seq_len,
        max_gap_minutes=30,
        num_samples=4000
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    i = 0
    for batch in loader:
        i += 1
        ids = batch['map_seq'].reshape(-1).numpy() - 1
        ids = ids[ids >= 0]
        print(map_encoder.inverse_transform(ids))
        print(map_encoder.inverse_transform(
            batch["target"].reshape(-1).numpy() - 1))
        print()
        if i >= 5:
            break

    print(pd.Series(df_server["map_name"]).value_counts().head(50))
    exit()

for server_name, df_server in df.groupby("server_name"):

    server_name = str(server_name)
    print(f"\n=== Training: {server_name} ===")
    print(f"Unique maps: {df_server['map_name'].nunique()}")
    start_time = time.time()

    map_encoder = LabelEncoder()
    df_server["map_id"] = map_encoder.fit_transform(df_server["map_name"]) + 1
    num_classes = int(df_server["map_id"].max()) + 1

    dataset = MapSequenceDatasetDynamic(
        df=df_server,
        min_seq_len=1,
        max_seq_len=max_seq_len,
        max_gap_minutes=30,
        num_samples=4000
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MapPredictor(map_encoder=map_encoder,
                         embedding_dim=embedding_dim,
                         hidden_dim=hidden_dim,
                         max_sequence_length=max_seq_len)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # model = torch.compile(model)

    print(f"Using {model.get_device()}")

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for batch in loader:
            map_seq = batch["map_seq"].to(device)
            target = batch["target"].to(device)
            # target[target == 0] = -1

            seq_ids = map_seq[0].cpu().numpy()
            target_id = target[0].cpu().item()

            seq_ids = [i for i in seq_ids if i != 0]

            seq_names = map_encoder.inverse_transform(np.array(seq_ids) - 1)
            target_name = map_encoder.inverse_transform([target_id - 1])[0]

            if 0:
                print(f"{', '.join(seq_names)} >>> {target_name}")
                # exit()

            assert target.min() > 0
            assert target.max() <= num_classes
            assert target.max(
            ) < num_classes, f"Target index {target.max().item()} exceeds num_classes={num_classes}"

            optimizer.zero_grad()
            logits = model(map_seq)

            if 0:
                print("num_classes:", num_classes)
                print("logits shape:", logits.shape)
                print("target min/max:", target.min().item(), target.max().item())

            loss = criterion(logits, target-0)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1) + 1
            correct = (pred == target).float().sum().item()
            total_correct += correct
            total_samples += target.size(0)

        acc = total_correct / total_samples

        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1).mean().item()

        if (epoch + 1) % 25 == 0 or 1:
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Loss: {(total_loss / len(loader)):.4f}, "
                  f"Acc: {acc:.2%}, "
                  f"Entropy: {entropy:.2f}")

    end_time = time.time()
    print(f"Training time: {(end_time-start_time):.2f}s")

    save_model(model, create_model_file_path(server_name))

    model.eval()

    with torch.no_grad():
        print("\n=============\n")

        most_common = most_common_next(
            df_server["map_name"].to_list(), df_server['map_name'].to_list()[-1], show_prediction_count)

        print(f"Current: {df_server['map_name'].to_list()[-1]}, most common next:", ", ".join(
            f"{m} ({c})"
            for m, c in most_common
        ))

        last_seq = df_server["map_name"].iloc[-max_seq_len:].tolist()

        print(f"\nInput: {last_seq}")

        print("\nPredictions:")
        maps, percs = model.predict(seq=last_seq, topk=show_prediction_count)
        for name, perc in zip(maps, percs):
            print(f"{name} | {perc:.2%}")

        print()
