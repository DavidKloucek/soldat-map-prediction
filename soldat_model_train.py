import time
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from soldat_model import MapPredictor, MapSequenceDataset, most_common_next, save_model

print("\n\n", "█" * 30, "\n\n")

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device = "cpu"
device = "mps"

target_servers = [
    '=AUS7RAL|12 EuroShots',
    '=AUS7RAL|12 Euro Wars',
    '=AUS7RAL|12 EuroShots #2',
    '[CTF] Oneshot Europe'
]
selected_server = target_servers[2]

df = pd.read_csv("soldat.csv", sep=";")
df = df[["server_name", "map_name", "date"]].dropna()
df = df[df["server_name"] == selected_server].reset_index(drop=True)

df["prev_server"] = df["server_name"].shift(1)
df["prev_map"] = df["map_name"].shift(1)
mask = ~((df["server_name"] == df["prev_server"])
         & (df["map_name"] == df["prev_map"]))
df = df[mask].drop(columns=["prev_server", "prev_map"]).reset_index(drop=True)


sequence_length = 8
num_epochs = 700
batch_size = 256
learning_rate = 0.001
show_prediction_count = 5
embedding_dim = 16
hidden_dim = 32

for server_name, df_server in df.groupby("server_name"):
    print(f"\n=== Training: {server_name} ===")
    print(f"Unique maps: {df_server['map_name'].nunique()}")
    start_time = time.time()

    map_encoder = LabelEncoder()
    df_server["map_id"] = map_encoder.fit_transform(df_server["map_name"])

    if len(df_server["map_id"].unique()) < 2:
        continue

    dataset = MapSequenceDataset(df_server, sequence_length)
    if len(dataset) < 5:
        continue

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MapPredictor(map_encoder=map_encoder,
                         embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Using {model.get_device()}")

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            map_seq = batch["map_seq"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(map_seq)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    end_time = time.time()
    print(f"Training time: {(end_time-start_time):.2f}s")

    save_model(model, "soldat.pt")

    model.eval()

    with torch.no_grad():
        print("\n=============\n")

        most_common = most_common_next(
            df_server["map_name"].to_list(), df_server['map_name'].to_list()[-1], show_prediction_count)

        print(f"Current: {df_server["map_name"].to_list()[-1]}, most common next: {", ".join(
            f"{m} ({c})"
            for m, c in most_common
        )}")

        last_seq = df_server["map_name"].iloc[-sequence_length:].tolist()

        print(f"\nInput: {last_seq}")

        print("\nPredictions:")
        maps, percs = model.predict(seq=last_seq, topk=show_prediction_count)
        for name, perc in zip(maps, percs):
            print(f"{name} | {perc:.2%}")

        print()
