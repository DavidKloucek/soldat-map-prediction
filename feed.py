from collections import deque
import os
import datetime
import json
import csv
from time import sleep
from pync import Notifier
from rich import print
import requests
from model import MapPredictor, create_model_file_path, load_model

print("\n\n", "█" * 30, "\n\n")

watched_maps: list[str] = [
    'ctf_Kampf',
    'ctf_Lanubya',
    'ctf_Snakebite',
    "ctf_Maya",
    "ctf_Laos",
    "ctf_Equinox",
    "ctf_Voland",
    "ctf_Viet",
    "ctf_Raspberry",
    "ctf_Run",
    # "ctf_Ash",
]
watched_servers: list[str] = [
    '=AUS7RAL|12 EuroShots',
    '=AUS7RAL|12 Euro Wars',
    '=AUS7RAL|12 EuroShots #2',
    '[CTF] Oneshot Europe'
]
log_changes_enabled = False
min_players = 0
sequence_length = 2


map_history: dict = {}


def add_map(server_name: str, map_name: str, maxlen: int):
    q = map_history.setdefault(server_name, deque(maxlen=maxlen))
    if q and q[-1] == map_name:
        return
    q.append(map_name)


def get_history(server_name: str):
    return list(map_history.get(server_name, []))


models: dict[str, MapPredictor] = {}


def get_model(server_name: str) -> MapPredictor:
    fn = create_model_file_path(server_name)
    if fn in models:
        return models[fn]
    models[fn] = load_model(path=fn, device="cpu")
    return models[fn]


def is_watched_map(name: str):
    return any(m.lower() in name.lower() for m in watched_maps)


csv_path = os.path.join(os.path.dirname(__file__), "soldat.csv")

print(f"CSV path: {csv_path}")

while True:
    req = requests.get("https://api.soldat.pl/v0/servers?version=1.7.1")
    data = json.loads(req.content)
    players_total = 0

    for item in data['Servers']:
        players = int(item['NumPlayers'])
        server_name = str(item['Name'])
        map_name = str(item['CurrentMap'])
        ip = str(item['IP'])
        port = str(item['Port'])

        players_total += players

        is_watched_server = any(x.lower() in server_name.lower()
                                for x in watched_servers)

        if is_watched_server and log_changes_enabled:
            with open(csv_path, mode="a+", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                                        'date', "ip", "port", 'server_name', 'map_name', 'players'], delimiter=";")
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow({
                    'date': str(datetime.datetime.now(datetime.timezone.utc)),
                    "ip": ip,
                    "port": port,
                    'server_name': server_name,
                    'map_name': map_name,
                    'players': players
                })

        if is_watched_server:
            add_map(server_name, map_name, sequence_length)

        if players >= min_players and is_watched_server:
            print(f"{server_name} | {players} | {item['IP']}:{item['Port']}")

            map_text = []

            if is_watched_map(map_name):
                map_text.append(f"Current: {map_name}")

            model = get_model(server_name)
            if model and len(get_history(server_name)) == sequence_length:
                print("get_history(server_name),", get_history(server_name),)
                pred_map, pred_perc = model.predict(
                    get_history(server_name), 2)
                print(pred_map, pred_perc)
                if len(pred_perc) > 0 and pred_perc[0] >= 0.7 and is_watched_map(pred_map[0]):
                    map_text.append(
                        f"Next: {pred_map[0]} ({round(pred_perc[0]*100)}%)")

            if len(map_text) > 0:
                print("********************")
                print("\n".join(map_text))
                Notifier.notify(
                    "\n".join(map_text),
                    title=f"{server_name}",
                    subtitle=f"Players: {players} / {item['MaxPlayers']}"
                )

    print(
        f"Refresh {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, found {len(data['Servers'])} servers, {players_total} players")

    sleep(60)
