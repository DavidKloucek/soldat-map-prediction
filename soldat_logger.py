import os
import datetime
import json
import csv
from time import sleep
from pync import Notifier
from rich import print
import requests

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
    "ctf_Run"
]
watched_servers: list[str] = [
    '=AUS7RAL|12 EuroShots',
    '=AUS7RAL|12 Euro Wars',
    '=AUS7RAL|12 EuroShots #2',
    '[CTF] Oneshot Europe'
]

csv_path = os.path.join(os.path.dirname(__file__), "soldat.csv")

print(f"CSV path: {csv_path}")

while True:
    req = requests.get("https://api.soldat.pl/v0/servers?version=1.7.1")
    data = json.loads(req.content)
    players_total = 0

    for item in data['Servers']:
        players = int(item['NumPlayers'])
        name = str(item['Name'])
        map_ = str(item['CurrentMap'])
        ip = str(item['IP'])
        port = str(item['Port'])

        players_total += players

        is_watched_server = any(x.lower() in name.lower()
                                for x in watched_servers)

        if is_watched_server:
            with open(csv_path, mode="a+", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                                        'date', "ip", "port", 'server_name', 'map_name', 'players'], delimiter=";")
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow({
                    'date': str(datetime.datetime.now(datetime.timezone.utc)),
                    "ip": ip,
                    "port": port,
                    'server_name': name,
                    'map_name': map_,
                    'players': players
                })

        if (
            players >= 5
            and any(x.lower() in map_.lower() for x in watched_maps)
            and is_watched_server
        ):
            print(
                f"{name} -> {map_} | {players} | {item['IP']}:{item['Port']}")

            Notifier.notify(
                "",
                title=f"{map_} | {name}",
                subtitle=f"Hráčů: {players}/{item['MaxPlayers']}"
            )

    print(
        f"Refresh {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, found {len(data['Servers'])} servers, {players_total} players")

    sleep(60)
