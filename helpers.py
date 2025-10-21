from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
import hashlib
from typing import Iterable


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


def create_model_file_path(server_name: str) -> str:
    return str(Path("models") / Path(f"soldat_{hashlib.md5(server_name.encode()).hexdigest()}.pt"))


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
