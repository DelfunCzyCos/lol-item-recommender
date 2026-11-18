
import json
from pathlib import Path
from collections import Counter

TIMELINES_DIR = Path("data/raw/EUNE_HIGH_ELO_TIMELINES")

files = list(TIMELINES_DIR.rglob("*.json"))

print(f"Liczba plików Timeline: {len(files)}")

if not files:
    print("Nie znaleziono plików!")
    exit()

# Wczytujemy jeden przykładowy mecz
with open(files[0], "r", encoding="utf-8") as file:
    timeline = json.load(file)

print("\nGŁÓWNE KLUCZE:")
print(list(timeline.keys()))

print("\nKLUCZE INFO:")
print(list(timeline["info"].keys()))

# Zliczamy rodzaje zdarzeń we wszystkich Timeline
event_counts = Counter()

for index, file_path in enumerate(files, start=1):

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for frame in data["info"]["frames"]:
            for event in frame.get("events", []):
                event_counts[event.get("type", "UNKNOWN")] += 1

    except Exception as error:
        print(f"Błąd pliku {file_path.name}: {error}")

    if index % 500 == 0:
        print(f"Sprawdzono {index}/{len(files)} plików")

print("\nRODZAJE ZDARZEŃ:")
for event_type, count in event_counts.most_common():
    print(f"{event_type}: {count}")

print("\nPRZYKŁADOWE ZDARZENIA ZAKUPU:")

shown = 0

for frame in timeline["info"]["frames"]:
    for event in frame.get("events", []):

        if event.get("type") in [
            "ITEM_PURCHASED",
            "ITEM_SOLD",
            "ITEM_DESTROYED",
            "ITEM_UNDO"
        ]:

            print(event)

            shown += 1

            if shown >= 10:
                break

    if shown >= 10:
        break
