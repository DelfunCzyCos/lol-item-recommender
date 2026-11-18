
import json
from pathlib import Path
from collections import Counter

MATCHES_DIR = Path("data/raw/EUNE_HIGH_ELO")
TIMELINES_DIR = Path("data/raw/EUNE_HIGH_ELO_TIMELINES")

# Wybieramy pierwszy Timeline
timeline_files = list(TIMELINES_DIR.rglob("*.json"))

if not timeline_files:
    raise FileNotFoundError("Brak plików Timeline!")

timeline_file = timeline_files[0]
match_id = timeline_file.stem
tier = timeline_file.parent.name

# Odpowiadający mu plik meczu
match_file = MATCHES_DIR / tier / f"{match_id}.json"

with open(timeline_file, "r", encoding="utf-8") as file:
    timeline = json.load(file)

with open(match_file, "r", encoding="utf-8") as file:
    match = json.load(file)

print("=" * 60)
print("PRZYKŁADOWY MECZ:", match_id)
print("=" * 60)

print("\nUCZESTNICY TIMELINE:")

for participant in timeline["info"]["participants"]:
    print(participant)

print("\nUCZESTNICY MATCH:")

for participant in match["info"]["participants"]:
    print({
        "participantId": participant.get("participantId"),
        "puuid": participant.get("puuid"),
        "champion": participant.get("championName"),
        "position": participant.get("teamPosition"),
        "win": participant.get("win")
    })

# Liczymy identyfikatory uczestników w zdarzeniach przedmiotów
participant_counts = Counter()

item_events = {
    "ITEM_PURCHASED",
    "ITEM_SOLD",
    "ITEM_DESTROYED",
    "ITEM_UNDO"
}

for frame in timeline["info"]["frames"]:
    for event in frame.get("events", []):
        if event.get("type") in item_events:
            participant_counts[event.get("participantId")] += 1

print("\nIDENTYFIKATORY W ZDARZENIACH PRZEDMIOTÓW:")

for participant_id, count in sorted(
    participant_counts.items(),
    key=lambda x: (-1 if x[0] is None else x[0])
):
    print(f"Participant ID {participant_id}: {count}")

print("\nPRZYKŁADOWE ITEM_UNDO:")

shown = 0

for frame in timeline["info"]["frames"]:
    for event in frame.get("events", []):
        if event.get("type") == "ITEM_UNDO":
            print(event)
            shown += 1

            if shown >= 5:
                break

    if shown >= 5:
        break

print("\nPRZYKŁADOWE ITEM_DESTROYED:")

shown = 0

for frame in timeline["info"]["frames"]:
    for event in frame.get("events", []):
        if event.get("type") == "ITEM_DESTROYED":
            print(event)
            shown += 1

            if shown >= 5:
                break

    if shown >= 5:
        break
