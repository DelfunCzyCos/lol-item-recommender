
import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def process_matches():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    player_rows = []
    item_rows = []

    files = list(RAW_DIR.rglob("*.json"))

    if not files:
        print("Nie znaleziono plików JSON w data/raw!")
        return

    print(f"Znaleziono {len(files)} plików z meczami.")

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                match = json.load(file)

            match_id = match["metadata"]["matchId"]
            info = match["info"]

            game_duration = info.get("gameDuration", 0)
            queue_id = info.get("queueId")

            for participant in info["participants"]:
                common_data = {
                    "matchId": match_id,
                    "puuid": participant.get("puuid"),
                    "champion": participant.get("championName"),
                    "position": participant.get("teamPosition"),
                    "win": participant.get("win"),
                    "gameDuration": game_duration,
                    "queueId": queue_id,
                    "kills": participant.get("kills"),
                    "deaths": participant.get("deaths"),
                    "assists": participant.get("assists"),
                    "goldEarned": participant.get("goldEarned"),
                    "visionScore": participant.get("visionScore"),
                }

                player_rows.append(common_data)

                # Przedmioty z głównych sześciu slotów.
                # Slot 6 to trinket, więc na razie go pomijamy.
                for slot in range(6):
                    item_id = participant.get(f"item{slot}", 0)

                    if item_id != 0:
                        item_rows.append({
                            **common_data,
                            "itemId": item_id,
                            "slot": slot,
                        })

        except (KeyError, ValueError, json.JSONDecodeError) as error:
            print(f"Problem z plikiem {file_path}: {error}")

    players_df = pd.DataFrame(player_rows)
    items_df = pd.DataFrame(item_rows)

    players_df.to_csv(
        OUTPUT_DIR / "players.csv",
        index=False,
        encoding="utf-8-sig"
    )

    items_df.to_csv(
        OUTPUT_DIR / "items.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPrzetwarzanie zakończone!")
    print(f"Liczba rekordów graczy: {len(players_df)}")
    print(f"Liczba rekordów przedmiotów: {len(items_df)}")
    print(f"Zapisano pliki w: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_matches()
