
import json
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/EUNE_HIGH_ELO")
OUTPUT_DIR = Path("data/processed/high_elo")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_matches():

    players_data = []
    items_data = []

    match_files = list(RAW_DIR.rglob("*.json"))

    print(f"Znaleziono meczów: {len(match_files)}")

    for index, file_path in enumerate(match_files, start=1):

        # Ranga znajduje się w nazwie folderu
        tier = file_path.parent.name

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                match = json.load(file)

            info = match["info"]
            metadata = match["metadata"]

            match_id = metadata["matchId"]

            for participant in info["participants"]:

                puuid = participant["puuid"]
                champion = participant["championName"]
                position = participant.get("teamPosition", "UNKNOWN")
                win = participant["win"]

                player_record = {
                    "matchId": match_id,
                    "puuid": puuid,
                    "tier": tier,
                    "champion": champion,
                    "position": position,
                    "win": win,
                    "kills": participant["kills"],
                    "deaths": participant["deaths"],
                    "assists": participant["assists"],
                    "goldEarned": participant["goldEarned"],
                    "totalDamageDealtToChampions": participant["totalDamageDealtToChampions"],
                    "gameDuration": info["gameDuration"]
                }

                players_data.append(player_record)

                # Przedmioty z sześciu głównych slotów
                for slot in range(6):

                    item_id = participant.get(f"item{slot}", 0)

                    if item_id == 0:
                        continue

                    items_data.append({
                        "matchId": match_id,
                        "puuid": puuid,
                        "tier": tier,
                        "champion": champion,
                        "position": position,
                        "win": win,
                        "item_id": item_id
                    })

        except Exception as error:
            print(f"Błąd pliku {file_path.name}: {error}")

        if index % 50 == 0:
            print(f"Przetworzono {index}/{len(match_files)} meczów")

    players_df = pd.DataFrame(players_data)
    items_df = pd.DataFrame(items_data)

    # Usuwamy ewentualne duplikaty
    players_df = players_df.drop_duplicates(
        subset=["matchId", "puuid"]
    )

    items_df = items_df.drop_duplicates(
        subset=["matchId", "puuid", "item_id"]
    )

    players_df.to_csv(
        OUTPUT_DIR / "players_high_elo.csv",
        index=False,
        encoding="utf-8-sig"
    )

    items_df.to_csv(
        OUTPUT_DIR / "items_high_elo.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 45)
    print("PRZETWARZANIE ZAKOŃCZONE")
    print("=" * 45)
    print(f"Mecze: {len(match_files)}")
    print(f"Rekordy graczy: {len(players_df)}")
    print(f"Rekordy przedmiotów: {len(items_df)}")
    print(f"Zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_matches()
