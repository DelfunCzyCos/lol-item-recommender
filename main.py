
import json
import re
from pathlib import Path

from src.api.riot_api import RiotAPI


def main():

    print("=" * 45)
    print("       LOL ITEM RECOMMENDER")
    print("=" * 45)

    riot = RiotAPI()

    # Pobranie Riot ID

    riot_id = input(
        "\nPodaj Riot ID (np. Player#EUW): "
    ).strip()

    if "#" not in riot_id:
        print("Niepoprawny format Riot ID!")
        return

    game_name, tag_line = riot_id.rsplit("#", 1)

    if not game_name or not tag_line:
        print("Niepoprawny Riot ID!")
        return

    try:

        # 1. Pobranie konta

        print("\n[1/4] Wyszukiwanie gracza...")

        account = riot.get_account(
            game_name,
            tag_line
        )

        puuid = account["puuid"]

        print("Znaleziono gracza!")
        print("Game Name:", account["gameName"])
        print("Tag Line:", account["tagLine"])

        # 2. Pobranie historii meczów

        print("\n[2/4] Pobieranie historii meczów...")

        match_ids = riot.get_match_ids(
            puuid,
            count=20,
            queue=420
        )

        print(f"Znaleziono {len(match_ids)} meczów.")

        # 3. Pobranie szczegółów

        print("\n[3/4] Pobieranie szczegółów meczów...")

        matches = riot.get_matches(match_ids)

        # 4. Zapis danych

        print("\n[4/4] Zapisywanie danych...")

        output_dir = Path("data/raw")
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            game_name
        )

        player_dir = output_dir / safe_name
        player_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for match in matches:

            match_id = match["metadata"]["matchId"]

            file_path = player_dir / f"{match_id}.json"

            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    match,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

        print("\nGotowe!")
        print(f"Pobrano {len(matches)} meczów.")
        print(f"Dane zapisano w: {player_dir}")

    except Exception as error:

        print("\nWystąpił błąd:")
        print(error)


if __name__ == "__main__":
    main()
