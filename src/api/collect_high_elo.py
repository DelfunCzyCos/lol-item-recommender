
import json
from pathlib import Path

from riot_api import RiotAPI


OUTPUT_DIR = Path("data/raw/EUNE_HIGH_ELO")

# Na początek ograniczamy pobieranie, żeby nie wykonywać
# tysięcy zapytań API.
TIERS = ["CHALLENGER", "GRANDMASTER", "MASTER"]

MAX_PLAYERS_PER_TIER = 20
MATCHES_PER_PLAYER = 10


def get_existing_match_ids():
    existing = set()

    if not OUTPUT_DIR.exists():
        return existing

    for file_path in OUTPUT_DIR.rglob("*.json"):
        existing.add(file_path.stem)

    return existing


def collect():
    riot = RiotAPI()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_ids = get_existing_match_ids()

    print(f"Już zapisane mecze: {len(existing_ids)}")

    for tier in TIERS:
        print("\n" + "=" * 50)
        print(f"POBIERANIE RANGI: {tier}")
        print("=" * 50)

        try:
            league = riot.get_ranked_league(tier)
            entries = league.get("entries", [])

            entries = sorted(
                entries,
                key=lambda player: player.get("leaguePoints", 0),
                reverse=True
            )

            selected_players = entries[:MAX_PLAYERS_PER_TIER]

            print(f"Liczba graczy wybranych do analizy: {len(selected_players)}")

        except Exception as error:
            print(f"Nie udało się pobrać rankingu {tier}: {error}")
            continue

        for index, player in enumerate(selected_players, start=1):
            puuid = player.get("puuid")

            if not puuid:
                print("Gracz nie ma PUUID - pomijam.")
                continue

            print(
                f"\n[{tier}] Gracz {index}/{len(selected_players)}"
                f" | LP: {player.get('leaguePoints', 0)}"
            )

            try:

                match_ids = riot.get_match_ids(
                    puuid,
                    count=MATCHES_PER_PLAYER,
                    queue=420
                )

                print(f"Znaleziono meczów: {len(match_ids)}")

                for match_id in match_ids:
                    if match_id in existing_ids:
                        continue

                    try:
                        match = riot.get_match(match_id)

                        # Dodatkowe zabezpieczenie: zapisujemy tylko Solo/Duo.
                        if match.get("info", {}).get("queueId") != 420:
                            continue

                        tier_dir = OUTPUT_DIR / tier
                        tier_dir.mkdir(parents=True, exist_ok=True)

                        file_path = tier_dir / f"{match_id}.json"

                        with open(file_path, "w", encoding="utf-8") as file:
                            json.dump(
                                match,
                                file,
                                indent=2,
                                ensure_ascii=False
                            )

                        existing_ids.add(match_id)

                        print(f"Zapisano: {match_id}")

                    except Exception as error:
                        print(f"Błąd meczu {match_id}: {error}")

            except Exception as error:
                print(f"Błąd gracza: {error}")

    print("\n" + "=" * 50)
    print("ZAKOŃCZONO POBIERANIE")
    print(f"Łącznie zapisanych meczów: {len(existing_ids)}")
    print(f"Lokalizacja: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    collect()
