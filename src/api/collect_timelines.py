
import json
from pathlib import Path

from riot_api import RiotAPI


# Folder z dotychczas pobranymi meczami
MATCHES_DIR = Path("data/raw/EUNE_HIGH_ELO")

# Osobny folder na Timeline
TIMELINES_DIR = Path("data/raw/EUNE_HIGH_ELO_TIMELINES")


def get_existing_timeline_ids():
    if not TIMELINES_DIR.exists():
        return set()

    return {
        file_path.stem
        for file_path in TIMELINES_DIR.rglob("*.json")
    }


def collect_timelines():

    riot = RiotAPI()

    TIMELINES_DIR.mkdir(parents=True, exist_ok=True)

    existing_ids = get_existing_timeline_ids()

    match_files = list(MATCHES_DIR.rglob("*.json"))

    print(f"Liczba znalezionych meczów: {len(match_files)}")
    print(f"Timeline już pobrane: {len(existing_ids)}")

    for index, match_file in enumerate(match_files, start=1):

        match_id = match_file.stem
        tier = match_file.parent.name

        if match_id in existing_ids:
            continue

        print(
            f"\n[{index}/{len(match_files)}] "
            f"Pobieranie Timeline: {match_id}"
        )

        try:
            timeline = riot.get_match_timeline(match_id)

            output_dir = TIMELINES_DIR / tier
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{match_id}.json"

            with open(output_file, "w", encoding="utf-8") as file:
                json.dump(
                    timeline,
                    file,
                    indent=2,
                    ensure_ascii=False
                )

            existing_ids.add(match_id)

            print(f"Zapisano: {output_file}")

        except Exception as error:
            print(f"Błąd Timeline {match_id}: {error}")

    print("\n" + "=" * 55)
    print("ZAKOŃCZONO POBIERANIE TIMELINE")
    print(f"Pobrane Timeline: {len(existing_ids)}")
    print(f"Lokalizacja: {TIMELINES_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    collect_timelines()
