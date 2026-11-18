
import json
from pathlib import Path

import pandas as pd


MATCHES_DIR = Path("data/raw/EUNE_HIGH_ELO")
TIMELINES_DIR = Path("data/raw/EUNE_HIGH_ELO_TIMELINES")
CATALOG_FILE = Path("data/processed/items_catalog.csv")

OUTPUT_DIR = Path("data/processed/high_elo")

PURCHASE_EVENTS_FILE = OUTPUT_DIR / "item_purchase_events.csv"


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def process_match(match_file, catalog):

    match_id = match_file.stem
    tier = match_file.parent.name

    timeline_file = TIMELINES_DIR / tier / f"{match_id}.json"

    if not timeline_file.exists():
        return []

    match = load_json(match_file)
    timeline = load_json(timeline_file)

    # Identyfikatory przedmiotów znajdujących się w katalogu
    catalog_ids = set(catalog["item_id"].astype(int))

    # Informacje o uczestnikach meczu
    participants = {}

    for player in match["info"]["participants"]:

        participant_id = player.get("participantId")

        participants[participant_id] = {
            "puuid": player.get("puuid"),
            "champion": player.get("championName"),
            "position": player.get("teamPosition"),
            "win": player.get("win")
        }

    # Zbieramy zakupy i cofnięcia zakupów
    events = []

    for frame in timeline["info"]["frames"]:

        for event in frame.get("events", []):

            if event.get("type") not in [
                "ITEM_PURCHASED",
                "ITEM_UNDO"
            ]:
                continue

            participant_id = event.get("participantId")

            if participant_id not in participants:
                continue

            events.append(event)

    # Sortowanie chronologiczne
    events.sort(key=lambda event: event.get("timestamp", 0))

    purchases = []

    for event in events:

        participant_id = event["participantId"]
        event_type = event["type"]

        if event_type == "ITEM_PURCHASED":

            item_id = event.get("itemId")

            if item_id is None:
                continue

            item_id = int(item_id)

            # Interesują nas tylko przedmioty z naszego katalogu
            if item_id not in catalog_ids:
                continue

            purchases.append({
                "match_id": match_id,
                "tier": tier,
                "participant_id": participant_id,
                "puuid": participants[participant_id]["puuid"],
                "champion": participants[participant_id]["champion"],
                "position": participants[participant_id]["position"],
                "win": participants[participant_id]["win"],
                "item_id": item_id,
                "timestamp": event.get("timestamp", 0),
                "undone": False
            })

        elif event_type == "ITEM_UNDO":

            before_id = int(event.get("beforeId") or 0)
            after_id = int(event.get("afterId") or 0)

            # beforeId != 0 i afterId == 0 oznacza cofnięcie zakupu.
            # Przy cofnięciu sprzedaży przedmiot jest przywracany
            # (afterId != 0), więc nie usuwamy historycznego zakupu.
            if before_id == 0 or after_id != 0:
                continue

            # Cofamy ostatni aktywny zakup tego przedmiotu
            # przez tego samego gracza.
            for purchase in reversed(purchases):

                if (
                    purchase["participant_id"] == participant_id
                    and purchase["item_id"] == before_id
                    and not purchase["undone"]
                ):

                    purchase["undone"] = True
                    break

    # Usuwamy wyłącznie zakupy rzeczywiście cofnięte
    purchases = [
        purchase
        for purchase in purchases
        if not purchase["undone"]
    ]

    # Numerujemy zachowane zakupy osobno dla każdego gracza
    purchase_numbers = {}

    for purchase in purchases:

        key = (
            purchase["match_id"],
            purchase["participant_id"]
        )

        purchase_numbers[key] = purchase_numbers.get(key, 0) + 1

        purchase["purchase_number"] = purchase_numbers[key]

        purchase["time_minutes"] = round(
            purchase["timestamp"] / 60000,
            2
        )

    return purchases


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu: {CATALOG_FILE}"
        )

    catalog = pd.read_csv(CATALOG_FILE)

    # Ujednolicamy nazwy kolumn
    if "itemId" in catalog.columns:
        catalog = catalog.rename(
            columns={"itemId": "item_id"}
        )

    catalog["item_id"] = catalog["item_id"].astype(int)

    # Wybieramy potrzebne informacje o przedmiotach
    catalog_columns = [
        column
        for column in [
            "item_id",
            "name",
            "category",
            "price"
        ]
        if column in catalog.columns
    ]

    catalog = catalog[catalog_columns].drop_duplicates(
        subset=["item_id"]
    )

    match_files = list(MATCHES_DIR.rglob("*.json"))

    print(f"Liczba znalezionych meczów: {len(match_files)}")
    print(f"Liczba przedmiotów w katalogu: {len(catalog)}")

    all_purchases = []

    for index, match_file in enumerate(match_files, start=1):

        try:

            purchases = process_match(match_file, catalog)

            all_purchases.extend(purchases)

        except Exception as error:

            print(f"Błąd meczu {match_file.stem}: {error}")

        if index % 500 == 0:
            print(
                f"Przetworzono {index}/{len(match_files)} meczów"
            )

    if not all_purchases:
        print("Nie znaleziono żadnych zakupów.")
        return

    df = pd.DataFrame(all_purchases)

    # Dodajemy nazwy, kategorie i ceny przedmiotów
    df = df.merge(
        catalog,
        on="item_id",
        how="left"
    )

    df = df.sort_values(
        by=[
            "match_id",
            "participant_id",
            "timestamp"
        ]
    )

    df.to_csv(
        PURCHASE_EVENTS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("PRZETWARZANIE TIMELINE ZAKOŃCZONE")
    print("=" * 60)

    print(f"Liczba meczów: {len(match_files)}")
    print(f"Liczba zachowanych zakupów: {len(df)}")
    print(f"Liczba graczy: {df['puuid'].nunique()}")
    print(f"Zakupy bez nazwy: {df['name'].isna().sum()}")

    print(f"\nZapisano: {PURCHASE_EVENTS_FILE}")

    print("\nPRZYKŁADOWE ZAKUPY:")

    print(
        df[
            [
                "champion",
                "position",
                "name",
                "category",
                "purchase_number",
                "time_minutes",
                "win"
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
