
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/processed/high_elo")
CATALOG_FILE = Path("data/processed/items_catalog.csv")


def main():

    stats_file = DATA_DIR / "item_stats_high_elo.csv"

    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono katalogu przedmiotów: {CATALOG_FILE}"
        )

    stats = pd.read_csv(stats_file)
    catalog = pd.read_csv(CATALOG_FILE)

    # Ujednolicamy nazwy kolumn ID
    if "itemId" in catalog.columns:
        catalog = catalog.rename(columns={"itemId": "item_id"})
    elif "ID" in catalog.columns:
        catalog = catalog.rename(columns={"ID": "item_id"})
    elif "id" in catalog.columns:
        catalog = catalog.rename(columns={"id": "item_id"})

    if "item_id" not in catalog.columns:
        raise ValueError(
            f"Nie znaleziono kolumny ID w katalogu. "
            f"Dostępne kolumny: {list(catalog.columns)}"
        )

    # Dopasowujemy typy ID
    stats["item_id"] = stats["item_id"].astype(str)
    catalog["item_id"] = catalog["item_id"].astype(str)

    # Wybieramy dostępne informacje o przedmiotach
    catalog_columns = [
        column for column in [
            "item_id",
            "name",
            "description",
            "plaintext",
            "price",
            "image_url",
            "version",
            "category"
        ]
        if column in catalog.columns
    ]

    catalog = catalog[catalog_columns].drop_duplicates(
        subset=["item_id"]
    )

    # Łączymy statystyki z katalogiem
    result = stats.merge(
        catalog,
        on="item_id",
        how="inner"
    )

    output_file = DATA_DIR / "item_stats_high_elo_named.csv"

    result.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    missing = result["name"].isna().sum() if "name" in result.columns else 0

    print("\nPOŁĄCZENIE ZAKOŃCZONE")
    print(f"Liczba rekordów: {len(result)}")
    print(f"Przedmioty bez dopasowanej nazwy: {missing}")
    print(f"Zapisano: {output_file}")

    print("\nTOP 15 PRZEDMIOTÓW:")

    top = result[result["eligible"]].head(15)

    columns_to_show = [
        column for column in [
            "champion",
            "position",
            "name",
            "item_id",
            "item_games",
            "win_rate",
            "smoothed_win_rate"
        ]
        if column in top.columns
    ]

    print(top[columns_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
