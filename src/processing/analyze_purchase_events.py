
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/high_elo/item_purchase_events.csv"
)

OUTPUT_DIR = Path("data/processed/high_elo/analysis")


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "match_id",
        "puuid",
        "champion",
        "position",
        "win",
        "item_id",
        "name",
        "category",
        "purchase_number"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Brakuje kolumn: {missing}"
        )

    return df


def analyze_categories(df):

    print("\n" + "=" * 60)
    print("1. ZAKUPY WEDŁUG KATEGORII")
    print("=" * 60)

    result = (
        df.groupby("category")
        .agg(
            liczba_zakupow=("item_id", "count"),
            liczba_graczy=("puuid", "nunique"),
            liczba_meczow=("match_id", "nunique")
        )
        .sort_values(
            "liczba_zakupow",
            ascending=False
        )
    )

    result["udzial_procentowy"] = (
        result["liczba_zakupow"]
        / len(df)
        * 100
    ).round(2)

    print(result.to_string())

    result.to_csv(
        OUTPUT_DIR / "category_analysis.csv",
        encoding="utf-8-sig"
    )


def analyze_champions_positions(df):

    print("\n" + "=" * 60)
    print("2. DANE WEDŁUG BOHATERA I POZYCJI")
    print("=" * 60)

    result = (
        df.groupby(["champion", "position"])
        .agg(
            liczba_meczow=("match_id", "nunique"),
            liczba_graczy=("puuid", "nunique"),
            liczba_zakupow=("item_id", "count"),
            win_rate=("win", "mean")
        )
        .reset_index()
    )

    result["win_rate"] = (
        result["win_rate"] * 100
    ).round(2)

    result = result.sort_values(
        "liczba_meczow",
        ascending=False
    )

    print("\nNajwięcej danych:")
    print(result.head(30).to_string(index=False))

    print("\nNajmniej danych:")
    print(result.tail(20).to_string(index=False))

    result.to_csv(
        OUTPUT_DIR / "champion_position_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )


def analyze_purchase_numbers(df):

    print("\n" + "=" * 60)
    print("3. LICZBA ZAKUPÓW WEDŁUG KOLEJNOŚCI")
    print("=" * 60)

    result = (
        df.groupby("purchase_number")
        .agg(
            liczba_zakupow=("item_id", "count"),
            liczba_meczow=("match_id", "nunique"),
            liczba_graczy=("puuid", "nunique")
        )
        .reset_index()
    )

    result["udzial_procentowy"] = (
        result["liczba_zakupow"]
        / len(df)
        * 100
    ).round(2)

    print(result.to_string(index=False))

    result.to_csv(
        OUTPUT_DIR / "purchase_number_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )


def analyze_champion_purchase_slots(df):

    print("\n" + "=" * 60)
    print("4. LICZBA OBSERWACJI DLA BOHATERA, POZYCJI I ZAKUPU")
    print("=" * 60)

    result = (
        df.groupby([
            "champion",
            "position",
            "purchase_number"
        ])
        .agg(
            liczba_obserwacji=("item_id", "count"),
            liczba_meczow=("match_id", "nunique")
        )
        .reset_index()
    )

    result = result.sort_values(
        [
            "champion",
            "position",
            "purchase_number"
        ]
    )

    print("\nPrzykładowe kombinacje:")
    print(result.head(40).to_string(index=False))

    result.to_csv(
        OUTPUT_DIR / "champion_purchase_slots.csv",
        index=False,
        encoding="utf-8-sig"
    )


def analyze_popular_items(df):

    print("\n" + "=" * 60)
    print("5. NAJCZĘŚCIEJ KUPOWANE PRZEDMIOTY")
    print("=" * 60)

    result = (
        df.groupby([
            "category",
            "item_id",
            "name"
        ])
        .agg(
            liczba_zakupow=("item_id", "count"),
            liczba_graczy=("puuid", "nunique"),
            liczba_meczow=("match_id", "nunique"),
            win_rate=("win", "mean")
        )
        .reset_index()
    )

    result["win_rate"] = (
        result["win_rate"] * 100
    ).round(2)

    result = result.sort_values(
        ["category", "liczba_zakupow"],
        ascending=[True, False]
    )

    for category in result["category"].unique():

        print(f"\nKategoria: {category}")

        category_data = result[
            result["category"] == category
        ].head(15)

        print(
            category_data[
                [
                    "name",
                    "liczba_zakupow",
                    "liczba_meczow",
                    "win_rate"
                ]
            ].to_string(index=False)
        )

    result.to_csv(
        OUTPUT_DIR / "popular_items_analysis.csv",
        index=False,
        encoding="utf-8-sig"
    )


def analyze_data_quality(df):

    print("\n" + "=" * 60)
    print("6. KONTROLA JAKOŚCI DANYCH")
    print("=" * 60)

    print(f"Liczba rekordów: {len(df)}")

    print(
        f"Brakujące nazwy: {df['name'].isna().sum()}"
    )

    print(
        f"Brakujące kategorie: {df['category'].isna().sum()}"
    )

    print(
        f"Brakujące pozycje: {df['position'].isna().sum()}"
    )

    print(
        f"Brakujące numery zakupów: "
        f"{df['purchase_number'].isna().sum()}"
    )

    duplicate_columns = [
        "match_id",
        "participant_id",
        "timestamp",
        "item_id"
    ]

    duplicates = df.duplicated(
        subset=duplicate_columns
    ).sum()

    print(f"Potencjalne duplikaty: {duplicates}")

    print("\nLiczba meczów:", df["match_id"].nunique())
    print("Liczba graczy:", df["puuid"].nunique())
    print("Liczba bohaterów:", df["champion"].nunique())


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Wczytywanie danych...")

    df = load_data()

    print(f"Wczytano rekordów: {len(df)}")

    analyze_categories(df)

    analyze_champions_positions(df)

    analyze_purchase_numbers(df)

    analyze_champion_purchase_slots(df)

    analyze_popular_items(df)

    analyze_data_quality(df)

    print("\n" + "=" * 60)
    print("ANALIZA ZAKOŃCZONA")
    print("=" * 60)

    print(f"Wyniki zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
