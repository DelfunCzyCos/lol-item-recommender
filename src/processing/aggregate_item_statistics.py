
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/high_elo/item_purchase_events.csv"
)

OUTPUT_DIR = Path(
    "data/processed/high_elo/statistics"
)

OUTPUT_FILE = OUTPUT_DIR / "champion_item_statistics.csv"


def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "match_id",
        "participant_id",
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

    # Ujednolicenie wartości win
    if df["win"].dtype != bool:
        df["win"] = (
            df["win"]
            .astype(str)
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False
            })
        )

    df = df.dropna(
        subset=[
            "match_id",
            "participant_id",
            "champion",
            "position",
            "win",
            "item_id",
            "purchase_number"
        ]
    ).copy()

    df["item_id"] = df["item_id"].astype(int)
    df["purchase_number"] = (
        df["purchase_number"].astype(int)
    )

    return df


def calculate_statistics(df):

    # Jedna obserwacja oznacza jednego gracza,
    # w jednym meczu, który kupił dany przedmiot
    # na określonej pozycji zakupowej.

    observations = df.drop_duplicates(
        subset=[
            "match_id",
            "participant_id",
            "champion",
            "position",
            "purchase_number",
            "item_id"
        ]
    ).copy()

    # Ogólny win rate liczony na poziomie uczestników meczów
    match_players = df.drop_duplicates(
        subset=["match_id", "participant_id"]
    )

    global_win_rate = match_players["win"].mean()

    print(
        f"Ogólny win rate w zbiorze: "
        f"{global_win_rate * 100:.2f}%"
    )

    # Liczba wszystkich obserwacji dla danej kombinacji
    # bohater + pozycja + numer zakupu
    slot_totals = (
        observations.groupby([
            "champion",
            "position",
            "purchase_number"
        ])
        .agg(
            total_observations=("match_id", "size"),
            total_matches=("match_id", "nunique")
        )
        .reset_index()
    )

    # Statystyki dla każdego przedmiotu
    result = (
        observations.groupby([
            "champion",
            "position",
            "purchase_number",
            "item_id",
            "name",
            "category"
        ])
        .agg(
            item_count=("match_id", "size"),
            matches=("match_id", "nunique"),
            wins=("win", "sum")
        )
        .reset_index()
    )

    result["losses"] = (
        result["item_count"] - result["wins"]
    )

    # Surowy win rate
    result["win_rate"] = (
        result["wins"] / result["item_count"]
    )

    # Łączymy statystyki przedmiotu z liczbą
    # wszystkich obserwacji w danym slocie.
    result = result.merge(
        slot_totals,
        on=[
            "champion",
            "position",
            "purchase_number"
        ],
        how="left"
    )

    # Udział przedmiotu w zakupach danego slotu
    result["item_share"] = (
        result["item_count"]
        / result["total_observations"]
    )

    # Wygładzony win rate.
    # Przy małej liczbie obserwacji wynik jest
    # częściowo przesuwany w stronę ogólnego win rate.
    prior_strength = 20

    result["win_rate_smoothed"] = (
        result["wins"]
        + prior_strength * global_win_rate
    ) / (
        result["item_count"] + prior_strength
    )

    # Czytelniejsze wartości procentowe
    for column in [
        "win_rate",
        "win_rate_smoothed",
        "item_share"
    ]:
        result[column] = (
            result[column] * 100
        ).round(2)

    result = result.sort_values(
        [
            "champion",
            "position",
            "purchase_number",
            "win_rate_smoothed",
            "item_count"
        ],
        ascending=[
            True,
            True,
            True,
            False,
            False
        ]
    )

    return result


def show_examples(result):

    print("\n" + "=" * 70)
    print("PRZYKŁADOWE REKOMENDACJE STATYSTYCZNE")
    print("=" * 70)

    # Przykładowo pokazujemy Ahri na MIDDLE
    example = result[
        (result["champion"] == "Ahri")
        & (result["position"] == "MIDDLE")
        & (result["purchase_number"] == 3)
    ]

    if example.empty:
        print(
            "Brak danych dla Ahri MIDDLE, "
            "trzeci zakup."
        )
        return

    example = example.sort_values(
        "win_rate_smoothed",
        ascending=False
    )

    print(
        example[
            [
                "name",
                "item_count",
                "wins",
                "losses",
                "win_rate",
                "win_rate_smoothed",
                "item_share"
            ]
        ].head(15).to_string(index=False)
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Wczytywanie danych...")

    df = load_data()

    print(f"Liczba wierszy źródłowych: {len(df)}")

    result = calculate_statistics(df)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("STATYSTYKI PRZEDMIOTÓW ZAKOŃCZONE")
    print("=" * 70)

    print(
        f"Liczba kombinacji bohater/pozycja/zakup/przedmiot: "
        f"{len(result)}"
    )

    print(
        f"Liczba bohaterów: "
        f"{result['champion'].nunique()}"
    )

    print(
        f"Liczba pozycji: "
        f"{result['position'].nunique()}"
    )

    print(f"\nZapisano: {OUTPUT_FILE}")

    show_examples(result)


if __name__ == "__main__":
    main()
