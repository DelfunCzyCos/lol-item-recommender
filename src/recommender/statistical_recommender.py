
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/high_elo/item_purchase_events.csv"
)

MIN_ITEM_COUNT = 10
MAX_RECOMMENDATIONS = 5

CATEGORY_ORDER = [
    "starter",
    "boots",
    "legendary"
]

CATEGORY_NAMES = {
    "starter": "STARTERY",
    "boots": "BUTY",
    "legendary": "FULL ITEMY"
}


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
        "timestamp"
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

    df = df.dropna(
        subset=[
            "match_id",
            "participant_id",
            "champion",
            "position",
            "item_id",
            "category"
        ]
    ).copy()

    df["item_id"] = df["item_id"].astype(int)
    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["win"] = df["win"].astype(bool)

    return df


def assign_category_order(df):

    """
    Wyznacza osobną kolejność zakupów dla każdej kategorii.

    Kolejność jest liczona osobno dla każdego gracza
    w każdym meczu.
    """

    df = df.copy()

    df = df.sort_values(
        [
            "match_id",
            "participant_id",
            "timestamp"
        ]
    )

    df["category_purchase_number"] = (
        df.groupby(
            [
                "match_id",
                "participant_id",
                "category"
            ]
        ).cumcount() + 1
    )

    return df


def choose_champion(df):

    champions = sorted(
        df["champion"].dropna().unique()
    )

    print("\nDOSTĘPNI BOHATEROWIE")

    for champion in champions:
        print(champion)

    while True:

        choice = input(
            "\nWpisz nazwę bohatera: "
        ).strip()

        matches = [
            champion
            for champion in champions
            if champion.lower() == choice.lower()
        ]

        if matches:
            return matches[0]

        print(
            "Nie znaleziono takiego bohatera. "
            "Spróbuj ponownie."
        )


def choose_position(df, champion):

    positions = sorted(
        df.loc[
            df["champion"] == champion,
            "position"
        ].dropna().unique()
    )

    print(
        f"\nDOSTĘPNE POZYCJE DLA {champion.upper()}"
    )

    for position in positions:
        print(position)

    while True:

        choice = input(
            "\nWpisz pozycję: "
        ).strip().upper()

        if choice in positions:
            return choice

        print(
            "Nie znaleziono takiej pozycji. "
            "Spróbuj ponownie."
        )


def generate_statistics(df, champion, position):

    selected = df[
        (df["champion"] == champion)
        & (df["position"] == position)
    ].copy()

    if selected.empty:
        return pd.DataFrame()

    statistics = (
        selected.groupby(
            [
                "category",
                "category_purchase_number",
                "item_id",
                "name"
            ]
        )
        .agg(
            item_count=("match_id", "count"),
            wins=("win", "sum"),
            matches=("match_id", "nunique")
        )
        .reset_index()
    )

    statistics["losses"] = (
        statistics["item_count"] - statistics["wins"]
    )

    statistics["win_rate"] = (
        statistics["wins"]
        / statistics["item_count"]
        * 100
    )

    # Globalny win rate dla wybranego bohatera i pozycji
    baseline_win_rate = selected["win"].mean() * 100

    # Wygładzanie wyników dla małych prób
    smoothing_strength = 20

    statistics["win_rate_smoothed"] = (
        (
            statistics["wins"]
            + smoothing_strength * baseline_win_rate / 100
        )
        /
        (
            statistics["item_count"]
            + smoothing_strength
        )
    ) * 100

    # Udział przedmiotu w zakupach danej kategorii i kolejności
    category_totals = (
        statistics.groupby(
            ["category", "category_purchase_number"]
        )["item_count"]
        .transform("sum")
    )

    statistics["item_share"] = (
        statistics["item_count"]
        / category_totals
        * 100
    )

    return statistics


def print_recommendations(statistics, champion, position):

    print("\n" + "=" * 75)
    print(
        f"REKOMENDACJE: {champion.upper()} | {position}"
    )
    print("=" * 75)

    if statistics.empty:
        print("Brak danych dla wybranej kombinacji.")
        return

    for category in CATEGORY_ORDER:

        category_data = statistics[
            (statistics["category"] == category)
            & (
                statistics["item_count"]
                >= MIN_ITEM_COUNT
            )
        ].copy()

        print("\n" + "-" * 75)
        print(CATEGORY_NAMES[category])
        print("-" * 75)

        if category_data.empty:
            print(
                "Brak przedmiotów z wystarczającą "
                "liczbą obserwacji."
            )
            continue

        purchase_numbers = sorted(
            category_data[
                "category_purchase_number"
            ].unique()
        )

        for purchase_number in purchase_numbers:

            slot_data = category_data[
                category_data[
                    "category_purchase_number"
                ] == purchase_number
            ].copy()

            slot_data = slot_data.sort_values(
                [
                    "win_rate_smoothed",
                    "item_count"
                ],
                ascending=False
            ).head(MAX_RECOMMENDATIONS)

            print(
                f"\nKolejność w kategorii: "
                f"{purchase_number}"
            )

            print(
                slot_data[
                    [
                        "name",
                        "item_count",
                        "wins",
                        "losses",
                        "win_rate",
                        "win_rate_smoothed",
                        "item_share"
                    ]
                ].round(2).to_string(index=False)
            )


def main():

    print("Wczytywanie danych...")

    df = load_data()

    print(f"Liczba zakupów: {len(df)}")

    # Wyliczamy oddzielną kolejność dla każdej kategorii
    df = assign_category_order(df)

    champion = choose_champion(df)

    position = choose_position(
        df,
        champion
    )

    statistics = generate_statistics(
        df,
        champion,
        position
    )

    print_recommendations(
        statistics,
        champion,
        position
    )

    print("\n" + "=" * 75)
    print("ANALIZA ZAKOŃCZONA")
    print("=" * 75)

if __name__ == "__main__":
    main()
