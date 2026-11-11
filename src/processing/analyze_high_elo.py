
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed/high_elo")

PLAYERS_FILE = DATA_DIR / "players_high_elo.csv"
ITEMS_FILE = DATA_DIR / "items_high_elo.csv"

OUTPUT_FILE = DATA_DIR / "item_stats_high_elo.csv"
TIER_OUTPUT_FILE = DATA_DIR / "item_stats_high_elo_by_tier.csv"

# Minimalna liczba gier, żeby przedmiot trafił do wiarygodnego rankingu
MIN_GAMES = 10

# Siła wygładzania win rate
PRIOR_STRENGTH = 20


def calculate_item_stats(players, items, group_columns):

    # Usuwamy ewentualne powtórzenia graczy w meczu
    players = players.drop_duplicates(
        subset=["matchId", "puuid"]
    ).copy()

    # Usuwamy ewentualne powtórzenia przedmiotów
    items = items.drop_duplicates(
        subset=["matchId", "puuid", "item_id"]
    ).copy()

    players["win"] = players["win"].astype(int)
    items["win"] = items["win"].astype(int)

    # Liczba wszystkich gier i bazowy win rate
    baseline = (
        players
        .groupby(group_columns)
        .agg(
            total_games=("matchId", "size"),
            baseline_win_rate=("win", "mean")
        )
        .reset_index()
    )

    # Liczba gier z danym przedmiotem i liczba zwycięstw
    stats = (
        items
        .groupby(group_columns + ["item_id"])
        .agg(
            item_games=("matchId", "size"),
            wins=("win", "sum")
        )
        .reset_index()
    )

    # Dołączamy liczbę wszystkich gier danej postaci i pozycji
    stats = stats.merge(
        baseline,
        on=group_columns,
        how="left"
    )

    # Podstawowe wskaźniki
    stats["pick_rate"] = (
        stats["item_games"] / stats["total_games"]
    )

    stats["win_rate"] = (
        stats["wins"] / stats["item_games"]
    )

    # Wygładzony win rate (Bayesian smoothing)
    stats["smoothed_win_rate"] = (
        stats["wins"]
        + stats["baseline_win_rate"] * PRIOR_STRENGTH
    ) / (
        stats["item_games"] + PRIOR_STRENGTH
    )

    stats["eligible"] = stats["item_games"] >= MIN_GAMES

    return stats.sort_values(
        by=["eligible", "smoothed_win_rate", "item_games"],
        ascending=[False, False, False]
    )


def main():

    print("Wczytywanie danych...")

    players = pd.read_csv(PLAYERS_FILE)
    items = pd.read_csv(ITEMS_FILE)

    # Ujednolicamy nazwę kolumny ID przedmiotu
    if "itemId" in items.columns and "item_id" not in items.columns:
        items = items.rename(columns={"itemId": "item_id"})

    required_players = [
        "matchId", "puuid", "champion", "position", "win", "tier"
    ]

    required_items = [
        "matchId", "puuid", "champion", "position",
        "win", "tier", "item_id"
    ]

    for column in required_players:
        if column not in players.columns:
            raise ValueError(f"Brakuje kolumny w players: {column}")

    for column in required_items:
        if column not in items.columns:
            raise ValueError(f"Brakuje kolumny w items: {column}")

    # Analiza wszystkich rang razem
    print("\nAnaliza wszystkich rang...")

    overall_stats = calculate_item_stats(
        players,
        items,
        ["champion", "position"]
    )

    overall_stats.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # Analiza osobno według źródłowej rangi meczu
    print("Analiza według rangi źródłowej...")

    tier_stats = calculate_item_stats(
        players,
        items,
        ["tier", "champion", "position"]
    )

    tier_stats.to_csv(
        TIER_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 45)
    print("ANALIZA ZAKOŃCZONA")
    print("=" * 45)

    print(f"Liczba przeanalizowanych graczy: {len(players)}")
    print(f"Liczba rekordów przedmiotów: {len(items)}")

    print(f"\nZapisano: {OUTPUT_FILE}")
    print(f"Zapisano: {TIER_OUTPUT_FILE}")

    # Pokazujemy przykładowe najlepsze wyniki
    print("\nTOP 15 PRZEDMIOTÓW:")

    top = overall_stats[
        overall_stats["eligible"]
    ].head(15)

    print(
        top[
            [
                "champion",
                "position",
                "item_id",
                "item_games",
                "pick_rate",
                "win_rate",
                "smoothed_win_rate"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
