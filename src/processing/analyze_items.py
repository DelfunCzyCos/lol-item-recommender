
from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/processed")


def analyze_items():
    players_path = DATA_DIR / "players.csv"
    items_path = DATA_DIR / "items.csv"

    if not players_path.exists() or not items_path.exists():
        print("Brakuje players.csv lub items.csv!")
        return

    players = pd.read_csv(players_path)
    items = pd.read_csv(items_path)

    # Usuwamy rekordy bez podstawowych informacji.
    players = players.dropna(
        subset=["matchId", "puuid", "champion", "position", "win"]
    )

    items = items.dropna(
        subset=["matchId", "puuid", "itemId"]
    )

    # Pomijamy nieprawidłowe ID przedmiotów.
    items["itemId"] = pd.to_numeric(items["itemId"], errors="coerce")
    items = items.dropna(subset=["itemId"])
    items = items[items["itemId"] != 0]

    # Jeden przedmiot liczymy maksymalnie raz na gracza w meczu.
    items = items.drop_duplicates(
        subset=["matchId", "puuid", "itemId"]
    )

    # Liczba rozegranych meczów dla każdej kombinacji bohatera i pozycji.
    totals = (
        players.groupby(["champion", "position"])
        .agg(total_games=("matchId", "count"),
             baseline_win_rate=("win", "mean"))
        .reset_index()
    )

    # Łączymy przedmioty z informacją o zwycięstwie.
    item_data = items.copy()

    # Statystyki każdego przedmiotu dla bohatera i pozycji.
    stats = (
        item_data.groupby(["champion", "position", "itemId"])
        .agg(
            item_games=("matchId", "count"),
            wins=("win", "sum")
        )
        .reset_index()
    )

    stats = stats.merge(
        totals,
        on=["champion", "position"],
        how="left"
    )

    stats["pick_rate"] = (
        stats["item_games"] / stats["total_games"]
    )

    stats["win_rate"] = (
        stats["wins"] / stats["item_games"]
    )

    # Wygładzony win rate ogranicza wpływ bardzo małych próbek.
    prior_strength = 10

    stats["smoothed_win_rate"] = (
        stats["wins"]
        + prior_strength * stats["baseline_win_rate"]
    ) / (
        stats["item_games"] + prior_strength
    )

    stats = stats.sort_values(
        by=["champion", "position", "smoothed_win_rate"],
        ascending=[True, True, False]
    )

    output_path = DATA_DIR / "item_stats.csv"
    stats.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAnaliza zakończona!")
    print(f"Przeanalizowano {len(stats)} kombinacji przedmiotów.")
    print(f"Wyniki zapisano w: {output_path}")

    print("\nPrzykładowe wyniki:")
    print(
        stats[
            [
                "champion",
                "position",
                "itemId",
                "item_games",
                "pick_rate",
                "win_rate",
                "smoothed_win_rate"
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    analyze_items()
