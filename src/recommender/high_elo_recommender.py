
from pathlib import Path
import pandas as pd


DATA_FILE = Path(
    "data/processed/high_elo/item_stats_high_elo_named.csv"
)

MIN_GAMES = 10
TOP_N = 10


def main():

    if not DATA_FILE.exists():
        print("Nie znaleziono pliku z analizą!")
        return

    df = pd.read_csv(DATA_FILE)

    # Wybór postaci
    champions = sorted(df["champion"].dropna().unique())

    print("\nDOSTĘPNE POSTACIE:")
    print(", ".join(champions))

    champion = input("\nPodaj nazwę postaci: ").strip()

    selected = df[
        df["champion"].str.lower() == champion.lower()
    ]

    if selected.empty:
        print("Nie znaleziono takiej postaci.")
        return

    # Wybór pozycji
    positions = sorted(selected["position"].dropna().unique())

    print("\nDOSTĘPNE POZYCJE:")
    print(", ".join(positions))

    position = input("\nPodaj pozycję: ").strip().upper()

    selected = selected[
        selected["position"].str.upper() == position
    ]

    if selected.empty:
        print("Brak danych dla tej postaci na wybranej pozycji.")
        return

    # Odrzucamy przedmioty z małą liczbą gier
    selected = selected[
        selected["item_games"] >= MIN_GAMES
    ].copy()

    if selected.empty:
        print(
            f"Za mało danych. Żaden przedmiot nie ma "
            f"minimum {MIN_GAMES} gier."
        )
        return

    # Sortujemy według wygładzonego win rate
    selected = selected.sort_values(
        by=["smoothed_win_rate", "item_games"],
        ascending=[False, False]
    )

    print("\n" + "=" * 65)
    print(f"REKOMENDACJE HIGH ELO: {champion} - {position}")
    print("=" * 65)

    for index, (_, item) in enumerate(
        selected.head(TOP_N).iterrows(),
        start=1
    ):

        print(f"\n{index}. {item['name']}")
        print(f"   ID przedmiotu: {item['item_id']}")
        print(f"   Liczba gier: {int(item['item_games'])}")
        print(f"   Win rate: {item['win_rate']:.1%}")
        print(
            f"   Wygładzony win rate: "
            f"{item['smoothed_win_rate']:.1%}"
        )
        print(f"   Pick rate: {item['pick_rate']:.1%}")

    print("\n" + "=" * 65)
    print("Analiza na podstawie meczów High Elo EUNE.")


if __name__ == "__main__":
    main()
