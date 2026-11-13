
from pathlib import Path
import pandas as pd


DATA_FILE = Path(
    "data/processed/high_elo/item_stats_high_elo_named.csv"
)

MIN_GAMES = 5

# Ile przedmiotów wyświetlamy w każdej kategorii
TOP_STARTER = 3
TOP_BOOTS = 3
TOP_BUILD = 10


def show_category(data, category, title, top_n):

    # Wybieramy przedmioty danej kategorii
    items = data[
        data["category"].str.lower() == category
    ].copy()

    if items.empty:
        print(f"\n{title}")
        print("Brak wystarczających danych.")
        return

    # Sortowanie według wygładzonego win rate
    items = items.sort_values(
        by=["smoothed_win_rate", "item_games"],
        ascending=[False, False]
    )

    print("\n" + "-" * 65)
    print(title)
    print("-" * 65)

    for index, (_, item) in enumerate(
        items.head(top_n).iterrows(),
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


def main():

    if not DATA_FILE.exists():
        print("Nie znaleziono pliku z analizą!")
        return

    df = pd.read_csv(DATA_FILE)

    if "category" not in df.columns:
        print("Brak kolumny category!")
        print("Uruchom ponownie merge_high_elo_item_data.py")
        return

    # Wybór postaci
    champions = sorted(df["champion"].dropna().unique())

    print("\nDOSTĘPNE POSTACIE:")
    print(", ".join(champions))

    champion = input("\nPodaj nazwę postaci: ").strip()

    selected = df[
        df["champion"].str.lower() == champion.lower()
    ].copy()

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
    ].copy()

    if selected.empty:
        print("Brak danych dla tej postaci na wybranej pozycji.")
        return

    # Minimalna liczba gier
    selected = selected[
        selected["item_games"] >= MIN_GAMES
    ].copy()

    print("\n" + "=" * 65)
    print(f"REKOMENDACJE HIGH ELO: {champion} - {position}")
    print("=" * 65)

    # Trzy osobne kategorie
    show_category(
        selected,
        "starter",
        "1. PRZEDMIOTY STARTOWE",
        TOP_STARTER
    )

    show_category(
        selected,
        "boots",
        "2. BUTY",
        TOP_BOOTS
    )

    show_category(
        selected,
        "legendary",
        "3. GŁÓWNY BUILD",
        TOP_BUILD
    )

    print("\n" + "=" * 65)
    print("Analiza na podstawie meczów High Elo EUNE.")


if __name__ == "__main__":
    main()
