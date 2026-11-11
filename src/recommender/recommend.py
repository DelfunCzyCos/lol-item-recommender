
from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/processed/item_recommendations.csv")


def recommend_items():
    if not DATA_PATH.exists():
        print("Nie znaleziono pliku item_recommendations.csv!")
        return

    df = pd.read_csv(DATA_PATH)

    champion = input("\nPodaj bohatera (np. Jinx): ").strip()
    position = input("Podaj pozycję (np. BOTTOM): ").strip().upper()

    # Wyszukujemy wybranego bohatera i pozycję.
    results = df[
        (df["champion"].str.casefold() == champion.casefold())
        & (df["position"].str.upper() == position)
    ].copy()

    if results.empty:
        print("\nNie znaleziono danych dla tego bohatera i pozycji.")
        print("Sprawdź pisownię bohatera i pozycji.")
        return

    # Normalizacja pick rate w obrębie wybranego bohatera i pozycji.
    max_pick_rate = results["pick_rate"].max()

    if max_pick_rate > 0:
        results["relative_pick_rate"] = (
            results["pick_rate"] / max_pick_rate
        )
    else:
        results["relative_pick_rate"] = 0

    # Pewność wyniku rośnie wraz z liczbą meczów.
    results["confidence"] = (
        results["item_games"] / (results["item_games"] + 10)
    )

    # Wynik algorytmu: win rate, popularność i liczba obserwacji.
    results["score"] = (
        0.65 * results["smoothed_win_rate"]
        + 0.25 * results["relative_pick_rate"]
        + 0.10 * results["confidence"]
    )

    results = results.sort_values(
        by="score",
        ascending=False
    )

    print(f"\nREKOMENDACJE DLA: {champion} - {position}")
    print("=" * 65)

    top_items = results.head(5)

    for index, (_, item) in enumerate(top_items.iterrows(), start=1):
        print(f"\n{index}. {item['name']} (ID: {int(item['itemId'])})")
        print(f"   Cena: {item['price']}")
        print(f"   Win rate: {item['win_rate']:.1%}")
        print(f"   Wygładzony win rate: {item['smoothed_win_rate']:.1%}")
        print(f"   Pick rate: {item['pick_rate']:.1%}")
        print(f"   Liczba meczów: {int(item['item_games'])}")
        print(f"   Wynik algorytmu: {item['score']:.3f}")

    print("\nUWAGA: Wyniki są eksperymentalne.")
    print("Mała liczba meczów oznacza niską wiarygodność rekomendacji.")


if __name__ == "__main__":
    recommend_items()
