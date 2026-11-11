from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/processed")


def merge_item_data():
    stats_path = DATA_DIR / "item_stats.csv"
    catalog_path = DATA_DIR / "items_catalog.csv"

    if not stats_path.exists() or not catalog_path.exists():
        print("Brakuje plików item_stats.csv lub items_catalog.csv!")
        return

    stats = pd.read_csv(stats_path)
    catalog = pd.read_csv(catalog_path)

    # Łączymy statystyki z katalogiem po ID przedmiotu.
    result = stats.merge(
        catalog[["itemId", "name", "price", "image_url"]],
        on="itemId",
        how="left"
    )

    result = result.sort_values(
        by="smoothed_win_rate",
        ascending=False
    )

    output_path = DATA_DIR / "item_recommendations.csv"

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPołączono dane!")
    print(f"Zapisano: {output_path}")

    print("\nPrzykładowe wyniki:")
    print(
        result[
            [
                "champion",
                "position",
                "name",
                "price",
                "item_games",
                "pick_rate",
                "win_rate",
                "smoothed_win_rate"
            ]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    merge_item_data()