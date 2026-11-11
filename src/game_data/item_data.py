
import requests
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("data/processed")

BASE_URL = "https://ddragon.leagueoflegends.com"


def download_item_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Pobieranie wersji Data Dragon...")

    versions_response = requests.get(
        f"{BASE_URL}/api/versions.json",
        timeout=20
    )
    versions_response.raise_for_status()

    version = versions_response.json()[0]

    print(f"Używana wersja: {version}")

    print("Pobieranie danych przedmiotów...")

    url = (
        f"{BASE_URL}/cdn/{version}/data/"
        f"pl_PL/item.json"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    data = response.json()["data"]

    items = []

    for item_id, item in data.items():
        image_name = item.get("image", {}).get("full", "")

        items.append({
            "itemId": int(item_id),
            "name": item.get("name"),
            "description": item.get("description"),
            "plaintext": item.get("plaintext"),
            "price": item.get("gold", {}).get("total"),
            "image": image_name,
            "image_url": (
                f"{BASE_URL}/cdn/{version}/img/item/{image_name}"
                if image_name else None
            ),
            "version": version
        })

    df = pd.DataFrame(items)

    output_path = OUTPUT_DIR / "items_catalog.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nGotowe!")
    print(f"Liczba przedmiotów: {len(df)}")
    print(f"Zapisano: {output_path}")

    print("\nPrzykładowe przedmioty:")
    print(df[["itemId", "name", "price"]].head(10).to_string(index=False))


if __name__ == "__main__":
    download_item_data()
