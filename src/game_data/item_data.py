
import requests
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("data/processed")

BASE_URL = "https://ddragon.leagueoflegends.com"


def get_item_category(item_id, item):

    tags = item.get("tags", [])

    # Przedmioty supportowe z questa
    if 3860 <= int(item_id) <= 3869:
        return "support"

    # Buty
    if "Boots" in tags:
        return "boots"

    # Startery
    STARTER_ITEMS = {
        1054,  # Tarcza Dorana
        1055,  # Ostrze Dorana
        1056,  # Pierścień Dorana
        1082,  # Pieczęć Mrozu
        1083,  # Ostrze Łowcy
        1101, 1102, 1103,  # Przedmioty dżunglowe
        3070,  # Łza Bogini     
    }

    if int(item_id) in STARTER_ITEMS:
        return "starter"

    return "legendary"

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
        if get_item_category(item_id, item) == "legendary":
            if item.get("gold", {}).get("total", 0) < 1700 and int(item_id) not in [3041]:
                continue
        elif get_item_category(item_id, item) == "boots":
            if item.get("gold", {}).get("total", 0) <= 300:
                continue
        elif item.get("into") and int(item_id) not in [1082, 3070]:
            continue
        if len(str(item_id)) != 4: continue

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
            "version": version,
            "category": get_item_category(int(item_id), item)
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
    print(df[["itemId", "name", "price", "category"]].to_string(index=False))


if __name__ == "__main__":
    download_item_data()
