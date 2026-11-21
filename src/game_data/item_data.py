
import json
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


def download_item_recipes(version: str | None = None) -> dict[int, dict]:
    """Pobiera drzewko receptur (komponenty i koszty łączenia) z Data Dragon."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if version is None:
        versions_resp = requests.get(f"{BASE_URL}/api/versions.json", timeout=20)
        versions_resp.raise_for_status()
        version = versions_resp.json()[0]

    url = f"{BASE_URL}/cdn/{version}/data/pl_PL/item.json"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()["data"]

    recipes = {}
    for item_id_str, item in data.items():
        if not item_id_str.isdigit():
            continue
        iid = int(item_id_str)
        gold = item.get("gold", {})
        recipes[iid] = {
            "name": item.get("name", f"Item {iid}"),
            "total": int(gold.get("total", 0)),
            "base": int(gold.get("base", 0)),
            "from": [int(x) for x in item.get("from", []) if str(x).isdigit()],
        }

    recipes_path = OUTPUT_DIR / "item_recipes.json"
    with open(recipes_path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False)
    print(f"Zapisano receptury ({len(recipes)} przedmiotów, wersja {version}) do {recipes_path}")
    return recipes


def download_item_icons(
    catalog_path: Path = OUTPUT_DIR / "items_catalog.csv",
    icons_dir: Path = Path("assets/item_icons"),
):
    """Pobiera ikony PNG przedmiotów z Data Dragon do lokalnego folderu assets/item_icons."""
    if not catalog_path.exists():
        raise FileNotFoundError(f"Brak katalogu: {catalog_path}")

    icons_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(catalog_path)
    downloaded = 0

    for row in df.itertuples(index=False):
        iid = int(row.itemId)
        url = getattr(row, "image_url", None)
        if not isinstance(url, str) or not url:
            continue
        out_file = icons_dir / f"{iid}.png"
        if out_file.exists():
            continue
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            out_file.write_bytes(resp.content)
            downloaded += 1

    print(f"Pobrano {downloaded} nowych ikon do {icons_dir}")


def download_champion_icons(
    version: str | None = None,
    icons_dir: Path = Path("assets/champion_icons"),
):
    """Pobiera kwadratowe ikony bohaterów z Data Dragon do assets/champion_icons."""
    icons_dir.mkdir(parents=True, exist_ok=True)
    if version is None:
        versions_resp = requests.get(f"{BASE_URL}/api/versions.json", timeout=20)
        versions_resp.raise_for_status()
        version = versions_resp.json()[0]

    url = f"{BASE_URL}/cdn/{version}/data/en_US/champion.json"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    champs = resp.json()["data"]

    downloaded = 0
    for champ_key, champ_info in champs.items():
        image_full = champ_info.get("image", {}).get("full", f"{champ_key}.png")
        out_file = icons_dir / f"{champ_key}.png"
        if out_file.exists():
            continue
        img_url = f"{BASE_URL}/cdn/{version}/img/champion/{image_full}"
        img_resp = requests.get(img_url, timeout=15)
        if img_resp.status_code == 200:
            out_file.write_bytes(img_resp.content)
            downloaded += 1

    print(f"Pobrano {downloaded} ikon bohaterów (wersja {version}) do {icons_dir}")


if __name__ == "__main__":
    download_item_data()
    download_item_recipes()
