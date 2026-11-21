"""
Trenuje finalny model XGBoost na całym zbiorze High Elo i zapisuje go
wraz z enkoderami i metadanymi przedmiotów do `models/xgb_recommender.joblib`.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.recommender.ml_dataset import (
    build_legendary_purchase_dataset,
    build_matchup_table,
    load_raw_tables,
)


MODEL_PATH = Path("models/xgb_recommender.joblib")
CATALOG_PATH = Path("data/processed/items_catalog.csv")
RECIPES_PATH = Path("data/processed/item_recipes.json")


def build_multihot(series, mapping):
    rows, cols = [], []
    for r_idx, vals in enumerate(series):
        for v in vals:
            c_idx = mapping.get(v)
            if c_idx is not None:
                rows.append(r_idx)
                cols.append(c_idx)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(series), len(mapping)))


def load_item_metadata(df: pd.DataFrame):
    """Buduje słownik metadanych przedmiotów oraz zbiory ID butów i legendarek."""
    item_metadata = {}

    for row in df.drop_duplicates("item_id").itertuples(index=False):
        item_metadata[int(row.item_id)] = {
            "item_id": int(row.item_id),
            "name": str(row.name),
            "price": int(row.price) if pd.notna(row.price) else 0,
            "category": str(row.category),
            "image_url": None,
        }

    boots_ids = set()
    legendary_ids = set(item_metadata.keys())

    starter_ids = set()

    if CATALOG_PATH.exists():
        cat_df = pd.read_csv(CATALOG_PATH)
        for row in cat_df.itertuples(index=False):
            iid = int(row.itemId)
            cat = str(row.category)
            if cat == "boots":
                boots_ids.add(iid)
            elif cat == "legendary":
                legendary_ids.add(iid)
            elif cat == "starter":
                starter_ids.add(iid)

            if iid not in item_metadata:
                item_metadata[iid] = {
                    "item_id": iid,
                    "name": str(row.name),
                    "price": int(row.price) if pd.notna(row.price) else 0,
                    "category": cat,
                    "image_url": str(row.image_url) if pd.notna(row.image_url) else None,
                }
            else:
                item_metadata[iid]["name"] = str(row.name)
                if pd.notna(row.price):
                    item_metadata[iid]["price"] = int(row.price)
                if pd.notna(row.image_url):
                    item_metadata[iid]["image_url"] = str(row.image_url)

    return item_metadata, boots_ids, legendary_ids, starter_ids


TIER3_TO_TIER2_BOOTS = {
    3168: 3008,  # Nieśmiertelna Ścieżka -> Nagolenniki Pasibrzucha
    3170: 3009,  # Żwawy Marsz -> Buty Prędkości
    3171: 3158,  # Karmazynowa Klarowność -> Ioniańskie Buty Jasności Umysłu
    3172: 3006,  # Spiżowe Nagolenniki -> Nagolenniki Berserkera
    3173: 3111,  # Spętane Kruszyciele -> Obuwie Merkurego
    3174: 3047,  # Opancerzone Natarcie -> Pancerniaki
    3175: 3020,  # Buty Miotacza Zaklęć -> Obuwie Maga
}


def build_starter_and_boots_stats():
    """Buduje statystyki High Elo dla starterów (początek gry) oraz butów (z matchupem)."""
    events, players = load_raw_tables()
    events = events.dropna(subset=["match_id", "puuid", "champion", "position", "item_id", "category"]).copy()
    events["item_id"] = events["item_id"].astype(int)
    events["time_minutes"] = pd.to_numeric(events["time_minutes"], errors="coerce").fillna(0.0)

    # Startery na start gry (pierwsze 2.5 min)
    starters = events[(events["category"] == "starter") & (events["time_minutes"] <= 2.5)]
    starter_stats = {}
    for (champ, pos), grp in starters.groupby(["champion", "position"]):
        counts = grp["item_id"].value_counts()
        total = counts.sum()
        starter_stats[(champ, pos)] = [(int(iid), float(cnt / total)) for iid, cnt in counts.head(3).items()]

    # Buty (łączenie Tier 3 z odpowiadającym Tier 2) z uwzględnieniem matchupu oraz fallbackiem
    boots = events[events["category"] == "boots"].copy()
    boots["item_id"] = boots["item_id"].replace(TIER3_TO_TIER2_BOOTS)
    matchups = build_matchup_table(players)[["match_id", "puuid", "enemy_lane_champion"]]
    boots = boots.merge(matchups, on=["match_id", "puuid"], how="left")
    boots["enemy_lane_champion"] = boots["enemy_lane_champion"].fillna("UNKNOWN")

    boots_stats = {}
    for (champ, pos), grp in boots.groupby(["champion", "position"]):
        counts = grp["item_id"].value_counts()
        total = counts.sum()
        boots_stats[(champ, pos)] = [(int(iid), float(cnt / total)) for iid, cnt in counts.head(3).items()]

    boots_matchup_stats = {}
    for (champ, pos, enemy), grp in boots.groupby(["champion", "position", "enemy_lane_champion"]):
        if len(grp) < 5:
            continue
        counts = grp["item_id"].value_counts()
        total = counts.sum()
        boots_matchup_stats[(champ, pos, enemy)] = [(int(iid), float(cnt / total)) for iid, cnt in counts.head(3).items()]

    return starter_stats, boots_stats, boots_matchup_stats


def train_and_save():
    print("[1/3] Wczytywanie i przygotowanie pełnego zbioru High Elo...")
    t0 = time.perf_counter()
    df = build_legendary_purchase_dataset(min_item_count=20)

    cat_cols = ["champion", "position", "tier", "enemy_lane_champion"]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat = ohe.fit_transform(df[cat_cols])

    num_cols = ["time_minutes", "legendary_slot", "has_boots"]
    scaler = StandardScaler()
    X_num = sparse.csr_matrix(scaler.fit_transform(df[num_cols].astype(float)))

    all_champs = sorted(set(df["champion"].unique()) | set(df["enemy_lane_champion"].unique()))
    champ_to_idx = {c: i for i, c in enumerate(all_champs)}

    all_items = sorted(df["item_id"].unique())
    item_to_idx = {int(item): i for i, item in enumerate(all_items)}

    X_enemy = build_multihot(df["enemy_champions"], champ_to_idx)
    X_ally = build_multihot(df["ally_champions"], champ_to_idx)
    X_prev = build_multihot(df["prev_items"], item_to_idx)

    X = sparse.hstack([X_cat, X_num, X_enemy, X_ally, X_prev], format="csr")

    le = LabelEncoder()
    y = le.fit_transform(df["item_id"])

    print(f"Zbiór gotowy w {time.perf_counter() - t0:.1f}s | Wiersze: {X.shape[0]}, Cechy: {X.shape[1]}, Klasy: {len(le.classes_)}")

    print("[2/3] Trenowanie modelu XGBoost...")
    t_train = time.perf_counter()
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    print(f"Trening zakończony w {time.perf_counter() - t_train:.1f}s")

    print("[3/3] Zapisywanie pakietu modelu...")
    item_metadata, boots_ids, legendary_ids, starter_ids = load_item_metadata(df)
    starter_stats, boots_stats, boots_matchup_stats = build_starter_and_boots_stats()

    item_recipes = {}
    if RECIPES_PATH.exists():
        with open(RECIPES_PATH, "r", encoding="utf-8") as f:
            raw_recipes = json.load(f)
        item_recipes = {int(k): v for k, v in raw_recipes.items()}

    bundle = {
        "model": model,
        "ohe": ohe,
        "scaler": scaler,
        "champ_to_idx": champ_to_idx,
        "item_to_idx": item_to_idx,
        "label_encoder": le,
        "item_metadata": item_metadata,
        "boots_ids": boots_ids,
        "legendary_ids": legendary_ids,
        "starter_ids": starter_ids,
        "starter_stats": starter_stats,
        "boots_stats": boots_stats,
        "boots_matchup_stats": boots_matchup_stats,
        "known_champions": all_champs,
        "item_recipes": item_recipes,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Zapisano model do: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
