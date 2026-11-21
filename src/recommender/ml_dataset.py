"""
Buduje zbiór danych ML z kontekstem meczu i matchupem na podstawie
`item_purchase_events.csv` oraz `players_high_elo.csv`.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


EVENTS_PATH = Path("data/processed/high_elo/item_purchase_events.csv")
PLAYERS_PATH = Path("data/processed/high_elo/players_high_elo.csv")


def load_raw_tables():
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"Brak pliku: {EVENTS_PATH}")
    if not PLAYERS_PATH.exists():
        raise FileNotFoundError(f"Brak pliku: {PLAYERS_PATH}")

    events = pd.read_csv(EVENTS_PATH)
    players = pd.read_csv(PLAYERS_PATH)
    return events, players


def build_matchup_table(players: pd.DataFrame) -> pd.DataFrame:
    """
    Dla każdego gracza w meczu wyznacza:
    - przeciwnika na tej samej pozycji (`enemy_lane_champion`)
    - listę wszystkich 5 przeciwników (`enemy_champions`)
    - listę 4 sojuszników (`ally_champions`)
    """
    p = players[["matchId", "puuid", "champion", "position", "win"]].copy()
    p["position"] = p["position"].fillna("UNKNOWN")

    # Mapa (matchId, win) -> lista bohaterów w drużynie
    team_champs = (
        p.groupby(["matchId", "win"])["champion"]
        .apply(list)
        .to_dict()
    )

    # Mapa (matchId, win, position) -> bohater na linii
    lane_champ = (
        p.drop_duplicates(subset=["matchId", "win", "position"])
        .set_index(["matchId", "win", "position"])["champion"]
        .to_dict()
    )

    records = []
    for row in p.itertuples(index=False):
        enemy_win = not bool(row.win)
        enemy_team = team_champs.get((row.matchId, enemy_win), [])
        ally_team = [c for c in team_champs.get((row.matchId, bool(row.win)), []) if c != row.champion]
        enemy_lane = lane_champ.get((row.matchId, enemy_win, row.position), "UNKNOWN")

        records.append({
            "match_id": row.matchId,
            "puuid": row.puuid,
            "enemy_lane_champion": enemy_lane,
            "enemy_champions": enemy_team,
            "ally_champions": ally_team,
        })

    return pd.DataFrame(records)


def build_legendary_purchase_dataset(min_item_count: int = 20):
    """
    Tworzy wiersze dla każdego zakupu przedmiotu `legendary` z historią
    wcześniejszych zakupów gracza w tym samym meczu oraz kontekstem matchupu.
    """
    events, players = load_raw_tables()

    # Sortowanie chronologiczne zakupów w ramach gracza i meczu
    events = events.dropna(
        subset=["match_id", "participant_id", "puuid", "champion", "position", "item_id", "category"]
    ).copy()
    events["item_id"] = events["item_id"].astype(int)
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="coerce").fillna(0)
    events["time_minutes"] = pd.to_numeric(events["time_minutes"], errors="coerce").fillna(0.0)
    events = events.sort_values(["match_id", "participant_id", "timestamp", "purchase_number"])

    # Zbieramy historię zakupów przed każdym zdarzeniem
    prev_items_list = []
    has_boots_list = []
    legendary_slot_list = []

    current_key = None
    owned_legendary = []
    has_boots = 0
    leg_count = 0

    for row in events.itertuples(index=False):
        key = (row.match_id, row.participant_id)
        if key != current_key:
            current_key = key
            owned_legendary = []
            has_boots = 0
            leg_count = 0

        if row.category == "legendary":
            leg_count += 1
            prev_items_list.append(list(owned_legendary))
            has_boots_list.append(has_boots)
            legendary_slot_list.append(leg_count)
            owned_legendary.append(row.item_id)
        else:
            prev_items_list.append(None)
            has_boots_list.append(has_boots)
            legendary_slot_list.append(0)
            if row.category == "boots":
                has_boots = 1

    events["prev_items"] = prev_items_list
    events["has_boots"] = has_boots_list
    events["legendary_slot"] = legendary_slot_list

    leg_df = events[events["category"] == "legendary"].copy()

    # Odfiltrowanie bardzo rzadkich przedmiotów, aby wszystkie klasy były obecne w train/test
    counts = leg_df["item_id"].value_counts()
    valid_items = counts[counts >= min_item_count].index
    leg_df = leg_df[leg_df["item_id"].isin(valid_items)].copy()

    # Dołączenie kontekstu matchupu
    matchups = build_matchup_table(players)
    leg_df = leg_df.merge(matchups, on=["match_id", "puuid"], how="left")
    leg_df["enemy_lane_champion"] = leg_df["enemy_lane_champion"].fillna("UNKNOWN")
    leg_df["enemy_champions"] = leg_df["enemy_champions"].apply(lambda x: x if isinstance(x, list) else [])
    leg_df["ally_champions"] = leg_df["ally_champions"].apply(lambda x: x if isinstance(x, list) else [])

    return leg_df.reset_index(drop=True)


def encode_features_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Dzieli dane po `match_id` (bez wycieku między train i test) i koduje cechy
    do macierzy rzadkiej CSR gotowej dla MLP, RandomForest i XGBoost.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["match_id"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # 1. Cechy kategoryczne: champion, position, tier, enemy_lane_champion
    cat_cols = ["champion", "position", "tier", "enemy_lane_champion"]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat_train = ohe.fit_transform(train_df[cat_cols])
    X_cat_test = ohe.transform(test_df[cat_cols])

    # 2. Cechy numeryczne: time_minutes, legendary_slot, has_boots
    num_cols = ["time_minutes", "legendary_slot", "has_boots"]
    scaler = StandardScaler()
    X_num_train = sparse.csr_matrix(scaler.fit_transform(train_df[num_cols].astype(float)))
    X_num_test = sparse.csr_matrix(scaler.transform(test_df[num_cols].astype(float)))

    # 3. Multi-hot: enemy_champions, ally_champions, prev_items
    all_champs = sorted(set(train_df["champion"].unique()) | set(train_df["enemy_lane_champion"].unique()))
    champ_to_idx = {c: i for i, c in enumerate(all_champs)}

    all_items = sorted(train_df["item_id"].unique())
    item_to_idx = {item: i for i, item in enumerate(all_items)}

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

    X_enemy_train = build_multihot(train_df["enemy_champions"], champ_to_idx)
    X_enemy_test = build_multihot(test_df["enemy_champions"], champ_to_idx)

    X_ally_train = build_multihot(train_df["ally_champions"], champ_to_idx)
    X_ally_test = build_multihot(test_df["ally_champions"], champ_to_idx)

    X_prev_train = build_multihot(train_df["prev_items"], item_to_idx)
    X_prev_test = build_multihot(test_df["prev_items"], item_to_idx)

    X_train = sparse.hstack([X_cat_train, X_num_train, X_enemy_train, X_ally_train, X_prev_train], format="csr")
    X_test = sparse.hstack([X_cat_test, X_num_test, X_enemy_test, X_ally_test, X_prev_test], format="csr")

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["item_id"])
    # Zachowujemy tylko wiersze testowe, których klasy wystąpiły w train
    known_mask = test_df["item_id"].isin(le.classes_)
    if not known_mask.all():
        test_df = test_df[known_mask].reset_index(drop=True)
        X_test = X_test[known_mask.values]
    y_test = le.transform(test_df["item_id"])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_df": train_df,
        "test_df": test_df,
        "label_encoder": le,
        "item_names": df.drop_duplicates("item_id").set_index("item_id")["name"].to_dict(),
    }


if __name__ == "__main__":
    df = build_legendary_purchase_dataset()
    print(f"Wiersze legendary: {len(df)}, unikalne przedmioty: {df['item_id'].nunique()}")
    bundle = encode_features_and_split(df)
    print(f"X_train: {bundle['X_train'].shape}, X_test: {bundle['X_test'].shape}")
    print(f"Liczba klas: {len(bundle['label_encoder'].classes_)}")
