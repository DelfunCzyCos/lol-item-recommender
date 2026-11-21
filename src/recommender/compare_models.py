"""
Porównanie algorytmów rekomendacji przedmiotów (Baseline Win Rate, Baseline Pick Rate,
RandomForest, XGBoost, MLP) na danych High Elo z uwzględnieniem matchupu i historii zakupów.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.recommender.ml_dataset import (
    build_legendary_purchase_dataset,
    encode_features_and_split,
)


OUTPUT_PATH = Path("data/processed/high_elo/recommendations/model_comparison.csv")


def compute_ranking_metrics(probs: np.ndarray, y_true: np.ndarray, prev_items_series, label_encoder):
    """
    Oblicza Top-1, Top-3, Top-5 Accuracy oraz MRR po odmaskowaniu przedmiotów,
    które gracz już posiada w ekwipunku (`prev_items`).
    """
    item_to_class = {int(item): idx for idx, item in enumerate(label_encoder.classes_)}
    masked_probs = probs.copy()

    for row_idx, owned in enumerate(prev_items_series):
        for item_id in owned:
            cls_idx = item_to_class.get(int(item_id))
            if cls_idx is not None:
                masked_probs[row_idx, cls_idx] = -1.0

    # Sortowanie klas malejąco po prawdopodobieństwie / wyniku
    ranked_indices = np.argsort(-masked_probs, axis=1)

    top1 = (ranked_indices[:, 0] == y_true).mean()
    top3 = np.any(ranked_indices[:, :3] == y_true[:, None], axis=1).mean()
    top5 = np.any(ranked_indices[:, :5] == y_true[:, None], axis=1).mean()

    # MRR (Mean Reciprocal Rank)
    ranks = np.argmax(ranked_indices == y_true[:, None], axis=1) + 1
    mrr = np.mean(1.0 / ranks)

    return {
        "top1_acc": float(top1),
        "top3_acc": float(top3),
        "top5_acc": float(top5),
        "mrr": float(mrr),
    }


def build_statistical_baseline_probs(train_df: pd.DataFrame, test_df: pd.DataFrame, label_encoder, mode: str):
    """
    Tworzy macierz wyników (n_test, n_classes) dla baseline'u statystycznego:
    - mode == 'win_rate': wygładzony win rate dla (champion, position, item_id)
    - mode == 'pick_rate': częstość wyboru dla (champion, position, item_id)
    """
    n_classes = len(label_encoder.classes_)
    item_to_class = {int(item): idx for idx, item in enumerate(label_encoder.classes_)}

    global_win_rate = train_df["win"].astype(float).mean()

    # Statystyki globalne (fallback)
    global_stats = (
        train_df.groupby("item_id")
        .agg(games=("win", "count"), wins=("win", "sum"))
        .reset_index()
    )
    global_stats["smoothed_wr"] = (global_stats["wins"] + 10 * global_win_rate) / (global_stats["games"] + 10)
    global_stats["pick_share"] = global_stats["games"] / len(train_df)

    global_vec = np.zeros(n_classes, dtype=np.float32)
    for row in global_stats.itertuples(index=False):
        cls_idx = item_to_class.get(int(row.item_id))
        if cls_idx is not None:
            global_vec[cls_idx] = row.smoothed_wr if mode == "win_rate" else row.pick_share

    # Statystyki dla (champion, position)
    cp_stats = (
        train_df.groupby(["champion", "position", "item_id"])
        .agg(games=("win", "count"), wins=("win", "sum"))
        .reset_index()
    )
    cp_totals = train_df.groupby(["champion", "position"]).size().to_dict()

    cp_vectors = {}
    for (champ, pos), group in cp_stats.groupby(["champion", "position"]):
        vec = global_vec.copy() * 0.01  # niski fallback dla nieznanych przedmiotów
        total_games = cp_totals.get((champ, pos), 1)
        for row in group.itertuples(index=False):
            cls_idx = item_to_class.get(int(row.item_id))
            if cls_idx is None:
                continue
            if mode == "win_rate":
                if row.games >= 5:
                    vec[cls_idx] = (row.wins + 10 * global_win_rate) / (row.games + 10)
                else:
                    vec[cls_idx] = 0.0
            else:
                vec[cls_idx] = row.games / total_games
        cp_vectors[(champ, pos)] = vec

    scores = np.zeros((len(test_df), n_classes), dtype=np.float32)
    for idx, row in enumerate(test_df[["champion", "position"]].itertuples(index=False)):
        scores[idx] = cp_vectors.get((row.champion, row.position), global_vec)

    return scores


def show_sample_predictions(test_df, probs_dict, label_encoder, item_names, n_samples: int = 3):
    """Wyświetla przykładowe porównanie rekomendacji dla wybranych sytuacji z testu."""
    item_to_class = {int(item): idx for idx, item in enumerate(label_encoder.classes_)}
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(test_df), size=min(n_samples, len(test_df)), replace=False)

    print("\n" + "=" * 75)
    print("PRZYKŁADOWE REKOMENDACJE NA ZBIORZE TESTOWYM (Z UWZGLĘDNIENIEM MATCHUPU)")
    print("=" * 75)

    for idx in sample_indices:
        row = test_df.iloc[idx]
        true_name = item_names.get(int(row["item_id"]), str(row["item_id"]))
        owned_names = [item_names.get(int(i), str(i)) for i in row["prev_items"]]
        print(
            f"\nGracz: {row['champion']} ({row['position']}) vs {row['enemy_lane_champion']} | "
            f"Slot: {row['legendary_slot']} | Czas: {row['time_minutes']:.1f} min"
        )
        print(f"Enemy team: {', '.join(row['enemy_champions'])}")
        print(f"Posiadane itemy: {', '.join(owned_names) if owned_names else 'brak'}")
        print(f"Rzeczywisty zakup (High Elo): {true_name}")

        for model_name, probs in probs_dict.items():
            vec = probs[idx].copy()
            for item_id in row["prev_items"]:
                cls_idx = item_to_class.get(int(item_id))
                if cls_idx is not None:
                    vec[cls_idx] = -1.0
            top3_idx = np.argsort(-vec)[:3]
            top3_names = [item_names.get(int(label_encoder.classes_[c]), "?") for c in top3_idx]
            print(f"  - {model_name:18s}: {', '.join(top3_names)}")


def main():
    print("[1/4] Budowanie zbioru danych z matchupem i historią zakupów...")
    t0 = time.perf_counter()
    df = build_legendary_purchase_dataset(min_item_count=20)
    bundle = encode_features_and_split(df, test_size=0.2, random_state=42)
    print(
        f"Gotowe w {time.perf_counter() - t0:.1f}s | "
        f"Train: {bundle['X_train'].shape[0]}, Test: {bundle['X_test'].shape[0]}, "
        f"Cechy: {bundle['X_train'].shape[1]}, Klasy: {len(bundle['label_encoder'].classes_)}"
    )

    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    train_df = bundle["train_df"]
    test_df = bundle["test_df"]
    le = bundle["label_encoder"]
    item_names = bundle["item_names"]

    results = []
    probs_for_examples = {}

    # 1. Baseline: Win Rate
    print("\n[2/4] Ewaluacja baseline'ów statystycznych...")
    t_start = time.perf_counter()
    wr_probs = build_statistical_baseline_probs(train_df, test_df, le, mode="win_rate")
    wr_time = time.perf_counter() - t_start
    wr_metrics = compute_ranking_metrics(wr_probs, y_test, test_df["prev_items"], le)
    results.append({"model": "Baseline: Win Rate", "train_sec": 0.0, "pred_sec": wr_time, **wr_metrics})
    probs_for_examples["Win Rate"] = wr_probs

    # 2. Baseline: Pick Rate
    t_start = time.perf_counter()
    pr_probs = build_statistical_baseline_probs(train_df, test_df, le, mode="pick_rate")
    pr_time = time.perf_counter() - t_start
    pr_metrics = compute_ranking_metrics(pr_probs, y_test, test_df["prev_items"], le)
    results.append({"model": "Baseline: Pick Rate", "train_sec": 0.0, "pred_sec": pr_time, **pr_metrics})

    # 3. Modele ML
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=25,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            batch_size=512,
            early_stopping=True,
            n_iter_no_change=5,
            max_iter=30,
            random_state=42,
        ),
    }

    print("\n[3/4] Trenowanie i ewaluacja modeli ML...")
    for name, model in models.items():
        print(f"  -> Trenowanie {name}...")
        t_train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_sec = time.perf_counter() - t_train_start

        t_pred_start = time.perf_counter()
        probs = model.predict_proba(X_test)
        pred_sec = time.perf_counter() - t_pred_start

        metrics = compute_ranking_metrics(probs, y_test, test_df["prev_items"], le)
        results.append({"model": name, "train_sec": train_sec, "pred_sec": pred_sec, **metrics})
        probs_for_examples[name] = probs
        print(
            f"     {name}: Top-1={metrics['top1_acc']:.1%}, "
            f"Top-3={metrics['top3_acc']:.1%}, Top-5={metrics['top5_acc']:.1%}, "
            f"MRR={metrics['mrr']:.3f} (train: {train_sec:.1f}s)"
        )

    print("\n[4/4] Podsumowanie wyników:")
    results_df = pd.DataFrame(results).sort_values("top1_acc", ascending=False)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\nZapisano wyniki do: {OUTPUT_PATH}")

    show_sample_predictions(test_df, probs_for_examples, le, item_names, n_samples=3)


if __name__ == "__main__":
    main()
