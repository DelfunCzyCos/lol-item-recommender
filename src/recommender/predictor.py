"""
Moduł predykcyjny dla zapisanego modelu XGBoost oraz parser danych
z Riot Live Client Data API (https://127.0.0.1:2999/liveclientdata/allgamedata).
"""

from collections import Counter
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


def get_default_model_path() -> Path:
    """Znajduje ścieżkę do modelu zarówno w zwykłym Pythonie, jak i w paczce PyInstaller."""
    rel = Path("models/xgb_recommender.joblib")
    if hasattr(sys, "_MEIPASS"):
        meipass_path = Path(sys._MEIPASS) / rel
        if meipass_path.exists():
            return meipass_path
    if getattr(sys, "frozen", False):
        exe_dir_path = Path(sys.executable).resolve().parent / rel
        if exe_dir_path.exists():
            return exe_dir_path
    return rel


DEFAULT_MODEL_PATH = get_default_model_path()

# Najczęstsze rozbieżności między nazwą wyświetlaną a ID postaci w Riot API
CHAMPION_ALIASES = {
    "wukong": "MonkeyKing",
    "nunuwillump": "Nunu",
    "renataglasc": "Renata",
    "fiddlesticks": "FiddleSticks",
}


class ItemRecommenderPredictor:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono modelu: {model_path}. "
                "Uruchom najpierw: python -m src.recommender.train_xgb"
            )
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.ohe = self.bundle["ohe"]
        self.scaler = self.bundle["scaler"]
        self.champ_to_idx = self.bundle["champ_to_idx"]
        self.item_to_idx = self.bundle["item_to_idx"]
        self.le = self.bundle["label_encoder"]
        self.item_metadata = self.bundle["item_metadata"]
        self.boots_ids = set(self.bundle["boots_ids"])
        self.legendary_ids = set(self.bundle["legendary_ids"])
        self.starter_ids = set(self.bundle.get("starter_ids", set()))
        self.starter_stats = self.bundle.get("starter_stats", {})
        self.boots_stats = self.bundle.get("boots_stats", {})
        self.boots_matchup_stats = self.bundle.get("boots_matchup_stats", {})
        self.item_recipes = self.bundle.get("item_recipes", {})

        # Mapa znormalizowanych nazw bohaterów (lowercase, bez spacji/apostrofów)
        self.norm_to_champ = {}
        for champ in self.bundle["known_champions"]:
            norm = re.sub(r"[^a-z0-9]", "", champ.lower())
            self.norm_to_champ[norm] = champ

    def normalize_champion_name(self, name: str) -> str:
        if not name:
            return "UNKNOWN"
        # Jeśli pochodzi z rawChampionName, np. "game_character_displayname_Aatrox"
        cleaned = name.replace("game_character_displayname_", "").strip()
        if cleaned in self.champ_to_idx:
            return cleaned

        norm = re.sub(r"[^a-z0-9]", "", cleaned.lower())
        if norm in CHAMPION_ALIASES:
            return CHAMPION_ALIASES[norm]
        return self.norm_to_champ.get(norm, cleaned)

    def _encode_single(self, game_state: dict):
        cat_df = pd.DataFrame([{
            "champion": game_state.get("champion", "UNKNOWN"),
            "position": game_state.get("position", "UNKNOWN") or "UNKNOWN",
            "tier": game_state.get("tier", "CHALLENGER"),
            "enemy_lane_champion": game_state.get("enemy_lane_champion", "UNKNOWN"),
        }])
        X_cat = self.ohe.transform(cat_df)

        num_df = pd.DataFrame([{
            "time_minutes": float(game_state.get("time_minutes", 10.0)),
            "legendary_slot": float(game_state.get("legendary_slot", 1)),
            "has_boots": float(game_state.get("has_boots", 0)),
        }])
        X_num = sparse.csr_matrix(self.scaler.transform(num_df))

        def multihot_single(values, mapping):
            cols = [mapping[v] for v in values if v in mapping]
            rows = [0] * len(cols)
            data = np.ones(len(cols), dtype=np.float32)
            return sparse.csr_matrix((data, (rows, cols)), shape=(1, len(mapping)))

        X_enemy = multihot_single(game_state.get("enemy_champions", []), self.champ_to_idx)
        X_ally = multihot_single(game_state.get("ally_champions", []), self.champ_to_idx)
        X_prev = multihot_single(game_state.get("prev_items", []), self.item_to_idx)

        return sparse.hstack([X_cat, X_num, X_enemy, X_ally, X_prev], format="csr")

    def compute_remaining_cost(
        self, target_item_id: int, owned_items: list[int], fallback_price: int = 0
    ) -> int:
        """Oblicza brakujący koszt przedmiotu z uwzględnieniem posiadanych komponentów."""
        if not self.item_recipes or int(target_item_id) not in self.item_recipes:
            return fallback_price

        available = Counter(int(i) for i in owned_items)

        def _cost(iid: int) -> int:
            if available[iid] > 0:
                available[iid] -= 1
                return 0
            node = self.item_recipes.get(iid)
            if not node:
                return 0
            children = node.get("from", [])
            if not children:
                return int(node.get("total", 0))
            total = int(node.get("base", 0))
            for child_id in children:
                total += _cost(int(child_id))
            return total

        return _cost(int(target_item_id))

    def _build_rec_entry(
        self, item_id: int, prob: float, current_gold: float, all_owned: list[int], category: str
    ) -> dict:
        meta = self.item_metadata.get(item_id, {})
        full_price = int(meta.get("price", 0))
        effective_price = self.compute_remaining_cost(item_id, all_owned, full_price)
        return {
            "item_id": item_id,
            "name": meta.get("name", f"Item {item_id}"),
            "category": category,
            "price": full_price,
            "effective_price": effective_price,
            "probability": prob,
            "can_afford": current_gold >= effective_price if effective_price > 0 else False,
            "gold_missing": max(0, effective_price - int(current_gold)) if effective_price > 0 else 0,
        }

    def _lookup_boots(self, champion: str, position: str, enemy_lane: str) -> list[tuple[int, float]]:
        if (champion, position, enemy_lane) in self.boots_matchup_stats:
            return self.boots_matchup_stats[(champion, position, enemy_lane)]
        if (champion, position) in self.boots_stats:
            return self.boots_stats[(champion, position)]
        for (c, _), items in self.boots_stats.items():
            if c == champion:
                return items
        return []

    def _lookup_starters(self, champion: str, position: str) -> list[tuple[int, float]]:
        if (champion, position) in self.starter_stats:
            return self.starter_stats[(champion, position)]
        for (c, _), items in self.starter_stats.items():
            if c == champion:
                return items
        return []

    def predict_boots(self, game_state: dict, top_k: int = 2) -> list[dict]:
        """Osobny rekomender dla butów (z uwzględnieniem matchupu i posiadanych komponentów)."""
        if int(game_state.get("has_boots", 0)):
            return []
        current_gold = float(game_state.get("current_gold", 0.0))
        all_owned = game_state.get("all_owned_items", game_state.get("prev_items", []))
        champion = game_state.get("champion", "UNKNOWN")
        position = game_state.get("position", "MIDDLE") or "MIDDLE"
        enemy_lane = game_state.get("enemy_lane_champion", "UNKNOWN")

        candidates = self._lookup_boots(champion, position, enemy_lane)
        return [
            self._build_rec_entry(iid, prob, current_gold, all_owned, "boots")
            for iid, prob in candidates[:top_k]
        ]

    def predict(self, game_state: dict, top_k: int = 3) -> list[dict]:
        """Zwraca Top-K rekomendowanych starterów (na początku gry) lub przedmiotów legendary."""
        current_gold = float(game_state.get("current_gold", 0.0))
        all_owned = game_state.get("all_owned_items", game_state.get("prev_items", []))
        champion = game_state.get("champion", "UNKNOWN")
        position = game_state.get("position", "MIDDLE") or "MIDDLE"
        time_minutes = float(game_state.get("time_minutes", 10.0))
        has_starter = int(game_state.get("has_starter", 0))
        prev_items = game_state.get("prev_items", [])

        # 1. Początek gry: rekomendacja startera
        if not has_starter and len(prev_items) == 0 and time_minutes < 3.0:
            starters = self._lookup_starters(champion, position)
            if starters:
                return [
                    self._build_rec_entry(iid, prob, current_gold, all_owned, "starter")
                    for iid, prob in starters[:top_k]
                ]

        # 2. Predykcja przedmiotów legendary z XGBoost
        X = self._encode_single(game_state)
        probs = self.model.predict_proba(X)[0].copy()

        item_to_class = {int(item): idx for idx, item in enumerate(self.le.classes_)}
        for owned_id in prev_items:
            cls_idx = item_to_class.get(int(owned_id))
            if cls_idx is not None:
                probs[cls_idx] = -1.0

        top_indices = np.argsort(-probs)[:top_k]

        recommendations = []
        for cls_idx in top_indices:
            prob = float(probs[cls_idx])
            if prob < 0:
                continue
            item_id = int(self.le.classes_[cls_idx])
            recommendations.append(
                self._build_rec_entry(item_id, prob, current_gold, all_owned, "legendary")
            )

        return recommendations

    def predict_all(self, game_state: dict, top_k: int = 3, boots_k: int = 2) -> dict:
        """Zwraca osobne rekomendacje dla głównego buildu (starter/legendary) oraz dla butów."""
        primary = self.predict(game_state, top_k=top_k)
        is_starter_phase = bool(primary and primary[0].get("category") == "starter")
        boots = [] if is_starter_phase else self.predict_boots(game_state, top_k=boots_k)
        return {
            "primary_title": "STARTERY" if is_starter_phase else "LEGENDARNE",
            "primary": primary,
            "boots": boots,
        }

    def parse_live_client_data(self, data: dict) -> dict:
        """Konwertuje JSON z /liveclientdata/allgamedata na format game_state."""
        active_player = data.get("activePlayer", {})
        all_players = data.get("allPlayers", [])
        game_data = data.get("gameData", {})

        active_riot_id = active_player.get("riotId", "")
        active_name = active_player.get("summonerName", "")
        current_gold = float(active_player.get("currentGold", 0.0))

        # Odnalezienie aktywnego gracza na liście allPlayers
        me = None
        for p in all_players:
            if active_riot_id and p.get("riotId") == active_riot_id:
                me = p
                break
            if active_name and p.get("summonerName") == active_name:
                me = p
                break

        if me is None and all_players:
            me = all_players[0]

        my_team = me.get("team", "ORDER") if me else "ORDER"
        my_champ = self.normalize_champion_name(
            (me.get("rawChampionName") or me.get("championName", "")) if me else ""
        )
        my_pos = (me.get("position") or "MIDDLE").upper() if me else "MIDDLE"
        if my_pos in ("", "NONE"):
            my_pos = "MIDDLE"

        enemy_champions = []
        ally_champions = []
        enemy_lane = "UNKNOWN"

        for p in all_players:
            champ_name = self.normalize_champion_name(
                p.get("rawChampionName") or p.get("championName", "")
            )
            p_pos = (p.get("position") or "").upper()
            if p.get("team") == my_team:
                if p is not me:
                    ally_champions.append(champ_name)
            else:
                enemy_champions.append(champ_name)
                if p_pos == my_pos:
                    enemy_lane = champ_name

        # Odczyt posiadanych przedmiotów z ekwipunku (w tym komponentów i starterów)
        owned_items = me.get("items", []) if me else []
        owned_legendary = []
        all_owned_items = []
        has_boots = 0
        has_starter = 0
        for item in owned_items:
            iid = int(item.get("itemID", 0))
            count = max(1, int(item.get("count", 1)))
            all_owned_items.extend([iid] * count)
            if iid in self.boots_ids:
                has_boots = 1
            if iid in self.starter_ids or (3860 <= iid <= 3869):
                has_starter = 1
            if iid in self.legendary_ids:
                owned_legendary.append(iid)

        game_time_sec = float(game_data.get("gameTime", 0.0))

        return {
            "champion": my_champ,
            "position": my_pos,
            "tier": "CHALLENGER",
            "enemy_lane_champion": enemy_lane,
            "enemy_champions": enemy_champions,
            "ally_champions": ally_champions,
            "time_minutes": game_time_sec / 60.0,
            "legendary_slot": len(owned_legendary) + 1,
            "has_boots": has_boots,
            "has_starter": has_starter,
            "prev_items": owned_legendary,
            "all_owned_items": all_owned_items,
            "current_gold": current_gold,
        }
