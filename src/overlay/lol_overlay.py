"""
Nakładka (overlay) w grze League of Legends wyświetlająca rekomendowane
przedmioty z modelu XGBoost na podstawie Live Client Data API.

Uruchomienie:
    python -m src.overlay.lol_overlay
    python -m src.overlay.lol_overlay --demo
"""

import argparse
from pathlib import Path
import sys
import threading
import tkinter as tk
from tkinter import messagebox
import traceback
import urllib3
import requests

from src.recommender.predictor import ItemRecommenderPredictor


LIVE_CLIENT_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
POLL_INTERVAL_MS = 2000

ROLE_DISPLAY_NAMES = {
    "TOP": "TOP",
    "JUNGLE": "JUNGLE",
    "MIDDLE": "MID",
    "BOTTOM": "BOT",
    "UTILITY": "SUPPORT",
}

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


DEMO_GAME_STATE = {
    "champion": "Vex",
    "position": "MIDDLE",
    "tier": "CHALLENGER",
    "enemy_lane_champion": "Ahri",
    "enemy_champions": ["Darius", "JarvanIV", "Ahri", "Jinx", "Thresh"],
    "ally_champions": ["KSante", "LeeSin", "Kaisa", "Nautilus"],
    "time_minutes": 9.5,
    "legendary_slot": 1,
    "has_boots": 0,
    "has_starter": 1,
    "prev_items": [],
    "all_owned_items": [1056, 1001, 3802],  # Pierścień Dorana + Zwykłe Buty + Zaginiony Rozdział
    "current_gold": 950,
}


def resolve_resource_dir(rel_path: str) -> Path:
    rel = Path(rel_path)
    if hasattr(sys, "_MEIPASS"):
        meipass_dir = Path(sys._MEIPASS) / rel
        if meipass_dir.exists():
            return meipass_dir
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent / rel
        if exe_dir.exists():
            return exe_dir
    return rel


class LoLItemOverlay:
    def __init__(self, demo_mode: bool = False):
        self.predictor = ItemRecommenderPredictor()
        self.icons_dir = resolve_resource_dir("assets/item_icons")
        self.champ_icons_dir = resolve_resource_dir("assets/champion_icons")
        self.icon_cache: dict[int, tk.PhotoImage] = {}
        self.champ_icon_cache: dict[str, tk.PhotoImage] = {}
        self.champ_file_map = {
            p.stem.lower(): p
            for p in (self.champ_icons_dir.glob("*.png") if self.champ_icons_dir.exists() else [])
        }
        self.demo_mode = demo_mode
        self.collapsed = False
        self.has_boots_section = False
        self._drag_x = 0
        self._drag_y = 0

        self.root = tk.Tk()
        self.root.title("LoL Item Recommender (XGBoost)")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.90)
        self.root.configure(bg="#0F172A")
        self.root.geometry("320x245+30+120")

        self._build_ui()
        self._schedule_poll()

    def _build_ui(self):
        # Pasek tytułowy (do przeciągania)
        self.header = tk.Frame(self.root, bg="#1E293B", height=24)
        self.header.pack(fill="x")
        self.header.bind("<Button-1>", self._start_drag)
        self.header.bind("<B1-Motion>", self._on_drag)

        self.title_lbl = tk.Label(
            self.header,
            text="LoL Recommender",
            fg="#F8FAFC",
            bg="#1E293B",
            font=("Segoe UI", 8, "bold"),
            padx=6,
        )
        self.title_lbl.pack(side="left", pady=2)
        self.title_lbl.bind("<Button-1>", self._start_drag)
        self.title_lbl.bind("<B1-Motion>", self._on_drag)

        close_btn = tk.Button(
            self.header,
            text="×",
            command=self.root.destroy,
            fg="#CBD5E1",
            bg="#1E293B",
            activebackground="#EF4444",
            activeforeground="#FFFFFF",
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=8,
            cursor="hand2",
        )
        close_btn.pack(side="right")

        collapse_btn = tk.Button(
            self.header,
            text="—",
            command=self._toggle_collapse,
            fg="#CBD5E1",
            bg="#1E293B",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            bd=0,
            font=("Segoe UI", 9, "bold"),
            padx=6,
            cursor="hand2",
        )
        collapse_btn.pack(side="right")

        self.demo_btn = tk.Button(
            self.header,
            text="Demo",
            command=self._toggle_demo,
            fg="#93C5FD" if self.demo_mode else "#94A3B8",
            bg="#1E293B",
            activebackground="#334155",
            activeforeground="#FFFFFF",
            bd=0,
            font=("Segoe UI", 8, "bold"),
            padx=6,
            cursor="hand2",
        )
        self.demo_btn.pack(side="right")

        # Główna zawartość
        self.body = tk.Frame(self.root, bg="#0F172A", padx=6, pady=4)
        self.body.pack(fill="both", expand=True)

        self.matchup_bar = tk.Frame(self.body, bg="#0F172A")
        self.matchup_bar.pack(fill="x", pady=(0, 4))

        self.my_champ_icon = tk.Label(self.matchup_bar, bg="#0F172A")
        self.vs_lbl = tk.Label(
            self.matchup_bar,
            text="vs",
            fg="#64748B",
            bg="#0F172A",
            font=("Segoe UI", 8, "bold"),
        )
        self.enemy_champ_icon = tk.Label(self.matchup_bar, bg="#0F172A")

        self.status_lbl = tk.Label(
            self.matchup_bar,
            text="Oczekiwanie na mecz LoL...",
            fg="#94A3B8",
            bg="#0F172A",
            font=("Segoe UI", 8),
            anchor="w",
            justify="left",
        )
        self.status_lbl.pack(side="left", fill="x", expand=True)

        self.primary_title_lbl = tk.Label(
            self.body,
            text="LEGENDARNE",
            fg="#64748B",
            bg="#0F172A",
            font=("Segoe UI", 7, "bold"),
            anchor="w",
        )
        self.primary_title_lbl.pack(fill="x", pady=(0, 1))

        self.cards_frame = tk.Frame(self.body, bg="#0F172A")
        self.cards_frame.pack(fill="x")

        self.item_rows = self._create_card_rows(self.cards_frame, count=3)

        self.boots_section = tk.Frame(self.body, bg="#0F172A")
        self.boots_title_lbl = tk.Label(
            self.boots_section,
            text="BUTY",
            fg="#64748B",
            bg="#0F172A",
            font=("Segoe UI", 7, "bold"),
            anchor="w",
        )
        self.boots_title_lbl.pack(fill="x", pady=(4, 1))
        self.boots_rows = self._create_card_rows(self.boots_section, count=2)

    def _create_card_rows(self, parent: tk.Frame, count: int) -> list:
        rows = []
        for idx in range(count):
            card = tk.Frame(parent, bg="#1E293B", padx=6, pady=3)
            card.pack(fill="x", pady=2)

            icon_lbl = tk.Label(card, bg="#0F172A", width=32, height=32)
            icon_lbl.pack(side="left", padx=(0, 6))

            info_frame = tk.Frame(card, bg="#1E293B")
            info_frame.pack(side="left", fill="both", expand=True)

            top_row = tk.Frame(info_frame, bg="#1E293B")
            top_row.pack(fill="x")

            name_lbl = tk.Label(
                top_row,
                text=f"{idx + 1}. —",
                fg="#F1F5F9",
                bg="#1E293B",
                font=("Segoe UI", 9, "bold"),
                anchor="w",
            )
            name_lbl.pack(side="left", fill="x", expand=True)

            prob_lbl = tk.Label(
                top_row,
                text="",
                fg="#38BDF8",
                bg="#1E293B",
                font=("Segoe UI", 9, "bold"),
            )
            prob_lbl.pack(side="right")

            sub_lbl = tk.Label(
                info_frame,
                text="",
                fg="#94A3B8",
                bg="#1E293B",
                font=("Segoe UI", 8),
                anchor="w",
            )
            sub_lbl.pack(fill="x")

            rows.append((icon_lbl, name_lbl, prob_lbl, sub_lbl))
        return rows

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _update_window_height(self):
        if self.collapsed:
            self.root.geometry("320x24")
        elif self.has_boots_section:
            self.root.geometry("320x360")
        else:
            self.root.geometry("320x245")

    def _toggle_collapse(self):
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.body.pack_forget()
        else:
            self.body.pack(fill="both", expand=True)
        self._update_window_height()

    def _toggle_demo(self):
        self.demo_mode = not self.demo_mode
        self.demo_btn.configure(fg="#93C5FD" if self.demo_mode else "#94A3B8")
        self._poll_once()

    def _schedule_poll(self):
        self._poll_once()
        self.root.after(POLL_INTERVAL_MS, self._schedule_poll)

    def _poll_once(self):
        if self.demo_mode:
            rec_bundle = self.predictor.predict_all(DEMO_GAME_STATE, top_k=3, boots_k=2)
            self._update_display(DEMO_GAME_STATE, rec_bundle, is_demo=True)
            return

        worker = threading.Thread(target=self._fetch_and_predict, daemon=True)
        worker.start()

    def _fetch_and_predict(self):
        try:
            resp = requests.get(LIVE_CLIENT_URL, timeout=1.2, verify=False)
            if resp.status_code != 200:
                self.root.after(0, self._show_waiting, f"Klient LoL zwrócił status {resp.status_code}")
                return

            data = resp.json()
            game_state = self.predictor.parse_live_client_data(data)
            rec_bundle = self.predictor.predict_all(game_state, top_k=3, boots_k=2)
            self.root.after(0, self._update_display, game_state, rec_bundle, False)
        except requests.RequestException:
            self.root.after(
                0,
                self._show_waiting,
                "Brak aktywnego meczu LoL (127.0.0.1:2999). Kliknij 'Demo', aby przetestować.",
            )
        except Exception as exc:
            self.root.after(0, self._show_waiting, f"Błąd odczytu: {exc}")

    def _get_item_icon(self, item_id: int) -> tk.PhotoImage | None:
        if item_id in self.icon_cache:
            return self.icon_cache[item_id]
        icon_path = self.icons_dir / f"{item_id}.png"
        if not icon_path.exists():
            return None
        try:
            img = tk.PhotoImage(file=str(icon_path)).subsample(2, 2)
            self.icon_cache[item_id] = img
            return img
        except tk.TclError:
            return None

    def _get_champ_icon(self, champ_name: str) -> tk.PhotoImage | None:
        if not champ_name or champ_name == "UNKNOWN":
            return None
        key = champ_name.lower()
        if key in self.champ_icon_cache:
            return self.champ_icon_cache[key]
        icon_path = self.champ_file_map.get(key)
        if not icon_path or not icon_path.exists():
            return None
        try:
            img = tk.PhotoImage(file=str(icon_path)).subsample(5, 5)
            self.champ_icon_cache[key] = img
            return img
        except tk.TclError:
            return None

    def _set_matchup_icons(self, my_champ: str | None, enemy_champ: str | None):
        self.my_champ_icon.pack_forget()
        self.vs_lbl.pack_forget()
        self.enemy_champ_icon.pack_forget()
        self.status_lbl.pack_forget()

        my_img = self._get_champ_icon(my_champ) if my_champ else None
        enemy_img = self._get_champ_icon(enemy_champ) if enemy_champ else None

        if my_img is not None:
            self.my_champ_icon.configure(image=my_img)
            self.my_champ_icon.image = my_img
            self.my_champ_icon.pack(side="left", padx=(0, 4))

        if enemy_img is not None:
            if my_img is not None:
                self.vs_lbl.pack(side="left", padx=(0, 4))
            self.enemy_champ_icon.configure(image=enemy_img)
            self.enemy_champ_icon.image = enemy_img
            self.enemy_champ_icon.pack(side="left", padx=(0, 8))

        self.status_lbl.pack(side="left", fill="x", expand=True)

    def _clear_rows(self, rows: list):
        for idx, (icon_lbl, name_lbl, prob_lbl, sub_lbl) in enumerate(rows):
            icon_lbl.configure(image="", width=32, height=32)
            icon_lbl.image = None
            name_lbl.configure(text=f"{idx + 1}. —")
            prob_lbl.configure(text="")
            sub_lbl.configure(text="")

    def _render_rows(self, rows: list, recs: list[dict]):
        for idx, (icon_lbl, name_lbl, prob_lbl, sub_lbl) in enumerate(rows):
            if idx >= len(recs):
                icon_lbl.configure(image="", width=32, height=32)
                icon_lbl.image = None
                name_lbl.configure(text=f"{idx + 1}. —")
                prob_lbl.configure(text="")
                sub_lbl.configure(text="")
                continue

            rec = recs[idx]
            icon = self._get_item_icon(rec["item_id"])
            if icon is not None:
                icon_lbl.configure(image=icon, width=32, height=32)
                icon_lbl.image = icon
            else:
                icon_lbl.configure(image="", width=32, height=32)
                icon_lbl.image = None

            name_lbl.configure(text=f"{idx + 1}. {rec['name']}")
            prob_lbl.configure(text=f"{rec['probability']:.1%}")

            effective_price = rec.get("effective_price", rec["price"])
            if effective_price > 0:
                if rec["can_afford"]:
                    sub_lbl.configure(text="STAĆ CIĘ", fg="#4ADE80")
                else:
                    sub_lbl.configure(
                        text=f"Brakuje {rec['gold_missing']}g",
                        fg="#FBBF24",
                    )
            else:
                sub_lbl.configure(text="", fg="#94A3B8")

    def _show_waiting(self, msg: str):
        self._set_matchup_icons(None, None)
        self.status_lbl.configure(text=msg, fg="#94A3B8")
        self.primary_title_lbl.configure(text="LEGENDARNE")
        self._clear_rows(self.item_rows)
        self._clear_rows(self.boots_rows)
        self.boots_section.pack_forget()
        self.has_boots_section = False
        self._update_window_height()

    def _update_display(self, game_state: dict, rec_bundle: dict, is_demo: bool):
        self._set_matchup_icons(game_state.get("champion"), game_state.get("enemy_lane_champion"))
        prefix = "[DEMO] " if is_demo else ""
        pos_display = ROLE_DISPLAY_NAMES.get(str(game_state["position"]).upper(), game_state["position"])
        status_text = (
            f"{prefix}{game_state['champion']} ({pos_display}) "
            f"vs {game_state['enemy_lane_champion']}\n"
            f"{game_state['time_minutes']:.1f} min  •  Złoto: {int(game_state['current_gold'])}g"
        )
        self.status_lbl.configure(text=status_text, fg="#E2E8F0")

        self.primary_title_lbl.configure(text=rec_bundle.get("primary_title", "LEGENDARNE"))
        self._render_rows(self.item_rows, rec_bundle.get("primary", []))

        boots = rec_bundle.get("boots", [])
        if boots:
            self._render_rows(self.boots_rows, boots)
            if not self.has_boots_section:
                self.boots_section.pack(fill="x")
                self.has_boots_section = True
        else:
            if self.has_boots_section:
                self.boots_section.pack_forget()
                self.has_boots_section = False

        self._update_window_height()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Nakładka LoL Item Recommender (XGBoost)")
    parser.add_argument("--demo", action="store_true", help="Uruchom z przykładowym stanem gry")
    args = parser.parse_args()

    try:
        app = LoLItemOverlay(demo_mode=args.demo)
        app.run()
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Błąd uruchomienia LoL Recommender",
            f"{exc}\n\n{traceback.format_exc()}",
        )
        root.destroy()


if __name__ == "__main__":
    main()
