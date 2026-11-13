import os
import time
import requests
from pathlib import Path
from urllib.parse import quote
from dotenv import load_dotenv

# Ścieżka do głównego folderu projektu
BASE_DIR = Path(__file__).resolve().parents[2]

# Wczytanie .env z głównego folderu projektu
load_dotenv(BASE_DIR / ".env", override=True)


class RiotAPI:

    def __init__(self):
        self.api_key = os.getenv("RIOT_API_KEY")

        if not self.api_key:
            raise ValueError("Brak RIOT_API_KEY w pliku .env")

        self.base_url = "https://europe.api.riotgames.com"
        self.platform_url = "https://eun1.api.riotgames.com"

        self.session = requests.Session()
        self.session.headers.update({
            "X-Riot-Token": self.api_key
        })

        self.last_request_time = 0

    def request(self, endpoint, params=None, platform=False):
        base_url = self.platform_url if platform else self.base_url
        url = base_url + endpoint

        # Ostrożne tempo dla klucza deweloperskiego:
        # około 96 zapytań na 2 minuty.
        elapsed = time.monotonic() - self.last_request_time
        if elapsed < 1.25:
            time.sleep(1.25 - elapsed)

        for attempt in range(3):
            self.last_request_time = time.monotonic()

            response = self.session.get(
                url,
                params=params,
                timeout=30
            )

            if response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", 10))
                print(f"Limit API. Czekam {wait_time} sekund...")
                time.sleep(wait_time + 1)
                continue

            response.raise_for_status()
            return response.json()

        raise Exception("Nie udało się wykonać zapytania po kilku próbach.")

    def get_ranked_league(self, tier):
        endpoints = {
            "CHALLENGER": "/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5",
            "GRANDMASTER": "/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5",
            "MASTER": "/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5",
        }

        if tier not in endpoints:
            raise ValueError(f"Nieobsługiwana ranga: {tier}")

        return self.request(endpoints[tier], platform=True)

    def get_summoner_by_id(self, summoner_id):
        summoner_id = quote(summoner_id, safe="")
        endpoint = f"/lol/summoner/v4/summoners/{summoner_id}"

        return self.request(endpoint, platform=True)

    def get_match_ids(self, puuid, count=10, queue=420):
        puuid = quote(puuid, safe="")
        endpoint = f"/lol/match/v5/matches/by-puuid/{puuid}/ids"

        params = {
            "start": 0,
            "count": count,
            "queue": queue
        }

        return self.request(endpoint, params=params)

    def get_match(self, match_id):
        match_id = quote(match_id, safe="")
        endpoint = f"/lol/match/v5/matches/{match_id}"

        return self.request(endpoint)

    
    def get_matches(self, match_ids):
        matches = []

        for index, match_id in enumerate(match_ids, start=1):
            print(f"Pobieranie meczu {index}/{len(match_ids)}: {match_id}")

            try:
                matches.append(self.get_match(match_id))
            except requests.RequestException as error:
                print(f"Błąd pobierania meczu {match_id}: {error}")

        return matches

    def get_match_timeline(self, match_id):
        match_id = quote(match_id, safe="")
        endpoint = f"/lol/match/v5/matches/{match_id}/timeline"

        return self.request(endpoint)


