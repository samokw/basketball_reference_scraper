import argparse
import os
import time
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

# --- Config ---
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_TIMEOUT = 30
PAUSE_SECONDS = 3  # be polite to the site


# --- Helpers ---
def season_label(end_year: int) -> str:
    return f"{end_year-1}-{str(end_year)[-2:]}"


def fetch_html(url: str) -> Optional[str]:
    print(f"Fetching: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  ERROR: request failed: {e}")
        return None


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    def clean_parts(parts):
        out = []
        for p in parts:
            s = str(p)
            if s.lower() == "nan" or s.strip() == "":
                continue
            if s.startswith("Unnamed"):
                # ignore placeholder level names like 'Unnamed: 2_level_0'
                continue
            out.append(s)
        return out

    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for tup in df.columns:
            parts = clean_parts(list(tup))
            if len(parts) == 0:
                # fallback to raw join if everything was 'Unnamed' or blank
                parts = [str(x) for x in tup if str(x).lower() != "nan"]
            col = "_".join(parts).strip("_")
            new_cols.append(col)
        df.columns = new_cols
    else:
        df.columns = [str(c) for c in df.columns]
    # After intelligent flattening, keep columns even if they originally had 'Unnamed'
    # but still drop any residual fully empty names
    df = df.loc[:, df.columns.astype(str).str.strip() != ""]
    return df


def _drop_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Basketball Reference sometimes repeats header rows within the table body
    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str) != "Rk"]
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df)
    df = _drop_repeated_header_rows(df)
    df = df.reset_index(drop=True)
    return df


def read_table_from_html(html: str, table_id: str, url_for_fallback: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Try reading a <table id=...> even if Basketball Reference placed it inside an HTML comment.
    If not found, optionally fall back to scanning all tables from the URL and heuristically selecting the target.
    """
    # First try: direct parse of the full HTML
    try:
        tables = pd.read_html(html, attrs={"id": table_id})
        if tables:
            return tables[0]
    except ValueError:
        pass
    except Exception as e:
        print(f"  WARN: pd.read_html(full) failed for '{table_id}': {e}")

    # Second try: pull the exact table block via BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    table_tag = soup.find("table", id=table_id)
    if table_tag is not None:
        try:
            tables = pd.read_html(str(table_tag))
            if tables:
                return tables[0]
        except Exception as e:
            print(f"  WARN: pd.read_html(direct table) failed for '{table_id}': {e}")

    # Third try: scan HTML comments, Basketball Reference often comments out tables
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        if table_id in c:
            try:
                tables = pd.read_html(str(c), attrs={"id": table_id})
                if tables:
                    return tables[0]
            except Exception:
                continue

    print(f"  WARN: table id='{table_id}' not found by-id")

    # Final fallback: read all tables from the URL and pick one with expected columns
    if url_for_fallback:
        try:
            all_tables = pd.read_html(url_for_fallback)
            # Prefer tables that contain typical columns
            priorities = [
                {"Player", "Tm"},
                {"Team", "W"},
                {"Rank", "Player"},
            ]
            for want in priorities:
                for t in all_tables:
                    cols = set([str(c) for c in t.columns])
                    if want.issubset(cols):
                        return t
            # Or just return the first non-empty
            for t in all_tables:
                if not t.empty:
                    return t
        except Exception as e:
            print(f"  WARN: fallback read_html(url) failed: {e}")

    return None


# --- Scrapers ---

def scrape_player_stats_for_season(year: int, out_dir: str) -> None:
    print(f"\n[Players] Season {season_label(year)} ({year})")
    endpoints = {
        "totals": (f"https://www.basketball-reference.com/leagues/NBA_{year}_totals.html", "totals_stats"),
        "per_game": (f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html", "per_game_stats"),
        "per_poss": (f"https://www.basketball-reference.com/leagues/NBA_{year}_per_poss.html", "per_poss_stats"),
        "advanced": (f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html", "advanced_stats"),

    }
    os.makedirs(out_dir, exist_ok=True)

    for label, (url, table_id) in endpoints.items():
        print(f"  -> {label}: downloading")
        html = fetch_html(url)
        if not html:
            print(f"  -> {label}: skipped (no HTML)")
            time.sleep(PAUSE_SECONDS)
            continue

        df = read_table_from_html(html, table_id, url)
        if df is None or df.empty:
            print(f"  -> {label}: no data")
            time.sleep(PAUSE_SECONDS)
            continue

        df = clean_df(df)
        df["season_end_year"] = year
        df["season"] = season_label(year)
        out_path = os.path.join(out_dir, f"players_{label}.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> {label}: saved {out_path} ({len(df)} rows)")
        time.sleep(PAUSE_SECONDS)


def scrape_standings_for_season(year: int, out_dir: str) -> None:
    print(f"\n[Standings] Season {season_label(year)} ({year})")
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
    os.makedirs(out_dir, exist_ok=True)

    html = fetch_html(url)
    if not html:
        print("  -> standings: skipped (no HTML)")
        return

    east = read_table_from_html(html, "confs_standings_E", url)
    west = read_table_from_html(html, "confs_standings_W", url)

    if (east is None or east.empty) and (west is None or west.empty):
        print("  -> standings: no data")
        return

    # Normalize columns: first column is conference name header; rename to 'Team'
    def normalize_conf(df: pd.DataFrame, conf_name: str) -> pd.DataFrame:
        df = clean_df(df)
        # Rename first column if it's named after the conference
        cols = list(df.columns)
        if cols:
            if cols[0].lower().startswith("eastern conference") or cols[0].lower().startswith("western conference"):
                df = df.rename(columns={cols[0]: "Team"})
        df["Conference"] = conf_name
        df["season_end_year"] = year
        df["season"] = season_label(year)
        return df

    east_norm = normalize_conf(east, "East") if east is not None and not east.empty else None
    west_norm = normalize_conf(west, "West") if west is not None and not west.empty else None

    # Save separate files
    if east_norm is not None:
        east_path = os.path.join(out_dir, "standings_east.csv")
        east_norm.to_csv(east_path, index=False)
        print(f"  -> standings_east: saved {east_path} ({len(east_norm)} rows)")
    if west_norm is not None:
        west_path = os.path.join(out_dir, "standings_west.csv")
        west_norm.to_csv(west_path, index=False)
        print(f"  -> standings_west: saved {west_path} ({len(west_norm)} rows)")

    # Also save combined tidy file for convenience
    frames = [f for f in [east_norm, west_norm] if f is not None]
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out_path = os.path.join(out_dir, "standings.csv")
        out.to_csv(out_path, index=False)
        print(f"  -> standings(all): saved {out_path} ({len(out)} rows)")


def scrape_mvp_voting_for_season(year: int, out_dir: str) -> None:
    print(f"\n[MVP Voting] Season {season_label(year)} ({year})")
    url = f"https://www.basketball-reference.com/awards/awards_{year}.html"
    os.makedirs(out_dir, exist_ok=True)

    html = fetch_html(url)
    if not html:
        print("  -> mvp: skipped (no HTML)")
        return

    df = read_table_from_html(html, "mvp", url)
    if df is None or df.empty:
        print("  -> mvp: no data")
        return

    df = clean_df(df)
    df["season_end_year"] = year
    df["season"] = season_label(year)
    out_path = os.path.join(out_dir, "mvp_voting.csv")
    df.to_csv(out_path, index=False)
    print(f"  -> mvp: saved {out_path} ({len(df)} rows)")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Basketball Reference scraper (simple)")
    parser.add_argument("--start-year", type=int, default=2009, help="First season end year (e.g., 2009)")
    parser.add_argument("--end-year", type=int, default=2024, help="Last season end year (e.g., 2024)")
    args = parser.parse_args()

    start, end = args.start_year, args.end_year
    if start > end:
        start, end = end, start

    print(f"Starting scrape for seasons {start}..{end}")

    for year in range(start, end + 1):
        season_dir = os.path.join("data", str(year))
        # Players
        scrape_player_stats_for_season(year, season_dir)
        # Standings
        scrape_standings_for_season(year, season_dir)
        # MVP voting
        scrape_mvp_voting_for_season(year, season_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
