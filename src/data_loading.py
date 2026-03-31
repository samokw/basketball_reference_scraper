"""
Load and merge scraped Basketball Reference data across seasons.
"""
import pandas as pd
import os
from typing import Optional


def normalize_team_name(team_name: str) -> str:
    """
    Normalize team names to remove asterisks and other formatting.
    
    Basketball Reference uses asterisks (*) to mark playoff teams.
    """
    if pd.isna(team_name):
        return team_name
    return str(team_name).replace('*', '').strip()


def load_season_data(year: int, base_path: str = "data/") -> pd.DataFrame:
    """
    Load all data for a given season and merge into single DataFrame.
    
    Args:
        year: Season year (e.g., 2024 for 2023-24 season)
        base_path: Path to data directory
    
    Returns:
        DataFrame with columns: Player, Team, Season, [stats], AwardShare
    """
    year_path = os.path.join(base_path, str(year))
    
    if not os.path.exists(year_path):
        raise FileNotFoundError(f"Data directory not found: {year_path}")
    
    # Load player stats - use per_game for primary stats, advanced for advanced metrics
    players_per_game = pd.read_csv(os.path.join(year_path, "players_per_game.csv"))
    players_advanced = pd.read_csv(os.path.join(year_path, "players_advanced.csv"))
    
    # Load team stats (standings)
    standings = pd.read_csv(os.path.join(year_path, "standings.csv"))
    
    # Load MVP voting results (award share is our target)
    try:
        mvp_voting = pd.read_csv(os.path.join(year_path, "mvp_voting.csv"))
    except FileNotFoundError:
        print(f"  WARN: MVP voting not found for {year}, creating empty voting data")
        mvp_voting = pd.DataFrame(columns=['Player', 'Voting_Share'])
    
    # Merge player stats (per_game + advanced)
    # Start with per_game as base
    merged = players_per_game.copy()
    
    # Merge with advanced stats on Player and Team
    # Select only advanced stats columns that don't overlap
    advanced_cols = ['Player', 'Team', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 
                     'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48',
                     'OBPM', 'DBPM', 'BPM', 'VORP']
    advanced_cols = [col for col in advanced_cols if col in players_advanced.columns]
    
    merged = merged.merge(
        players_advanced[advanced_cols],
        on=['Player', 'Team'],
        how='left',
        suffixes=('', '_advanced')
    )
    
    # Normalize team names for merging with standings
    merged['Team'] = merged['Team'].apply(normalize_team_name)
    standings['Team'] = standings['Team'].apply(normalize_team_name)
    
    # Merge with team standings
    # Standings columns: Team, W, L, W/L%, GB, PS/G, PA/G, SRS, Conference
    team_cols = ['Team', 'W', 'L', 'W/L%', 'SRS', 'Conference']
    team_cols = [col for col in team_cols if col in standings.columns]
    
    merged = merged.merge(
        standings[team_cols],
        on='Team',
        how='left'
    )
    
    # Calculate win percentage if not present
    if 'W/L%' not in merged.columns and 'W' in merged.columns and 'L' in merged.columns:
        merged['W/L%'] = merged['W'] / (merged['W'] + merged['L'])
    
    # Rename W/L% to WinPct for consistency
    if 'W/L%' in merged.columns:
        merged = merged.rename(columns={'W/L%': 'WinPct'})
    
    # Calculate conference rank (1 = best team in conference, higher = worse)
    if 'Conference' in merged.columns and 'WinPct' in merged.columns:
        merged['ConferenceRank'] = merged.groupby('Conference')['WinPct'].rank(
            ascending=False, method='min', na_option='keep'
        )
        # Fill NaN with a high rank (999) to indicate missing data
        merged['ConferenceRank'] = merged['ConferenceRank'].fillna(999).astype(int)
    
    # Merge with MVP voting (left join - players with no votes get 0)
    if 'Player' in mvp_voting.columns and 'Voting_Share' in mvp_voting.columns:
        mvp_voting_slim = mvp_voting[['Player', 'Voting_Share']].copy()
        
        # Normalize player names (remove special characters, case-insensitive merge)
        merged = merged.merge(
            mvp_voting_slim,
            on='Player',
            how='left'
        )
        merged['AwardShare'] = merged['Voting_Share'].fillna(0)
        merged = merged.drop(columns=['Voting_Share'], errors='ignore')
    else:
        # No MVP voting data, set AwardShare to 0
        merged['AwardShare'] = 0.0
    
    # Add season identifier
    merged['Season'] = year
    
    # Apply 65-game eligibility filter (NBA rule since 2023-24, but we apply to all)
    # Only filter if G (games) column exists
    if 'G' in merged.columns:
        # Handle both numeric and string types
        merged['G'] = pd.to_numeric(merged['G'], errors='coerce')
        merged = merged[merged['G'] >= 65].copy()
    
    return merged


def load_all_seasons(
    start_year: int = 2015, 
    end_year: int = 2024,
    save_path: str = "data/processed/mvp_consolidated.csv"
) -> pd.DataFrame:
    """
    Load and concatenate all seasons into master dataset.
    
    Args:
        start_year: First season to load
        end_year: Last season to load
        save_path: Path to save consolidated CSV
    
    Returns:
        Consolidated DataFrame with all seasons
    """
    all_data = []
    
    print("="*60)
    print("LOADING SEASONS")
    print("="*60)
    
    for year in range(start_year, end_year + 1):
        print(f"\nLoading {year} season...")
        try:
            season_data = load_season_data(year)
            all_data.append(season_data)
            print(f"  [OK] {len(season_data)} eligible players (>=65 games)")
        except Exception as e:
            print(f"  [ERROR] Failed: {e}")
    
    if not all_data:
        raise ValueError("No data loaded. Check data directory and file structure.")
    
    # Concatenate all seasons
    full_data = pd.concat(all_data, ignore_index=True)
    
    print("\n" + "="*60)
    print(f"TOTAL: {len(full_data)} player-seasons")
    print(f"Seasons: {full_data['Season'].nunique()}")
    print(f"Unique players: {full_data['Player'].nunique()}")
    print("="*60)
    
    # Save consolidated dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    full_data.to_csv(save_path, index=False)
    print(f"\nSaved to {save_path}")
    
    return full_data


def load_processed_data(path: str = "data/processed/mvp_consolidated.csv") -> pd.DataFrame:
    """
    Load the consolidated dataset (use this after running load_all_seasons once).
    
    Args:
        path: Path to consolidated CSV file
    
    Returns:
        Consolidated DataFrame
    """
    return pd.read_csv(path)


if __name__ == "__main__":
    # Load all seasons and create consolidated dataset
    df = load_all_seasons(start_year=2015, end_year=2024)
    
    print("\nDataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nMVP winners per season:")
    mvp_winners = df[df['AwardShare'] > 0.5][['Season', 'Player', 'AwardShare']].sort_values('Season')
    print(mvp_winners)

