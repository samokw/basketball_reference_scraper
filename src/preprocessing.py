"""
Preprocess data and engineer features for MVP prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features from raw stats.
    
    Features to create:
    1. Within-season z-scores (normalize relative to season averages)
    2. Pace-adjusted stats (per-75 possessions or per-36 minutes)
    3. Interaction terms (WinPct × individual stats)
    4. Team context indicators (top seed, playoff team)
    
    Args:
        df: DataFrame with raw stats
    
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()
    
    # 1. Within-season z-scores for key stats
    stats_to_normalize = ['PTS', 'TRB', 'AST', 'PER', 'WS', 'VORP', 'BPM']
    
    for stat in stats_to_normalize:
        if stat in df.columns:
            # Convert to numeric, handling any string values
            df[stat] = pd.to_numeric(df[stat], errors='coerce')
            
            # Calculate z-scores within each season
            df[f'{stat}_zscore'] = df.groupby('Season')[stat].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
    
    # 2. Pace-adjusted stats (per-36 minutes, since we don't have possessions in all datasets)
    if 'MP' in df.columns:
        df['MP'] = pd.to_numeric(df['MP'], errors='coerce')
        pace_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
        for stat in pace_stats:
            if stat in df.columns:
                df[stat] = pd.to_numeric(df[stat], errors='coerce')
                # Per-36 minutes
                df[f'{stat}_per36'] = (df[stat] / df['MP']) * 36
                # Avoid division by zero
                df[f'{stat}_per36'] = df[f'{stat}_per36'].replace([np.inf, -np.inf], np.nan)
    
    # 3. Interaction terms (WinPct × performance metrics)
    if 'WinPct' in df.columns:
        df['WinPct'] = pd.to_numeric(df['WinPct'], errors='coerce')
        interaction_stats = ['WS', 'PER', 'VORP', 'BPM', 'PTS']
        for stat in interaction_stats:
            if stat in df.columns:
                df[stat] = pd.to_numeric(df[stat], errors='coerce')
                df[f'WinPct_x_{stat}'] = df['WinPct'] * df[stat]
    
    # 4. Team context indicators
    if 'ConferenceRank' in df.columns:
        df['ConferenceRank'] = pd.to_numeric(df['ConferenceRank'], errors='coerce')
        df['TopSeed'] = (df['ConferenceRank'] <= 2).astype(int)
        df['PlayoffTeam'] = (df['ConferenceRank'] <= 8).astype(int)  # Top 8 per conference
    elif 'WinPct' in df.columns:
        # Fallback: use win percentage threshold
        df['TopSeed'] = (df['WinPct'] >= 0.65).astype(int)
        df['PlayoffTeam'] = (df['WinPct'] >= 0.5).astype(int)
    
    # 5. Usage rate interactions (high usage + high efficiency = MVP candidate)
    if 'USG%' in df.columns and 'TS%' in df.columns:
        df['USG%'] = pd.to_numeric(df['USG%'], errors='coerce')
        df['TS%'] = pd.to_numeric(df['TS%'], errors='coerce')
        df['USG_x_TS'] = df['USG%'] * df['TS%']
    
    # 6. Triple-double indicators (approximate)
    if all(col in df.columns for col in ['PTS', 'TRB', 'AST']):
        df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')
        df['TRB'] = pd.to_numeric(df['TRB'], errors='coerce')
        df['AST'] = pd.to_numeric(df['AST'], errors='coerce')
        df['NearTripleDouble'] = (
            (df['PTS'] >= 8) & (df['TRB'] >= 8) & (df['AST'] >= 8)
        ).astype(int)
    
    return df


def prepare_features_target(
    df: pd.DataFrame,
    target_type: str = 'regression'
) -> tuple:
    """
    Prepare feature matrix X and target variable y.
    
    Args:
        df: DataFrame with all features
        target_type: 'regression' (predict award share) or 'classification' (predict winner)
    
    Returns:
        X (features DataFrame), y (target Series), feature_names (list)
    """
    # Select feature columns (adjust based on available data)
    feature_cols = [
        # Raw stats (per game)
        'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        # Shooting percentages
        'FG%', '3P%', 'FT%', 'TS%', 'eFG%',
        # Usage and advanced rates
        'USG%', 'PER',
        # Advanced metrics
        'WS', 'WS/48', 'BPM', 'VORP',
        # Team context
        'W', 'L', 'WinPct', 'SRS', 'ConferenceRank',
        # Engineered features
        'WinPct_x_WS', 'WinPct_x_PER', 'WinPct_x_VORP', 'WinPct_x_BPM', 'WinPct_x_PTS',
        'TopSeed', 'PlayoffTeam', 'USG_x_TS', 'NearTripleDouble',
        # Z-scores
        'PTS_zscore', 'TRB_zscore', 'AST_zscore', 'PER_zscore',
        'WS_zscore', 'VORP_zscore', 'BPM_zscore',
        # Per-36 stats
        'PTS_per36', 'TRB_per36', 'AST_per36', 'STL_per36', 'BLK_per36',
    ]
    
    # Keep only columns that exist in df
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame. Check column names.")
    
    X = df[feature_cols].copy()
    
    # Convert all columns to numeric, handling any string values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values (median imputation)
    X = X.fillna(X.median())
    
    # Create target variable
    if target_type == 'regression':
        if 'AwardShare' not in df.columns:
            raise ValueError("AwardShare column not found for regression target")
        y = pd.to_numeric(df['AwardShare'], errors='coerce').fillna(0)
    elif target_type == 'classification':
        # Binary: MVP winner (1) vs non-winner (0)
        # Threshold: award share > 0.5 typically means winner
        if 'AwardShare' not in df.columns:
            raise ValueError("AwardShare column not found for classification target")
        y = (pd.to_numeric(df['AwardShare'], errors='coerce') > 0.5).astype(int)
    else:
        raise ValueError("target_type must be 'regression' or 'classification'")
    
    return X, y, feature_cols


def temporal_train_test_split(
    df: pd.DataFrame,
    train_end: int = 2022,
    val_year: int = 2023,
    test_year: int = 2024
):
    """
    Split data temporally (NOT random) to avoid leakage.
    
    Args:
        df: DataFrame with Season column
        train_end: Last year for training data (inclusive)
        val_year: Year for validation data
        test_year: Year for test data
    
    Returns:
        train_df, val_df, test_df
    """
    if 'Season' not in df.columns:
        raise ValueError("DataFrame must have 'Season' column for temporal splitting")
    
    train_df = df[df['Season'] <= train_end].copy()
    val_df = df[df['Season'] == val_year].copy()
    test_df = df[df['Season'] == test_year].copy()
    
    print(f"Train: {len(train_df)} samples (seasons <= {train_end})")
    print(f"Val:   {len(val_df)} samples (season {val_year})")
    print(f"Test:  {len(test_df)} samples (season {test_year})")
    
    return train_df, val_df, test_df

