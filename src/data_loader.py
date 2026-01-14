import pandas as pd
import numpy as np
from pathlib import Path

# Constants for Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'

def load_and_merge_data():
    """
    Loads raw data, cleans KenPom names, and merges everything into a single DataFrame.
    Returns:
        pd.DataFrame: The enriched training data (similar to model-3.ipynb).
    """
    # Load Raw CSVs
    try:
        bracket_training = pd.read_csv(DATA_DIR / "bracket_training.csv")
        bracket_test = pd.read_csv(DATA_DIR / "bracket_test.csv")
        college_info = pd.read_csv(DATA_DIR / "institutions.csv", encoding='utf-8')
        df_kenpom = pd.read_csv(DATA_DIR / "Kenpom Data.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find data files in {DATA_DIR}. Error: {e}")

    # --- Step 1: Clean KenPom Data ---
    # Logic copied from model-3.ipynb
    df_kenpom['Team_Name'] = df_kenpom['Team'].apply(lambda x: ' '.join(x.split()[:-1]))
    
    mapping = {
        "Connecticut": "UConn", "Houston": "Houston", "Purdue": "Purdue", "Auburn": "Auburn",
        "Tennessee": "Tennessee", "Arizona": "Arizona", "Duke": "Duke", "Iowa St.": "Iowa St.",
        "North Carolina": "North Carolina", "Illinois": "Illinois", "Creighton": "Creighton",
        "Gonzaga": "Gonzaga", "Marquette": "Marquette", "Alabama": "Alabama", "Baylor": "Baylor",
        "Michigan St.": "Michigan St.", "Wisconsin": "Wisconsin", "BYU": "BYU", "Clemson": "Clemson",
        "Saint Mary's": "Saint Mary's", "San Diego St.": "San Diego St.", "Kentucky": "Kentucky",
        "Colorado": "Colorado", "Texas": "Texas", "Florida": "Florida", "Kansas": "Kansas",
        "New Mexico": "New Mexico", "Nebraska": "Nebraska", "Texas Tech": "Texas Tech",
        "Dayton": "Dayton", "Mississippi St.": "Mississippi St.", "Texas A&M": "Texas A&M",
        "Colorado St.": "Colorado St.", "Nevada": "Nevada", "Northwestern": "Northwestern",
        "Washington St.": "Washington St.", "TCU": "TCU", "Boise St.": "Boise St.",
        "N.C. State": "NC State", "Florida Atlantic": "FAU", "Utah St.": "Utah St.",
        "Grand Canyon": "Grand Canyon", "Drake": "Drake", "South Carolina": "South Carolina",
        "Oregon": "Oregon", "James Madison": "James Madison", "McNeese St.": "McNeese",
        "Virginia": "Virginia", "Samford": "Samford", "Duquesne": "Duquesne", "Yale": "Yale",
        "Charleston": "Charleston", "Vermont": "Vermont", "UAB": "UAB", "Morehead St.": "Morehead St.",
        "Akron": "Akron", "Oakland": "Oakland", "Western Kentucky": "Western Ky.",
        "South Dakota St.": "South Dakota St.", "Colgate": "Colgate", "Longwood": "Longwood",
        "Long Beach St.": "Long Beach St.", "Saint Peter's": "Saint Peter's", "Stetson": "Stetson",
        "Montana St.": "Montana St.", "Grambling St.": "Grambling St.", "Howard": "Howard", "Wagner": "Wagner"
    }
    
    df_kenpom['Team_Name'] = df_kenpom['Team_Name'].map(mapping)
    
    # Extract Seed Rank
    df_kenpom['Seed_Rank'] = df_kenpom['Team'].str.extract(r'(\d+)$')
    df_kenpom = df_kenpom.dropna(subset=['Seed_Rank'])
    df_kenpom['Seed_Rank'] = df_kenpom['Seed_Rank'].astype(int)
    
    # Select clean columns
    df_ken_clean = df_kenpom.loc[:, ['Rk', 'Team_Name', 'Seed_Rank', 'NetRtg', 'Luck']]
    df_ken_clean = df_ken_clean.set_index('Team_Name')

    # --- Step 2: Merge with College Info ---
    college_info_ken_df = college_info.join(df_ken_clean, how='left', on='InstitutionName')
    
    # Calculate Win Percentage
    college_info_ken_df['win_%'] = college_info_ken_df['RegularSeasonWins'] / (
        college_info_ken_df['RegularSeasonWins'] + college_info_ken_df['RegularSeasonLosses']
    )
    
    # Set Index for easier joining
    college_info_ken_df = college_info_ken_df.set_index('InstitutionID')

    # --- Step 3: Create Full Training Set ---
    # Join the college info for all 4 regions (East, West, Midwest, South)
    train_df = bracket_training.join(
        college_info_ken_df.add_prefix("W_"), on="RegionWinner_West"
    ).join(
        college_info_ken_df.add_prefix("E_"), on="RegionWinner_East"
    ).join(
        college_info_ken_df.add_prefix('M_'), on="RegionWinner_Midwest"
    ).join(
        college_info_ken_df.add_prefix('S_'), on='RegionWinner_South'
    )

    return train_df, college_info_ken_df

def get_team_metadata():
    """
    Returns a dictionary of team stats for the App.
    Key: Team Name
    Value: Dict with Lat, Lon, Win %, Seed, etc.
    """
    _, college_df = load_and_merge_data()
    
    # We want to lookup by Name, not ID
    college_df = college_df.reset_index().set_index('InstitutionName')
    
    # Convert to dictionary (orient='index')
    return college_df.to_dict(orient='index')
