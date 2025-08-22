import pandas as pd
from pathlib import Path


# Define the final list of columns to keep after merging
FINAL_COLUMNS = [
    'CustomerID', 'CustomerPostalCodeLatitude', 'CustomerPostalCodeLongitude', 
    'CustomerDMACode', 'CustomerDMADescription', 'NCAACustomerRecordCreated', 
    'BracketEntryId', 'BracketEntryCreatedDate', 'RegionWinner_East', 
    'RegionWinner_West', 'RegionWinner_South', 'RegionWinner_Midwest', 
    'SemifinalWinner_East_West', 'SemifinalWinner_South_Midwest', 'NationalChampion',
    'E_InstitutionName', 'E_InstitutionDMACode', 'E_InstitutionLatitude', 
    'E_InstitutionLongitude', 'E_InstitutionConference', 'E_InstitutionEnrollment_Male', 
    'E_InstitutionEnrollment_Female', 'E_InstitutionEnrollment_Total', 
    'E_InstitutionNCAAMemberSinceDate', 'E_RegularSeasonWins', 'E_RegularSeasonLosses', 
    'E_RegularSeasonAverageAttendance', 'E_RegularSeasonAverageScore', 'E_Rk', 
    'E_Seed_Rank', 'E_NetRtg', 'E_Luck', 'M_InstitutionName', 'M_InstitutionDMACode', 
    'M_InstitutionLatitude', 'M_InstitutionLongitude', 'M_InstitutionConference', 
    'M_InstitutionEnrollment_Male', 'M_InstitutionEnrollment_Female', 
    'M_InstitutionEnrollment_Total', 'M_InstitutionNCAAMemberSinceDate', 
    'M_RegularSeasonWins', 'M_RegularSeasonLosses', 'M_RegularSeasonAverageAttendance', 
    'M_RegularSeasonAverageScore', 'M_Rk', 'M_Seed_Rank', 'M_NetRtg', 'M_Luck',
    'S_InstitutionName', 'S_InstitutionDMACode', 'S_InstitutionLatitude', 
    'S_InstitutionLongitude', 'S_InstitutionConference', 'S_InstitutionEnrollment_Male', 
    'S_InstitutionEnrollment_Female', 'S_InstitutionEnrollment_Total', 
    'S_InstitutionNCAAMemberSinceDate', 'S_RegularSeasonWins', 'S_RegularSeasonLosses', 
    'S_RegularSeasonAverageAttendance', 'S_RegularSeasonAverageScore', 'S_Rk', 
    'S_Seed_Rank', 'S_NetRtg', 'S_Luck', 'W_InstitutionName', 'W_InstitutionDMACode', 
    'W_InstitutionLatitude', 'W_InstitutionLongitude', 'W_InstitutionConference', 
    'W_InstitutionEnrollment_Male', 'W_InstitutionEnrollment_Female', 
    'W_InstitutionEnrollment_Total', 'W_InstitutionNCAAMemberSinceDate', 
    'W_RegularSeasonWins', 'W_RegularSeasonLosses', 'W_RegularSeasonAverageAttendance', 
    'W_RegularSeasonAverageScore', 'W_Rk', 'W_Seed_Rank', 'W_NetRtg', 'W_Luck'
]

def load_and_merge_data(data_path: Path, bracket_file: str, is_training: bool = True) -> pd.DataFrame:
    """
    Loads raw data files, processes KenPom data, and merges them into a single DataFrame.

    Args:
        data_path (Path): The path to the directory containing the CSV files.
        bracket_file (str): The filename of the bracket data (e.g., 'bracket_training.csv').

    Returns:
        pd.DataFrame: A single, merged DataFrame ready for feature engineering.
    """
    # --- Step 1: Read Raw Data ---
    bracket_df = pd.read_csv(data_path / bracket_file)
    college_info = pd.read_csv(data_path / 'institutions.csv', encoding='utf-8')
    df_kenpom = pd.read_csv(data_path / 'Kenpom Data.csv')

    # --- Step 2: Process KenPom Data ---
    df_kenpom['Team_Name'] = df_kenpom['Team'].apply(lambda x: ' '.join(x.split()[:-1]))
    # This mapping should be externalized in a real project (e.g., a config file)
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
    df_kenpom['Seed_Rank'] = df_kenpom['Team'].str.extract(r'(\d+)$')
    df_kenpom = df_kenpom.dropna(subset=['Seed_Rank'])
    df_kenpom['Seed_Rank'] = df_kenpom['Seed_Rank'].astype(int)
    df_ken_clean = df_kenpom.loc[:, ['Rk', 'Team_Name', 'Seed_Rank', 'NetRtg', 'Luck']].set_index('Team_Name')
    
    # --- Step 3: Merge and Join ---
    college_info_ken_df = college_info.join(df_ken_clean, how='left', on='InstitutionName').set_index('InstitutionID')
    
    merged_df = bracket_df.join(
        college_info_ken_df.add_prefix("W_"), on="RegionWinner_West"
    ).join(
        college_info_ken_df.add_prefix("E_"), on="RegionWinner_East"
    ).join(
        college_info_ken_df.add_prefix('M_'), on="RegionWinner_Midwest"
    ).join(
        college_info_ken_df.add_prefix('S_'), on='RegionWinner_South'
    )
    if is_training:
        # The training set has target columns that the test set doesn't
        final_cols_for_df = FINAL_COLUMNS
    else:
        # Remove target columns not present in the test set
        test_cols_to_remove = ['SemifinalWinner_East_West', 'SemifinalWinner_South_Midwest', 'NationalChampion']
        final_cols_for_df = [col for col in FINAL_COLUMNS if col not in test_cols_to_remove]

    filtered_df = merged_df[final_cols_for_df]
    
    print(f"Data from {bracket_file} loaded, merged, and filtered successfully.")
    return filtered_df