import pandas as pd
import numpy as np
from src.utils import haversine_distance

def calculate_advanced_features(df):
    """
    Adds the Distance and Probability features to the dataframe.
    This works for both the full training set AND the single-row inference dataframe.
    """
    epsilon = 1e-5

    # 1. Calculate Distances (Haversine)
    # We iterate through the regions. 
    # NOTE: For inference, we will just use 'E_' and 'W_' as placeholders for Team 1 and Team 2
    # to match the model's expected input format.
    
    regions = ['E_', 'W_', 'M_', 'S_']
    
    for region in regions:
        lat_col = f'{region}InstitutionLatitude'
        lon_col = f'{region}InstitutionLongitude'
        
        # Check if columns exist (Inference might only have E and W)
        if lat_col in df.columns:
            df[f'{region}distance'] = df.apply(
                lambda row: haversine_distance(
                    row['CustomerPostalCodeLatitude'],
                    row['CustomerPostalCodeLongitude'],
                    row[lat_col],
                    row[lon_col]
                ), axis=1
            )
            
            # 2. Calculate Distance Score (Inverse Distance)
            df[f'{region}dist_score'] = 1 / (df[f'{region}distance'] + epsilon)
            
            # Clean up raw distance column if desired, or keep it.
            # The notebook dropped it, so we should likely drop it to avoid leaking
            # But let's keep it for now and drop it before model input
            # df = df.drop(columns=[f'{region}distance'])

    # 3. Calculate Normalized Probabilities
    # East vs West
    if 'E_dist_score' in df.columns and 'W_dist_score' in df.columns:
        df['total_dist_score_EW'] = df[['E_dist_score', 'W_dist_score']].sum(axis=1)
        df['E_dist_prob'] = df['E_dist_score'] / df['total_dist_score_EW']
        df['W_dist_prob'] = df['W_dist_score'] / df['total_dist_score_EW']
    
    # Midwest vs South
    if 'M_dist_score' in df.columns and 'S_dist_score' in df.columns:
        df['total_dist_score_MS'] = df[['M_dist_score', 'S_dist_score']].sum(axis=1)
        df['M_dist_prob'] = df['M_dist_score'] / df['total_dist_score_MS']
        df['S_dist_prob'] = df['S_dist_score'] / df['total_dist_score_MS']
        
    return df

def prepare_inference_data(user_lat, user_lon, team1_data, team2_data):
    """
    Creates a single-row DataFrame for the App to feed into the model.
    Maps Team 1 -> 'E_' (East) features
    Maps Team 2 -> 'W_' (West) features
    """
    # Create a dictionary representing one row of data
    row = {
        'CustomerPostalCodeLatitude': user_lat,
        'CustomerPostalCodeLongitude': user_lon,
        
        # Team 1 (Treated as East)
        'E_InstitutionLatitude': team1_data['InstitutionLatitude'],
        'E_InstitutionLongitude': team1_data['InstitutionLongitude'],
        'E_Seed_Rank': team1_data.get('Seed_Rank', 8), # Default mid-seed if missing
        'E_NetRtg': team1_data.get('NetRtg', 0),
        'E_Luck': team1_data.get('Luck', 0),
        'E_win_%': team1_data.get('win_%', 0.5),
        
        # Team 2 (Treated as West)
        'W_InstitutionLatitude': team2_data['InstitutionLatitude'],
        'W_InstitutionLongitude': team2_data['InstitutionLongitude'],
        'W_Seed_Rank': team2_data.get('Seed_Rank', 8),
        'W_NetRtg': team2_data.get('NetRtg', 0),
        'W_Luck': team2_data.get('Luck', 0),
        'W_win_%': team2_data.get('win_%', 0.5),
    }
    
    df = pd.DataFrame([row])
    
    # Calculate the derived features (Distance, Probabilities)
    df = calculate_advanced_features(df)
    
    # Select only the columns the model expects
    # Note: We must ensure these match EXACTLY what train_model.py uses
    features = [
        'E_dist_prob', 'W_dist_prob',
        'E_win_%', 'W_win_%',
        'E_Seed_Rank', 'W_Seed_Rank',
        'E_NetRtg', 'W_NetRtg',
        'E_Luck', 'W_Luck'
    ]
    
    return df[features]
