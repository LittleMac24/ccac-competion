# src/feature_engineer.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def _haversine_distance(lat1, lon1, lat2, lon2):
    """Helper function to calculate distance between two geo-coordinates."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

class FeatureEngineer:
    """
    A class to handle all feature engineering steps for the bracket data.
    Separates fitting on training data from transforming new data to prevent leakage.
    """
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.num_cols = ['CustomerPostalCodeLatitude', 'CustomerPostalCodeLongitude']
        self.cat_cols = ['CustomerDMACode', 'CustomerDMADescription']
    
    def fit(self, df: pd.DataFrame):
        """Fits the imputers on the training data."""
        self.num_imputer.fit(df[self.num_cols])
        self.cat_imputer.fit(df[self.cat_cols])
        print("FeatureEngineer has been fitted.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all feature engineering transformations."""
        df_copy = df.copy()

        # --- Impute Missing Values using the fitted imputers ---
        df_copy[self.num_cols] = self.num_imputer.transform(df_copy[self.num_cols])
        df_copy[self.cat_cols] = self.cat_imputer.transform(df_copy[self.cat_cols])

        # --- Create Win Percentages ---
        for region in ['m', 's', 'e', 'w']:
            wins_col = f'{region.upper()}_RegularSeasonWins'
            losses_col = f'{region.upper()}_RegularSeasonLosses'
            df_copy[f'{region}_win_%'] = df_copy[wins_col] / (df_copy[wins_col] + df_copy[losses_col])

        # --- Create Distance-Based Popularity Features ---
        for region, lat, lon in zip(
            ['E_', 'W_', 'M_', 'S_'],
            ['E_InstitutionLatitude', 'W_InstitutionLatitude', 'M_InstitutionLatitude', 'S_InstitutionLatitude'],
            ['E_InstitutionLongitude', 'W_InstitutionLongitude', 'M_InstitutionLongitude', 'S_InstitutionLongitude']
        ):
            df_copy[f'{region}distance'] = df_copy.apply(
                lambda row: _haversine_distance(row['CustomerPostalCodeLatitude'], row['CustomerPostalCodeLongitude'], row[lat], row[lon]),
                axis=1
            )
            df_copy[f'{region}dist_score'] = 1 / (df_copy[f'{region}distance'] + 1e-5)

        # --- Normalize Distance Scores to Probabilities ---
        df_copy['total_dist_score_EW'] = df_copy['E_dist_score'] + df_copy['W_dist_score']
        df_copy['E_dist_prob'] = df_copy['E_dist_score'] / df_copy['total_dist_score_EW']
        df_copy['W_dist_prob'] = df_copy['W_dist_score'] / df_copy['total_dist_score_EW']

        df_copy['total_dist_score_MS'] = df_copy['M_dist_score'] + df_copy['S_dist_score']
        df_copy['M_dist_prob'] = df_copy['M_dist_score'] / df_copy['total_dist_score_MS']
        df_copy['S_dist_prob'] = df_copy['S_dist_score'] / df_copy['total_dist_score_MS']
        
        # --- Clean up intermediate columns ---
        cols_to_drop = [col for col in df_copy.columns if 'distance' in col or 'dist_score' in col]
        df_copy = df_copy.drop(columns=cols_to_drop)
        
        print("Feature transformations applied successfully.")
        return df_copy