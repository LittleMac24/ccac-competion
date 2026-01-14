import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pathlib import Path

from src.data_loader import load_and_merge_data
from src.features import calculate_advanced_features

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fan_model.joblib"

def train_and_save_model():
    print("⏳ Loading Data...")
    df_train, _ = load_and_merge_data()
    
    print("⏳ Calculating Features...")
    df_train = calculate_advanced_features(df_train)
    
    # Define Target and Features
    # We are training on the East vs West Semifinal data
    target_col = "SemifinalWinner_East_West"
    
    # We must ensure we filter out rows where the target is NaN
    df_train = df_train.dropna(subset=[target_col])
    
    # Create Binary Target: 1 if East Won, 0 if West Won
    # We assume 'RegionWinner_East' holds the team name
    y = (df_train[target_col] == df_train['RegionWinner_East']).astype(int)
    
    # Feature List (Must match prepare_inference_data in features.py)
    features = [
        'E_dist_prob', 'W_dist_prob',
        'E_win_%', 'W_win_%',
        'E_Seed_Rank', 'W_Seed_Rank',
        'E_NetRtg', 'W_NetRtg',
        'E_Luck', 'W_Luck'
    ]
    
    X = df_train[features]
    
    print(f"✅ Data Ready: {X.shape[0]} samples")
    
    # Define Pipeline
    # Numeric features need imputation and scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("⏳ Training Model...")
    pipeline.fit(X, y)
    
    print(f"💾 Saving model to {MODEL_PATH}")
    joblib.dump(pipeline, MODEL_PATH)
    print("✅ Done!")

if __name__ == "__main__":
    train_and_save_model()
