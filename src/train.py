# src/train.py

# 1. Imports
import pandas as pd
from pathlib import Path
import numpy as np

# Import our custom modules
from data_builder import load_and_merge_data
from feature_engineer import FeatureEngineer

# Import sklearn components
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

# 2. Constants and Setup
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / 'data'

# Custom sigmoid transformer
def sigmoid_transform(X):
    return 1 - (1 / (1 + np.exp(-X)))
sigmoid_transformer = FunctionTransformer(sigmoid_transform, validate=False)


# 4. Main Execution Block
if __name__ == "__main__":
    # --- Data Preparation ---
    train_df = load_and_merge_data(DATA_PATH, bracket_file='bracket_training.csv')
    test_df = load_and_merge_data(DATA_PATH, bracket_file='bracket_test.csv', is_training=False)

    # --- Feature Engineering ---
    engineer = FeatureEngineer()
    engineer.fit(train_df) # Fit on training data
    train_featured = engineer.transform(train_df) # Transform training data
    test_featured = engineer.transform(test_df) # Transform test data

    # --- Modeling: East vs. West ---
    print("\n--- Training East vs. West Model ---")
    
    # Define features and target
    features_ew = [col for col in train_featured.columns if col.startswith(('E_', 'e_', 'W_', 'w_'))]
    train_featured['target_EW_Binary'] = (train_featured["SemifinalWinner_East_West"] == train_featured['RegionWinner_East']).astype(int)
    
    X_ew = train_featured[features_ew]
    y_ew = train_featured['target_EW_Binary']
    
    X_train_ew, X_test_ew, y_train_ew, y_test_ew = train_test_split(X_ew, y_ew, test_size=0.2, random_state=24)

    # Define preprocessor
    ordinal_features_ew = [col for col in features_ew if col.endswith("_Rk") or col.endswith('_Seed_Rank')]
    numerical_features_ew = [col for col in features_ew if col not in ordinal_features_ew and train_featured[col].dtype in ['int64', 'float64']]
    
    preprocessor_ew = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_ew),
            ('ord', sigmoid_transformer, ordinal_features_ew)
        ], remainder='passthrough'
    )

    # Define model pipeline
    model_ew = Pipeline(steps=[
        ('preprocessor', preprocessor_ew),
        ('selector', RFE(LogisticRegressionCV(cv=5, max_iter=2000), n_features_to_select=10)),
        ('classifier', LogisticRegressionCV(cv=5, max_iter=2000))
    ])
    
    # Train and evaluate
    model_ew.fit(X_train_ew, y_train_ew)
    y_pred_ew = model_ew.predict(X_test_ew)
    print(f'Accuracy for EW Model: {accuracy_score(y_true=y_test_ew, y_pred=y_pred_ew):.4f}')

    # --- Modeling: Midwest vs. South ---
    # (You would repeat a similar process for the MS model)
    print("\n--- Training Midwest vs. South Model ---")
    # ... (code for MS model would go here) ...