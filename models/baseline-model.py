import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, recall_score
import numpy as np

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 100)

# The location of the current .py file
SCRIPT_DIR = Path(__file__).resolve().parent
# The project root is one level up from the script's directory
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_PATH = PROJECT_ROOT/'data'/'training_set.csv'

def evaluate_baseline_model(dataframe, team1, team2, win_perc1, win_perc2, actual_winner):
    '''Calculates the accuracy and recall by choosing the team with higher win percentage'''
    pred_winner = (dataframe[win_perc1] > dataframe[win_perc2]).astype(int)
    act_winner = (dataframe[actual_winner] == dataframe[team1]).astype(int)

    recall = recall_score(act_winner, pred_winner)
    accuracy = accuracy_score(act_winner, pred_winner)

    print(f'Baseline Model ({team1} vs {team2}): Recall {recall:.4f}, Accuracy {accuracy:.4f}')

    return pred_winner


def prepare_nat_champ_data(df_training, mid_south_df, east_west_df, ms_predictions, ew_predictions):
    """
    Prepares the DataFrame for predicting the national champion based on semifinal winners.

    Args:
        df_training (pd.DataFrame): The original full training dataset.
        mid_south_df (pd.DataFrame): The DataFrame for the Mid-South semifinal.
        east_west_df (pd.DataFrame): The DataFrame for the East-West semifinal.
        ms_predictions (pd.Series): Binary predictions from the Mid-South game.
        ew_predictions (pd.Series): Binary predictions from the East-West game.

    Returns:
        pd.DataFrame: A new DataFrame ready for national champion evaluation.
    """
    
    nat_champ_df = pd.DataFrame({
        # If ew_prediction is 1 (East wins), use east win %, otherwise use west win %
        "ew_winner_win_%": east_west_df['e_win_%'].where(ew_predictions == 1, east_west_df['w_win_%']),
        "ew_winner_name": east_west_df['RegionWinner_East'].where(ew_predictions == 1, east_west_df['RegionWinner_West']),
        
        # If ms_prediction is 1 (Midwest wins), use midwest win %, otherwise use south win %
        "ms_winner_win_%": mid_south_df['m_win_%'].where(ms_predictions == 1, mid_south_df['s_win_%']),
        
        "actual_nat_champ": df_training['NationalChampion']
    })
    
    return nat_champ_df


if __name__ == "__main__":
    # --- Load Data ---
    df_training = pd.read_csv(DATA_PATH)

    # --- Prepare Data for Semifinals ---
    # Create the mid_south_basic_df and east_west_basic_df
    mid_south_basic_df = df_training.loc[:, ["RegionWinner_South", 'RegionWinner_Midwest', 'm_win_%','s_win_%', 'SemifinalWinner_South_Midwest']]
    east_west_basic_df = df_training.loc[:, ["RegionWinner_East", 'RegionWinner_West', 'e_win_%','w_win_%', 'SemifinalWinner_East_West']]

    # --- Semifinal Evaluation ---
    print("--- Semifinal Evaluations ---")
    ms_predictions = evaluate_baseline_model(
        dataframe=mid_south_basic_df,
        team1='RegionWinner_Midwest',
        team2='RegionWinner_South',
        win_perc1='m_win_%',
        win_perc2='s_win_%',
        actual_winner='SemifinalWinner_South_Midwest'
    )
    ew_predictions = evaluate_baseline_model(
        dataframe=east_west_basic_df,
        team1='RegionWinner_East',
        team2='RegionWinner_West',
        win_perc1='e_win_%',
        win_perc2='w_win_%',
        actual_winner='SemifinalWinner_East_West'
    )

    # --- National Champion Evaluation ---
    print("\n--- National Champion Evaluation ---")
    # 1. Prepare data
    nat_champ_df = prepare_nat_champ_data(
        df_training,
        mid_south_basic_df,
        east_west_basic_df,
        ms_predictions,
        ew_predictions
    )
    # 2. Evaluate
    # Note: We added 'ms_winner_name' during prep to make this call work
    nat_champ_df['ms_winner_name'] = mid_south_basic_df['RegionWinner_Midwest'].where(ms_predictions == 1, mid_south_basic_df['RegionWinner_South'])
    evaluate_baseline_model(
        dataframe=nat_champ_df,
        team1='ew_winner_name',
        team2='ms_winner_name',
        win_perc1='ew_winner_win_%',
        win_perc2='ms_winner_win_%',
        actual_winner='actual_nat_champ'
    )
