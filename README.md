# 🏀 March Madness: Fan Behavior Predictor

<p align="center">
  <strong>🏆 3rd Place Winner @ 2024 NCAA & CCAC Research Competition</strong>
</p>

## 🚀 Overview

This project uses **Machine Learning (RandomForest)** to predict which teams fans will select in their March Madness brackets. 

**Key Discovery:** Fan behavior is driven by **Affinity Bias** (Geographic Proximity) as much as objective team performance. Our model predicts fan choices with **67% accuracy** by quantifying the "Home Team Advantage" effect.

## 🎮 Interactive Demo

We have built a **Streamlit App** to demonstrate the model live.

### How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (First time only)
   This generates the `models/fan_model.joblib` file used by the app.
   ```bash
   python train_model.py
   ```

3. **Launch the App**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

*   `app.py`: The interactive Fan Simulator and Analytics Dashboard.
*   `src/`: Core logic for data processing and feature engineering.
*   `train_model.py`: Script to retrain and save the ML model.
*   `notebooks/`: Original research and EDA notebooks (`model-3.ipynb` contains the primary research).
*   `data/`: Raw competition datasets.

## 🔬 Methodology

1.  **Data**: 76,000+ Fan Brackets + KenPom Analytics + Geographic Data.
2.  **Feature Engineering**: 
    *   **Haversine Distance**: Calculated distance between every fan and every team.
    *   **Distance Differential**: Normalized score comparing proximity of two competing teams.
3.  **Model**: RandomForest Classifier optimizing for "East-West" Semifinal selections.

## 👤 Author
**Michael Whitfield**  
*Data Science | Behavioral Analytics*
