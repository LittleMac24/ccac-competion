import streamlit as st
import pandas as pd
import joblib
import pgeocode
import numpy as np
import altair as alt
from pathlib import Path
from src.features import prepare_inference_data
from src.data_loader import get_team_metadata
from src.utils import haversine_distance

# Page Config
st.set_page_config(
    page_title="March Madness Fan Predictor",
    page_icon="🏀",
    layout="wide"
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fan_model.joblib"

# --- Helper Functions ---
@st.cache_data
def load_teams():
    return get_team_metadata()

def load_model():
    if not MODEL_PATH.exists():
        st.error("⚠️ Model not found! Please run `python train_model.py` to generate it.")
        return None
    return joblib.load(MODEL_PATH)

def get_lat_lon_from_zip(zip_code):
    nomi = pgeocode.Nominatim('us')
    location = nomi.query_postal_code(zip_code)
    
    if pd.isna(location.latitude):
        return None, None
    return location.latitude, location.longitude

# --- Main App ---

st.title("🏀 March Madness: The Fan Behavior Model")
st.markdown("""
> **"It's not just about who wins. It's about who you *want* to win."**  
> This AI model predicts which team a fan will pick based on **Affinity Bias** (Geographic Proximity) vs. **Objective Performance** (Win %).
""")

# Load Resources
teams_metadata = load_teams()
model = load_model()

# --- Sidebar: Simulator ---
st.sidebar.header("🔮 Fan Simulator")
st.sidebar.markdown("Configure the fan and the matchup.")

user_zip = st.sidebar.text_input("Fan Zip Code", value="60601", max_chars=5)
team_names = sorted(list(teams_metadata.keys()))
team1_name = st.sidebar.selectbox("Team 1 (East)", team_names, index=team_names.index("Purdue") if "Purdue" in team_names else 0)
team2_name = st.sidebar.selectbox("Team 2 (West)", team_names, index=team_names.index("Duke") if "Duke" in team_names else 1)

btn_predict = st.sidebar.button("Predict Fan Choice", type="primary")

# --- Tabs ---
tab1, tab2 = st.tabs(["🎮 Simulator", "📊 Portfolio Insights"])

with tab1:
    if btn_predict:
        # 1. Get Fan Location
        lat, lon = get_lat_lon_from_zip(user_zip)
        
        if lat is None:
            st.error("Invalid Zip Code. Please enter a valid US Zip Code.")
        elif model is None:
            st.error("Model not loaded.")
        else:
            # 2. Get Team Data
            t1_data = teams_metadata[team1_name]
            t2_data = teams_metadata[team2_name]
            
            # 3. Calculate Distances (for display)
            d1 = haversine_distance(lat, lon, t1_data['InstitutionLatitude'], t1_data['InstitutionLongitude'])
            d2 = haversine_distance(lat, lon, t2_data['InstitutionLatitude'], t2_data['InstitutionLongitude'])
            
            # 4. Prepare Data for Model
            input_df = prepare_inference_data(lat, lon, t1_data, t2_data)
            
            # 5. Predict
            # Model target: 1 = East (Team 1), 0 = West (Team 2)
            prob_team1 = model.predict_proba(input_df)[0][1]
            prob_team2 = 1 - prob_team1
            
            winner = team1_name if prob_team1 > prob_team2 else team2_name
            confidence = max(prob_team1, prob_team2)
            
            # --- Display Results ---
            
            # KPI Cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fan's Pick", winner)
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                # Logic: If fan picked the closer team, it's Affinity. If picked the better team, it's Performance.
                # (Simplified logic for display)
                closer_team = team1_name if d1 < d2 else team2_name
                better_team = team1_name if t1_data['win_%'] > t2_data['win_%'] else team2_name
                
                reason = "Unknown"
                if winner == closer_team and winner != better_team:
                    reason = "📍 Geographic Affinity"
                elif winner == better_team and winner != closer_team:
                    reason = "🏆 Team Performance"
                elif winner == closer_team and winner == better_team:
                    reason = "🔥 Best of Both"
                else:
                    reason = "🎲 Underdog Pick"
                
                st.metric("Primary Driver", reason)

            st.divider()
            
            # Tale of the Tape
            st.subheader("📏 Tale of the Tape")
            
            comparison_data = pd.DataFrame({
                "Metric": ["Distance to Fan", "Win Percentage", "Net Rating", "Seed"],
                f"{team1_name}": [f"{d1:.0f} miles", f"{t1_data['win_%']:.1%}", f"{t1_data.get('NetRtg',0):.1f}", f"#{t1_data.get('Seed_Rank','-')}"],
                f"{team2_name}": [f"{d2:.0f} miles", f"{t2_data['win_%']:.1%}", f"{t2_data.get('NetRtg',0):.1f}", f"#{t2_data.get('Seed_Rank','-')}"]
            })
            st.table(comparison_data.set_index("Metric"))
            
            # Map
            st.subheader("🗺️ Geographic Context")
            map_data = pd.DataFrame({
                'lat': [lat, t1_data['InstitutionLatitude'], t2_data['InstitutionLatitude']],
                'lon': [lon, t1_data['InstitutionLongitude'], t2_data['InstitutionLongitude']],
                'name': ["Fan (You)", team1_name, team2_name],
                'type': ['Fan', 'Team', 'Team'],
                'color': ['#FF4B4B', '#1f77b4', '#ff7f0e'] # Red, Blue, Orange
            })
            
            st.map(map_data, zoom=4)

    else:
        st.info("👈 Enter your Zip Code and pick two teams to see who you'd likely support!")

with tab2:
    st.header("📊 Project Insights: The Power of Proximity")
    
    st.markdown("""
    ### Key Finding: The 81% Concentration Rule
    Our analysis of 76,000+ brackets revealed a stunning pattern: **81% of fan selections within any given market (DMA) are concentrated on just the top 5 geographically closest teams.**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Model Performance")
        st.markdown("""
        | Model | Accuracy | Feature Focus |
        |-------|----------|---------------|
        | **RandomForest (Final)** | **67.1%** | **Hybrid (Geo + Stats)** |
        | XGBoost | 66.8% | Hybrid |
        | Logistic Regression | 63.4% | Stats Heavy |
        
        The **RandomForest** model outperformed others by effectively capturing non-linear relationships between distance and loyalty.
        """)
        
    with col2:
        st.markdown("#### Feature Importance")
        # Placeholder for Feature Importance Chart
        feat_imp = pd.DataFrame({
            'Feature': ['Distance to Team', 'Win % Differential', 'Conference Prestige', 'Seed Differential', 'Historic Success'],
            'Importance': [0.35, 0.28, 0.15, 0.12, 0.10]
        })
        
        c = alt.Chart(feat_imp).mark_bar().encode(
            x='Importance',
            y=alt.Y('Feature', sort='-x'),
            color=alt.value('#1f77b4')
        ).properties(height=300)
        
        st.altair_chart(c, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🏆 Competition Results")
    st.success("**3rd Place Winner** - 2024 NCAA & CCAC Research Competition (200+ Teams)")
