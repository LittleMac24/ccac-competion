# 🏀 March Madness: Predicting Fan Predictions

<p align="center">
  <strong>🏆 3rd Place Winner @ 2024 NCAA & CCAC Research Competition (out of 200+ teams) 🏆</strong>
</p>

---

## 📋 Project Overview

A **behavioral analytics** project that uses machine learning to predict fan bracket selections for the NCAA March Madness tournament. This project uncovers **affinity bias** in fan decision-making, revealing that geographic proximity and regional loyalty drive bracket choices more than objective team performance metrics.

**Project Type**: Data Science  
**Domain**: Sports Analytics, Behavioral Analytics  
**Competition**: 2024 NCAA & CCAC Research Competition

---

## 🎯 Problem Statement

**Research Question**: Can we predict which teams fans will select in their March Madness brackets, and what factors drive these selections?

**Key Hypothesis**: Fan selections are driven more by regional affinity and geographic proximity than by objective team performance metrics.

---

## 🔑 Key Findings

### 1. Regional Affinity Dominance (81% Concentration)
- On average, **81% of fan selections within each DMA** (Designated Market Area) were concentrated among just the **top 5 teams**
- This reveals strong **regional loyalty** that overrides objective team performance
- **Business Implication**: Geographic targeting is more effective than performance-based marketing

### 2. Geographic Proximity > Win Percentage
- **Geographic closeness** (Haversine distance) is a stronger predictor than team win percentage
- Fans consistently favor teams closer to their location, even when those teams have lower win rates
- **Model Performance**: 67% accuracy for semifinal predictions using geographic features

### 3. Top Teams Show Consistent Dominance
- Houston, Purdue, Marquette, Tennessee, Kentucky, and Creighton were the most frequently selected teams
- This pattern holds across different regions, suggesting national brand influence

---

## 📊 Dataset

- **Training Set**: ~76,000+ bracket entries
- **Features**: 
  - Customer demographics (DMA codes, postal codes, geographic coordinates)
  - Team performance metrics (win %, regular season stats, attendance)
  - Institution information (enrollment, conference, location)
  - KenPom advanced statistics
- **Target Variables**:
  - `SemifinalWinner_East_West`
  - `SemifinalWinner_South_Midwest`
  - `NationalChampion`

---

## 🛠️ Methodology

### 1. Exploratory Data Analysis (EDA)
- Analyzed distribution of team selections across DMAs
- Identified regional concentration patterns (81% in top 5 teams)
- Visualized geographic vs. performance-based selection patterns

### 2. Feature Engineering
- **Geographic Closeness**: Haversine distance calculation between customer location and team locations
- **Win Percentage Differential**: Difference in win rates between competing teams
- **Regional Affinity Scores**: DMA-based team preference metrics
- **Institution Metrics**: Enrollment, attendance, conference strength

### 3. Model Development
Multiple machine learning approaches were tested:

| Model | Semifinal (E-W) Accuracy | Semifinal (S-M) Accuracy | National Champion Accuracy |
|-------|-------------------------|-------------------------|---------------------------|
| RandomForest | 67.1% | 62.8% | 43.3% |
| XGBoost | ~67% | ~63% | 61.6% |
| CatBoost | ~67% | ~63% | ~62% |
| Logistic Regression | 63.4% | 60.7% | 57.4% |

**Best Approach**: Ensemble of RandomForest with feature selection (top 10 features) + XGBoost for final predictions

### 4. Model Pipeline
```
Data Loading → Feature Engineering → Preprocessing (StandardScaler, OneHotEncoder) 
→ Model Training → Hyperparameter Tuning → Cross-Validation → Prediction
```

---

## 💻 Technical Stack

- **Languages**: Python 3.x
- **Libraries**:
  - `pandas`, `numpy` - Data manipulation
  - `scikit-learn` - Machine learning pipeline
  - `xgboost`, `catboost` - Gradient boosting
  - `matplotlib`, `seaborn`, `plotnine` - Visualization
  - `imbalanced-learn` - Handling class imbalance (SMOTE)

---

## 📁 Project Structure

```
ccac-competion/
├── data/                          # Raw and processed datasets
│   ├── training_set.csv
│   ├── bracket_training.csv
│   ├── institutions.csv
│   └── ...
├── notebooks/                     # Jupyter notebooks
│   ├── eda-analysis.ipynb        # Exploratory data analysis
│   ├── model-1.ipynb             # Logistic regression baseline
│   ├── ccac.ipynb                # Main modeling notebook
│   └── ...
├── models/                        # Production-ready model scripts
│   └── baseline-model.py
├── output/                        # Predictions and results
│   └── competition-results/
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn plotnine imbalanced-learn
```

### Running the Analysis
```bash
# Run the analysis script
python analyze_project.py

# Or open notebooks in Jupyter
jupyter notebook eda-analysis.ipynb
```

---

## 📈 Results & Performance

### Competition Results
- **Rank**: 3rd Place
- **Competition**: 2024 NCAA & CCAC Research Competition
- **Participants**: 200+ teams
- **Evaluation Metric**: Prediction accuracy on held-out test set

### Model Performance
- **Semifinal Predictions**: 63-67% accuracy
- **National Champion**: 43-62% accuracy (varies by model)
- **Key Insight**: Geographic features significantly outperform performance-only models

---

## 💡 Business Applications

### 1. Targeted Marketing
- Use regional affinity data to target marketing campaigns
- Identify high-engagement DMAs for specific teams
- Optimize ad spend based on geographic fan loyalty

### 2. Content Strategy
- Sports media companies can tailor content to regional preferences
- Create team-specific content for high-affinity regions
- Optimize broadcast schedules based on regional interest

### 3. Sponsorship Alignment
- Brands can align with teams that have strong regional presence
- Identify sponsorship opportunities in high-engagement markets

### 4. Fan Engagement
- Predict which teams fans will support
- Personalize bracket recommendations based on location
- Enhance user experience with location-aware features

---

## 🔬 Key Insights for Data Science

1. **Behavioral Bias is Predictable**: Even "irrational" fan behavior follows patterns that can be modeled
2. **Geographic Features Matter**: Location-based features often outperform traditional performance metrics
3. **Ensemble Methods Win**: Combining multiple models improves robustness
4. **Feature Engineering is Critical**: Domain-specific features (Haversine distance) provide significant lift

---

## 📝 Future Work

- [ ] Deploy model as REST API (Flask/FastAPI)
- [ ] Create interactive dashboard (Streamlit/Dash)
- [ ] Incorporate real-time team performance data
- [ ] Add sentiment analysis from social media
- [ ] Build recommendation system for bracket suggestions
- [ ] Expand to other sports/competitions

---

## 👤 Author

**Michael Whitfield**

- Competition: 3rd Place @ 2024 NCAA & CCAC Research Competition
- Project Type: Data Science / Behavioral Analytics
- Domain: Sports Analytics

---

## 📄 License

This project was created for academic/competition purposes.

---

## 🙏 Acknowledgments

- NCAA & CCAC for organizing the competition
- Data providers for the comprehensive dataset
- Competition judges and organizers

---

## 📚 References

- KenPom Analytics for advanced basketball statistics
- Haversine formula for geographic distance calculations
- Scikit-learn documentation for ML pipeline implementation

---

<p align="center">
  <strong>Built with ❤️ for the love of data science and basketball</strong>
</p>
