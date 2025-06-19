March Madness: Analyzing School Affinity & Predicting Championship Rounds
1. Project Overview
This project was developed for the NCAA & CCAC Undergraduate Research Competition, where it achieved a 3rd place finish out of 200+ teams.

The core objective is to analyze submitted NCAA March Madness brackets to address a two-part challenge:

Behavioral Analysis: Determine if school affinity, regional biases, or other customer characteristics play a significant role in how participants make their predictions.

Predictive Modeling: Develop and evaluate machine learning models to forecast the final two rounds of the championship, leveraging historical data and insights from the bracket submissions.

This repository contains the exploratory data analysis and initial modeling work, demonstrating an end-to-end workflow from raw data to actionable insights and predictive evaluation. The findings from this analysis are intended to be visualized and communicated through a dashboarding tool like Tableau, as per the competition guidelines.

2. Key Findings & Results
Affinity Analysis: The exploratory data analysis uncovered statistically significant evidence of regional and school-based biases in bracket submissions. For example, participants were more likely to predict teams from their own geographic region or alma mater would advance further than statistical models would suggest.

Model Performance: A Random Forest Classifier proved to be the most effective model for forecasting game outcomes in the final rounds, balancing predictive accuracy with interpretability.

Key Predictors: The analysis identified that while team performance metrics (e.g., season record, strength of schedule) were the primary drivers of predictions, the "affinity bias" was a measurable secondary factor that could be used to segment and understand participant behavior.


(Specific accuracy scores and model details can be found in the Jupyter Notebook.)

3. Tech Stack
Language: Python 3.9

Libraries:

pandas & numpy for data manipulation and analysis.

matplotlib & seaborn for data visualization and exploratory analysis.

scikit-learn for machine learning model development and evaluation.

Environment: Jupyter Notebook

4. Setup and How to Run
To explore this project's analysis, you can run the Jupyter Notebook locally.

1. Clone the repository:

git clone https://github.com/LittleMac24/ccac-competion.git
cd ccac-competion

2. Create and activate a virtual environment (recommended):

# Create a virtual environment
python3 -m venv venv
# Activate it (Mac/Linux)
source venv/bin/activate
# Or activate it (Windows)
# .\venv\Scripts\activate

3. Install the required dependencies:
To ensure reproducibility, a requirements.txt file should be generated from your project environment by running pip freeze > requirements.txt.

# Install dependencies from the file (if it exists)
pip install -r requirements.txt
# Or install manually
pip install pandas numpy matplotlib seaborn scikit-learn jupyterlab

4. Launch Jupyter and open the notebook:

jupyter lab CCAC-Final.ipynb

5. Future Work & Production-Level Improvements
While this repository contains the completed analysis for the competition, the following steps would be taken to evolve it into a production-level system:

Code Modularization: Refactor the code from the Jupyter Notebook into a src/ directory with separate Python scripts for data loading (data_loader.py), preprocessing (preprocessor.py), and model training (model_trainer.py).

Model Deployment: Containerize the trained Random Forest model using Docker and expose it via a REST API (using Flask or FastAPI) to allow for real-time predictions.

Dashboard Integration: Create a live data pipeline that feeds model predictions and analysis results into a Tableau or Power BI dashboard for business stakeholders.

Hyperparameter Tuning: Implement GridSearchCV or RandomizedSearchCV to systematically fine-tune the model's hyperparameters for improved predictive accuracy.

Automated Testing: Develop a suite of unit tests (pytest) for the data processing and modeling functions to ensure the reliability and robustness of the code.
