import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from data_loader import load_cinema_data

def preprocess_and_train():
    """
    Loads data, cleans it, engineers features, trains a model, and returns
    the trained model alongside evaluation metrics.
    """
    # 1. Load Data
    df = load_cinema_data()
    
    # 2. Clean and Feature Engineer
    # Drop columns that aren't useful for general regression or cause leakage
    # 'total_sales' is directly correlated with tickets_sold * price, so we drop it to avoid cheating.
    cols_to_drop = ['date', 'quarter', 'show_time', 'total_sales']
    df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Fill any missing values safely
    df_cleaned = df_cleaned.dropna()
    
    # Define Target and Features
    # Target: tickets_sold (The number of tickets sold for a show)
    X = df_cleaned.drop(columns=['tickets_sold'])
    y = df_cleaned['tickets_sold']
    
    # Keep track of feature names for the Streamlit UI inputs
    feature_names = X.columns.tolist()
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train LightGBM Model (Fast, accurate, and lightweight for cloud deployment)
    model = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    metrics = {"MAE": mae, "R2": r2}
    
    return model, feature_names, metrics

if __name__ == "__main__":
    # Test training pipeline locally
    print("Starting model training pipeline...")
    model, features, metrics = preprocess_and_train()
    print("Training Complete!")
    print(f"Features used: {features}")
    print(f"Model Performance -> Mean Absolute Error: {metrics['MAE']:.2f}, R2 Score: {metrics['R2']:.2f}")