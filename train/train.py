import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Load dataset
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Going up one level from 'train' folder
    data_path = os.path.join(project_root, 'data', 'Advertising.csv')

    df = pd.read_csv(data_path)
    df = df.dropna()

    # Outlier handling using IQR
    def remove_outliers_iqr(data, cols):
        cleaned = data.copy()
        for col in cols:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
        return cleaned

    df = remove_outliers_iqr(df, ["TV", "Radio", "Newspaper", "Sales"])

    # Feature engineering
    df["Total_Spend"] = df["TV"] + df["Radio"] + df["Newspaper"]
    df["TV_Radio"] = df["TV"] * df["Radio"]
    df["TV_Newspaper"] = df["TV"] * df["Newspaper"]
    df["Radio_Newspaper"] = df["Radio"] * df["Newspaper"]
    df["TV_Sq"] = df["TV"] ** 2
    df["Radio_Sq"] = df["Radio"] ** 2
    df["Newspaper_Sq"] = df["Newspaper"] ** 2

    X = df[[ 
        "TV", "Radio", "Newspaper", 
        "Total_Spend", 
        "TV_Radio", "TV_Newspaper", "Radio_Newspaper", 
        "TV_Sq", "Radio_Sq", "Newspaper_Sq" 
    ]]
    y = df["Sales"]

    return X, y

# Function to compare and choose best model
def find_best_model(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=500, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "MAPE": mape
        }

    results_df = pd.DataFrame(results).T
    print("📊 Model Comparison:")
    print(results_df)

    best_model_name = results_df["MAE"].idxmin()
    best_model = models[best_model_name]

    return best_model_name, best_model

# Function to train the best model on the full dataset and save it
def train_and_save_best_model(best_model, X, y):
    best_model.fit(X, y)

    # Set up path for saving
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_dir = os.path.join(project_root, 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Save using model class name
    model_filename = os.path.join(model_dir, f"{best_model.__class__.__name__.lower()}_model.pkl")
    joblib.dump(best_model, model_filename)

    print(f"✅ Best Model saved to '{model_filename}'")

# Main function
def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    best_model_name, best_model = find_best_model(X_train, X_test, y_train, y_test)

    train_and_save_best_model(best_model, X, y)

    print(f"🏆 Best Model Selected: {best_model_name}")

if __name__ == "__main__":
    main()
