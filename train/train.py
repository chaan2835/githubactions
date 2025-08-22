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
    # Relative path to the dataset
    data_path = os.path.join("data", "Advertising.csv")

    # Load dataset using the relative path
    df = pd.read_csv(data_path)
    df = df.dropna()
    
    # Outlier handling (IQR method)
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

    # Features & Target
    X = df[[ 
        "TV", "Radio", "Newspaper", 
        "Total_Spend", 
        "TV_Radio", "TV_Newspaper", "Radio_Newspaper", 
        "TV_Sq", "Radio_Sq", "Newspaper_Sq" 
    ]]
    y = df["Sales"]

    return X, y

# Function to find the best model based on the metrics
def find_best_model(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=500, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42),
        "XGBoost Regressor": xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42)
    }

    results = {}

    # Train each model and evaluate
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate the model
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

    # Convert results to a DataFrame for comparison
    results_df = pd.DataFrame(results).T
    print("Model Comparison:")
    print(results_df)

    # Select the best model based on the lowest MAE (or RMSE)
    best_model_name = results_df["MAE"].idxmin()  # Choosing the model with the lowest MAE as the best model
    best_model = models[best_model_name]

    return best_model_name, best_model

# Function to train the best model on the entire dataset and save it
def train_and_save_best_model(best_model, X, y):
    # Train the best model on the entire dataset
    best_model.fit(X, y)

    # Save the best model as a .pkl file with the model name
    os.makedirs("model", exist_ok=True)
    model_filename = f"model/{best_model.__class__.__name__.lower()}_model.pkl"  # Save with model name
    joblib.dump(best_model, model_filename)

    print(f"✅ Best Model saved to '{model_filename}'")

# Main function to execute the workflow
def main():
    # Load data
    X, y = load_data()

    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 1: Find the best model
    best_model_name, best_model = find_best_model(X_train, X_test, y_train, y_test)

    # Step 2: Train the best model on the full dataset and save the model
    train_and_save_best_model(best_model, X, y)

    print(f"Best Model: {best_model_name}")

# Run the main function
if __name__ == "__main__":
    main()
