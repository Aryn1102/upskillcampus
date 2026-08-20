"""Project 10 - Quality Prediction in a Mining Process."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA_PATH = Path("MiningProcess_Flotation_Plant_Database.csv")
OUTPUT_DIR = Path("outputs_mining")
OUTPUT_DIR.mkdir(exist_ok=True)

def find_time_col(df):
    for c in df.columns:
        n = str(c).strip().lower()
        if n in {"date","datetime","timestamp","time"} or "date" in n or "time" in n:
            return c
    return None

def find_target(df):
    for c in df.columns:
        n = str(c).strip().lower()
        if "silica" in n and "concentrate" in n:
            return c
    raise ValueError("Silica concentrate target column was not found.")

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    time_col = find_time_col(df)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col).reset_index(drop=True)
    for c in df.columns:
        if df[c].dtype == "object":
            x = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            if x.notna().mean() > .80:
                df[c] = x
    return df, time_col

def add_features(df, time_col, target):
    df = df.copy()
    if time_col:
        df["hour"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.dayofweek
        df["month"] = df[time_col].dt.month
    for lag in range(1, 4):
        df[f"target_lag_{lag}"] = df[target].shift(lag)
    return df

def evaluate(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return {"model": name, "MAE": mean_absolute_error(yte,p),
            "RMSE": np.sqrt(mean_squared_error(yte,p)), "R2": r2_score(yte,p)}, p

def main():
    df, time_col = load_data()
    target = find_target(df)
    df = add_features(df, time_col, target)
    numeric = df.select_dtypes(include=np.number).dropna(subset=[target])
    X = numeric.drop(columns=[target])
    y = numeric[target]
    split = int(len(X)*.80)
    Xtr, Xte, ytr, yte = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    models = {
        "LinearRegression": Pipeline([("imputer",SimpleImputer(strategy="median")),
                                       ("scale",StandardScaler()),("model",LinearRegression())]),
        "RandomForest": Pipeline([("imputer",SimpleImputer(strategy="median")),
                                  ("model",RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1))]),
        "GradientBoosting": Pipeline([("imputer",SimpleImputer(strategy="median")),
                                      ("model",GradientBoostingRegressor(random_state=42))])
    }
    results, pred = [], pd.DataFrame({"actual": yte.values})
    for name, model in models.items():
        m, p = evaluate(name, model, Xtr, Xte, ytr, yte)
        results.append(m); pred[name] = p
    pd.DataFrame(results).sort_values("RMSE").to_csv(OUTPUT_DIR/"model_metrics.csv", index=False)
    pred.to_csv(OUTPUT_DIR/"test_predictions.csv", index=False)
    df.isna().sum().sort_values(ascending=False).to_csv(OUTPUT_DIR/"missing_values.csv")
    if time_col:
        plt.figure(figsize=(11,5))
        plt.plot(df[time_col], df[target], linewidth=.8)
        plt.xlabel("Time"); plt.ylabel(target); plt.title("Silica Concentration Over Time")
        plt.tight_layout(); plt.savefig(OUTPUT_DIR/"target_over_time.png", dpi=150); plt.close()
    print(pd.DataFrame(results).sort_values("RMSE").to_string(index=False))

if __name__ == "__main__":
    main()
