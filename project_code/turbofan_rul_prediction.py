"""Project 6 - Remaining Useful Life Prediction for Turbofan Engines."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
DATA_PATH = Path("train_FD001.txt")
OUTPUT_DIR = Path("outputs_turbofan")
OUTPUT_DIR.mkdir(exist_ok=True)

BASE = ["unit_id","cycle","op_setting_1","op_setting_2","op_setting_3"]
SENSORS = [f"sensor_{i}" for i in range(1,22)]
COLUMNS = BASE + SENSORS

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :len(COLUMNS)]
    df.columns = COLUMNS[:df.shape[1]]
    return df.dropna(how="all").reset_index(drop=True)

def add_rul(df):
    df = df.copy()
    last = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = last - df["cycle"]
    return df

def add_rolling_features(df, window=5):
    df = df.copy()
    for s in [c for c in SENSORS if c in df.columns]:
        g = df.groupby("unit_id")[s]
        df[f"{s}_mean"] = g.transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{s}_std"] = g.transform(lambda x: x.rolling(window, min_periods=2).std())
    return df

def evaluate(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr,ytr); p=model.predict(Xte)
    return {"model":name,"MAE":mean_absolute_error(yte,p),
            "RMSE":np.sqrt(mean_squared_error(yte,p)),"R2":r2_score(yte,p)},p

def main():
    df = add_rul(load_data())
    df = add_rolling_features(df)
    units = sorted(df.unit_id.unique())
    cut = max(1, int(len(units)*.80))
    tr = df[df.unit_id.isin(units[:cut])].copy()
    te = df[df.unit_id.isin(units[cut:])].copy()
    features = [c for c in df.select_dtypes(include=np.number).columns if c not in {"RUL","unit_id"}]
    Xtr,ytr,Xte,yte = tr[features],tr["RUL"],te[features],te["RUL"]
    models = {
        "Ridge": Pipeline([("imputer",SimpleImputer(strategy="median")),
                           ("scale",StandardScaler()),("model",Ridge(alpha=1.0))]),
        "RandomForest": Pipeline([("imputer",SimpleImputer(strategy="median")),
                                  ("model",RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1))]),
        "GradientBoosting": Pipeline([("imputer",SimpleImputer(strategy="median")),
                                      ("model",GradientBoostingRegressor(random_state=42))])
    }
    results=[]; pred=pd.DataFrame({"unit_id":te.unit_id.values,"cycle":te.cycle.values,"actual_RUL":yte.values})
    for name,model in models.items():
        m,p=evaluate(name,model,Xtr,Xte,ytr,yte); results.append(m); pred[name+"_RUL"]=p
    pd.DataFrame(results).sort_values("RMSE").to_csv(OUTPUT_DIR/"model_metrics.csv",index=False)
    pred.to_csv(OUTPUT_DIR/"test_predictions.csv",index=False)
    print(pd.DataFrame(results).sort_values("RMSE").to_string(index=False))

if __name__ == "__main__":
    main()
