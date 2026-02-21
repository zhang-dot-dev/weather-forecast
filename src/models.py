import joblib
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import *

def build_models():
    return {
        'lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=LASSO_ALPHA, max_iter=5000))
        ]),
        'elasticnet': Pipeline([
            ('scaler', StandardScaler()),
            ('model', ElasticNet(alpha=ENET_ALPHA, l1_ratio=ENET_L1_RATIO))
        ]),
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                n_jobs=-1,
                random_state=RANDOM_STATE
            ))
        ])
    }

def save_model(model, name):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{name}.pkl")

def load_model(name):
    return joblib.load(MODEL_DIR / f"{name}.pkl")