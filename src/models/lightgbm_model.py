from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', LGBMRegressor(
                n_estimators=800, max_depth=-1, learning_rate=0.1,
                num_leaves=31,
                n_jobs=-1, random_state=42, verbose=-1,
            )),
        ])

    def param_grid(self):
        return {
            'model__num_leaves': [15, 23, 31],
            'model__n_estimators': [300,500, 800],
            'model__learning_rate': [0.03, 0.05, 0.1, 0.2],
        }
