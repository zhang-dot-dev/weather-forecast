from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    name = "catboost"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.1,
                verbose=0, random_seed=42,
            )),
        ])

    def param_grid(self):
        return {
            'model__iterations': [300, 500, 800],
            'model__depth': [4, 6, 8],
            'model__learning_rate': [0.03, 0.1, 0.2],
        }
