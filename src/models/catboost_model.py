from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    name = "catboost"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', CatBoostRegressor(
                iterations=800, depth=6, learning_rate=0.1,
                early_stopping_rounds=50,
                verbose=0, random_seed=42,
            )),
        ])

    def param_grid(self):
        return {
            'model__depth': [4, 6, 8],
            'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
            'model__l2_leaf_reg': [1, 3, 5, 7, 9],
            'model__bagging_temperature': [0, 1],
        }
