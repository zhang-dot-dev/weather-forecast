from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    name = "random_forest"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', RandomForestRegressor(
                n_estimators=200, max_depth=10,
                n_jobs=-1, random_state=42,
            )),
        ])

    def param_grid(self):
        return {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [6, 10, 15],
            'model__min_samples_leaf': [1, 5, 10],
        }
