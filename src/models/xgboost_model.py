from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    name = "xgboost"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=42, verbosity=0,
            )),
        ])

    def param_grid(self):
        return {
            'model__n_estimators': [300, 500, 800],
            'model__max_depth': [4, 6, 8],
            'model__learning_rate': [0.03, 0.1, 0.2],
        }
