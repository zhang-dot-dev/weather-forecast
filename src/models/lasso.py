from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class LassoModel(BaseModel):
    name = "lasso"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', Lasso(alpha=0.1, max_iter=5000)),
        ])

    def param_grid(self):
        return {
            'model__alpha': [0.01, 0.05, 0.1, 0.5, 1.0],
        }
