from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class ElasticNetModel(BaseModel):
    name = "elasticnet"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        ])

    def param_grid(self):
        return {
            'model__alpha': [0.01, 0.05, 0.2, 0.1, 0.5],
            'model__l1_ratio': [0.1,0.2, 0.5, 0.8],
        }
