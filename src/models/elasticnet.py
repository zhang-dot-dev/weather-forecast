from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class ElasticNetModel(BaseModel):
    name = "elasticnet"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000)),
        ])

    def param_grid(self):
        return {}  # 已用历史最优参数，跳过网格搜索
