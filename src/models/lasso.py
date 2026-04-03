from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class LassoModel(BaseModel):
    name = "lasso"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', Lasso(alpha=0.01, max_iter=10000)),
        ])

    def param_grid(self):
        return {}  # 已用历史最优参数，跳过网格搜索
