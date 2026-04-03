from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', LGBMRegressor(
                n_estimators=2000, max_depth=-1, learning_rate=0.05,
                num_leaves=15,
                n_jobs=1, random_state=42, verbose=-1,
            )),
        ])

    def param_grid(self):
        return {}  # 已用历史最优参数，跳过网格搜索
