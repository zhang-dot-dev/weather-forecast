from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    name = "random_forest"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', RandomForestRegressor(
                n_estimators=2000, max_depth=14,
                min_samples_leaf=5,
                n_jobs=-1, random_state=42,
            )),
        ])

    def param_grid(self):
        return {}  # 已用历史最优参数，跳过网格搜索
