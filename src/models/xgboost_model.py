from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    name = "xgboost"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', XGBRegressor(
                n_estimators=2000, max_depth=4, learning_rate=0.05,
                n_jobs=1, random_state=42, verbosity=0,
            )),
        ])

    def param_grid(self):
        return {}  # 已用历史最优参数，跳过网格搜索
