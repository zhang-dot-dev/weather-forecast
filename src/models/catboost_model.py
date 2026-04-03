from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from src.models.base import BaseModel


class CatBoostModel(BaseModel):
    name = "catboost"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', CatBoostRegressor(
                iterations=8000,
                learning_rate=0.02,
                depth=8,
                min_data_in_leaf=20,
                l2_leaf_reg=5.0,
                random_strength=1.0,
                bagging_temperature=0.5,
                rsm=0.8,
                early_stopping_rounds=300,
                verbose=0, random_seed=42,
            )),
        ])

    def param_grid(self):
        return {}  # 暂时跳过网格搜索，直接用 build_pipeline 的默认参数
