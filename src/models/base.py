from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BaseModel(ABC):
    """所有模型的基类。每位成员继承此类，实现自己的模型。

    使用方式:
        class MyModel(BaseModel):
            name = "my_model"
            def build_pipeline(self):
                return Pipeline([...])
            def param_grid(self):
                return {"model__alpha": [0.01, 0.1, 1.0]}
    """

    name: str = ""

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """返回一个 sklearn Pipeline，最后一步命名为 'model'。"""
        ...

    def param_grid(self) -> dict:
        """返回超参搜索空间（可选）。键名需带 'model__' 前缀。"""
        return {}

    @staticmethod
    def default_scaler():
        return StandardScaler()
