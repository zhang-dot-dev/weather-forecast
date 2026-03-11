"""
自动发现 src/models/ 下所有 BaseModel 子类。

新增模型只需在此目录下创建 .py 文件并继承 BaseModel，无需修改任何其他文件。
"""
import importlib
import pkgutil
import joblib
from pathlib import Path

from config import MODEL_DIR
from src.models.base import BaseModel


def discover_models() -> dict[str, BaseModel]:
    """扫描当前包，返回 {name: instance} 字典。"""
    package_dir = Path(__file__).parent
    models = {}

    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name == 'base':
            continue
        module = importlib.import_module(f"src.models.{info.name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type)
                    and issubclass(attr, BaseModel)
                    and attr is not BaseModel):
                instance = attr()
                models[instance.name] = instance

    return models


def build_models() -> dict:
    """返回 {name: Pipeline} 字典，兼容旧接口。"""
    return {name: m.build_pipeline() for name, m in discover_models().items()}


def save_model(model, name, run_id=None):
    """保存模型。指定 run_id 时存到 MODEL_DIR/<run_id>/ 子目录。"""
    target = MODEL_DIR / run_id if run_id else MODEL_DIR
    target.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target / f"{name}.pkl")


def load_model(name, run_id=None):
    """加载模型。指定 run_id 可恢复某次历史运行的模型。"""
    target = MODEL_DIR / run_id if run_id else MODEL_DIR
    return joblib.load(target / f"{name}.pkl")


def list_runs():
    """列出所有保存过模型的 run_id。"""
    return sorted([d.name for d in MODEL_DIR.iterdir() if d.is_dir()])
