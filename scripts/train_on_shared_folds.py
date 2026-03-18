#!/usr/bin/env python3
"""
组员本地调参脚本：加载 shared_cv_folds 中的 3 折数据，在各自电脑上训练并找到最佳参数。

用法:
  python scripts/train_on_shared_folds.py --data_dir outputs/shared_cv_folds --model lightgbm
  python scripts/train_on_shared_folds.py --data_dir outputs/shared_cv_folds --model lasso --n_iter 30

依赖: pandas, numpy, scikit-learn；若用 lightgbm/xgboost/catboost 需额外安装对应库。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler

# 可选模型
try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from catboost import CatBoostRegressor
    HAS_CB = True
except ImportError:
    HAS_CB = False

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_RF = True
except ImportError:
    HAS_RF = True  # usually available


def load_folds(data_dir: Path):
    """加载 meta.json 和 3 个 fold 的 train/val 数据。"""
    data_dir = Path(data_dir)
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    n_folds = meta["n_folds"]
    folds = []
    for i in range(n_folds):
        fold_dir = data_dir / f"fold_{i}"
        X_tr = pd.read_csv(fold_dir / "X_train.csv")
        y_tr = pd.read_csv(fold_dir / "y_train.csv").squeeze()
        X_val = pd.read_csv(fold_dir / "X_val.csv")
        y_val = pd.read_csv(fold_dir / "y_val.csv").squeeze()
        folds.append((X_tr, y_tr, X_val, y_val))
    return meta, folds


def build_model_and_grid(name: str, random_state: int = 42):
    """返回 (Pipeline, param_grid)。"""
    scaler = StandardScaler()
    if name == "lasso":
        model = Lasso(random_state=random_state, max_iter=10000)
        grid = {"model__alpha": [0.001, 0.01, 0.1, 1.0]}
    elif name == "elasticnet":
        model = ElasticNet(random_state=random_state, max_iter=10000)
        grid = {
            "model__alpha": [0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8, 1.0],
        }
    elif name == "lightgbm" and HAS_LGB:
        model = LGBMRegressor(
            n_estimators=800, max_depth=-1, learning_rate=0.1,
            num_leaves=31, n_jobs=-1, random_state=random_state, verbose=-1,
        )
        grid = {
            "model__num_leaves": [7, 15, 23, 31],
            "model__n_estimators": [500, 800, 1200],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        }
    elif name == "xgboost" and HAS_XGB:
        model = XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.1,
            n_jobs=-1, random_state=random_state, verbosity=0,
        )
        grid = {
            "model__n_estimators": [500, 1000, 1500],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.05, 0.1, 0.2],
        }
    elif name == "catboost" and HAS_CB:
        model = CatBoostRegressor(
            iterations=2000, depth=6, learning_rate=0.1,
            verbose=0, random_seed=random_state,
        )
        grid = {
            "model__depth": [4, 5, 6],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__l2_leaf_reg": [1, 3, 5],
        }
    elif name == "random_forest" and HAS_RF:
        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        grid = {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [10, 15, 20],
            "model__min_samples_leaf": [5, 10, 20],
        }
    else:
        raise ValueError(
            f"未知或未安装模型: {name}. "
            f"可用: lasso, elasticnet"
            + (", lightgbm" if HAS_LGB else "")
            + (", xgboost" if HAS_XGB else "")
            + (", catboost" if HAS_CB else "")
            + (", random_forest" if HAS_RF else "")
        )
    pipe = Pipeline([("scaler", scaler), ("model", model)])
    return pipe, grid


def compute_cv_rmse(pipe, folds):
    """在 3 折上训练-预测，返回平均 RMSE。"""
    from sklearn.base import clone
    rmse_list = []
    for X_tr, y_tr, X_val, y_val in folds:
        p = clone(pipe)
        p.fit(X_tr, y_tr)
        pred = p.predict(X_val)
        rmse_list.append(np.sqrt(mean_squared_error(y_val, pred)))
    return np.mean(rmse_list)


def main():
    parser = argparse.ArgumentParser(description="在 shared_cv_folds 上做 3 折 CV 调参")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="outputs/shared_cv_folds",
        help="shared_cv_folds 目录路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lasso",
        choices=["lasso", "elasticnet", "lightgbm", "xgboost", "catboost", "random_forest"],
        help="要调参的模型",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=20,
        help="随机采样的参数组合数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "meta.json").exists():
        print(f"错误: 未找到 {data_dir}/meta.json，请先运行 notebook 中的导出 cell。")
        sys.exit(1)

    meta, folds = load_folds(data_dir)
    print(f"加载 {len(folds)} 折数据，特征: {meta['selected_features'][:3]}... 等共 {len(meta['selected_features'])} 个\n")

    pipe, param_grid = build_model_and_grid(args.model, random_state=args.seed)
    param_list = list(
        ParameterSampler(param_grid, n_iter=args.n_iter, random_state=args.seed)
    )

    best_rmse = float("inf")
    best_params = None

    for i, params in enumerate(param_list):
        pipe.set_params(**params)
        rmse = compute_cv_rmse(pipe, folds)
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(param_list)}] CV RMSE: {rmse:.4f}")

    print(f"\n{'='*50}")
    print(f"  模型: {args.model}")
    print(f"  最佳 CV RMSE: {best_rmse:.4f}")
    print(f"  最佳参数: {json.dumps(best_params, ensure_ascii=False, indent=2)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
