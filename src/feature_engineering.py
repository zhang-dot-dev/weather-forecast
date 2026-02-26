import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import LASSO_ALPHA, ENET_ALPHA, ENET_L1_RATIO, RF_MAX_DEPTH, RANDOM_STATE, FIG_DIR


def _norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-10)


def lasso_importance(X_scaled, y, features):
    """Lasso 特征重要性，返回归一化得分 Series。"""
    model = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
    model.fit(X_scaled, y)
    return _norm(pd.Series(np.abs(model.coef_), index=features))


def elasticnet_importance(X_scaled, y, features):
    """ElasticNet 特征重要性，返回归一化得分 Series。"""
    model = ElasticNet(alpha=ENET_ALPHA, l1_ratio=ENET_L1_RATIO, max_iter=5000)
    model.fit(X_scaled, y)
    return _norm(pd.Series(np.abs(model.coef_), index=features))


def rf_importance(X_scaled, y, features):
    """RandomForest 特征重要性，返回归一化得分 Series。"""
    model = RandomForestRegressor(
        n_estimators=30, max_depth=6,
        max_features='sqrt', min_samples_leaf=50,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    model.fit(X_scaled, y)
    return _norm(pd.Series(model.feature_importances_, index=features))


CACHE_DIR = FIG_DIR.parent / "cache"
_SELECTED_CACHE = CACHE_DIR / "selected_score.csv"
_FULL_CACHE = CACHE_DIR / "full_score.csv"


def compute_ensemble_importance(X, y, top_n=30, use_cache=True):
    """分别用 Lasso / ElasticNet / RF 各取 top_n 特征，取交集后按平均得分排序。

    首次计算后结果缓存到 outputs/cache/，后续调用直接加载（秒级）。
    传入 use_cache=False 可强制重新计算。

    Returns:
        selected_score: pd.Series, 投票通过特征的归一化平均得分 (降序)
        full_score: pd.Series, 全量特征的归一化平均得分 (降序)
    """
    if use_cache and _SELECTED_CACHE.exists() and _FULL_CACHE.exists():
        print("  Loading cached importance scores …")
        selected_score = pd.read_csv(_SELECTED_CACHE, index_col=0).squeeze()
        full_score = pd.read_csv(_FULL_CACHE, index_col=0).squeeze()
        print(f"  Loaded: {len(selected_score)} selected, {len(full_score)} total features")
        return selected_score, full_score

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    features = X.columns

    print("  [1/3] Lasso …")
    lasso_imp = lasso_importance(X_scaled, y, features)
    print("  [2/3] ElasticNet …")
    enet_imp = elasticnet_importance(X_scaled, y, features)
    print("  [3/3] RandomForest …")
    rf_imp = rf_importance(X_scaled, y, features)

    full_score = ((lasso_imp + enet_imp + rf_imp) / 3).sort_values(ascending=False)

    top_lasso = set(lasso_imp.nlargest(top_n).index)
    top_enet = set(enet_imp.nlargest(top_n).index)
    top_rf = set(rf_imp.nlargest(top_n).index)

    from collections import Counter
    votes = Counter(list(top_lasso) + list(top_enet) + list(top_rf))
    common = {f for f, cnt in votes.items() if cnt >= 2}
    print(f"  Top-{top_n} 投票 ≥2/3: {len(common)} 个特征"
          f" (3/3={sum(1 for c in votes.values() if c==3)},"
          f" 2/3={sum(1 for c in votes.values() if c==2)})")

    common_list = sorted(common)
    selected_score = (lasso_imp[common_list] + enet_imp[common_list] + rf_imp[common_list]) / 3
    selected_score = selected_score.sort_values(ascending=False)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    selected_score.to_csv(_SELECTED_CACHE)
    full_score.to_csv(_FULL_CACHE)
    print(f"  Cached scores to {CACHE_DIR}/")

    return selected_score, full_score


def select_features(df, target_col, top_n=None):
    """端到端特征选择：构建矩阵 → 计算重要性 → 返回筛选后的 X, y。

    Args:
        df: 清洗后的 DataFrame
        target_col: 目标列名
        top_n: 保留前 N 个特征；None 则保留全部

    Returns:
        X, y, selected_features (list)
    """
    from src.preprocessing import build_feature_matrix

    X, y, all_features = build_feature_matrix(df, target_col)

    if top_n is None or top_n >= len(all_features):
        return X, y, all_features

    print(f"  Computing ensemble feature importance on {len(all_features)} features...")
    scores, _ = compute_ensemble_importance(X, y)
    selected = scores.head(top_n).index.tolist()
    print(f"  Selected top {top_n} features (best: {selected[0]})")
    return X[selected], y, selected
