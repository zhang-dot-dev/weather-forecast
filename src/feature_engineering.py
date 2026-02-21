import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from config import LASSO_ALPHA, ENET_ALPHA, ENET_L1_RATIO, RF_MAX_DEPTH, RANDOM_STATE


def compute_ensemble_importance(X, y):
    """用 Lasso + ElasticNet + RandomForest 计算集成特征重要性。

    Returns:
        ensemble_score: pd.Series, 归一化后三模型平均得分 (0~1)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    features = X.columns

    lasso = Lasso(alpha=LASSO_ALPHA, max_iter=5000)
    lasso.fit(X_scaled, y)
    lasso_imp = pd.Series(np.abs(lasso.coef_), index=features)

    enet = ElasticNet(alpha=ENET_ALPHA, l1_ratio=ENET_L1_RATIO, max_iter=5000)
    enet.fit(X_scaled, y)
    enet_imp = pd.Series(np.abs(enet.coef_), index=features)

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=RF_MAX_DEPTH,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf.fit(X_scaled, y)
    rf_imp = pd.Series(rf.feature_importances_, index=features)

    def _norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-10)

    ensemble_score = (_norm(lasso_imp) + _norm(enet_imp) + _norm(rf_imp)) / 3
    return ensemble_score.sort_values(ascending=False)


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
    scores = compute_ensemble_importance(X, y)
    selected = scores.head(top_n).index.tolist()
    print(f"  Selected top {top_n} features (best: {selected[0]})")
    return X[selected], y, selected
