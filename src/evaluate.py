import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from config import CV_FOLDS, RANDOM_STATE


def cross_validate_all(models, X, y, cv=None):
    """对多个模型执行交叉验证，返回 {name: {rmse, r2}} 字典。"""
    cv = cv or CV_FOLDS
    results = {}
    for name, model in models.items():
        mse_scores = cross_val_score(
            model, X, y, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1,
        )
        r2_scores = cross_val_score(
            model, X, y, cv=cv,
            scoring='r2', n_jobs=-1,
        )
        results[name] = {
            'rmse': np.sqrt(-mse_scores.mean()),
            'rmse_std': np.sqrt(-mse_scores).std(),
            'r2': r2_scores.mean(),
            'r2_std': r2_scores.std(),
        }
    return results


def results_to_dataframe(results):
    """将 cross_validate_all 的结果转为可展示的 DataFrame。"""
    rows = []
    for name, m in results.items():
        rows.append({
            'Model': name,
            'RMSE': f"{m['rmse']:.3f} ± {m['rmse_std']:.3f}",
            'R²':   f"{m['r2']:.3f} ± {m['r2_std']:.3f}",
        })
    return pd.DataFrame(rows).set_index('Model')


def feature_count_curve(X, y, ensemble_scores, counts=None, cache_path=None):
    """不同特征数量下的 CV RMSE，用于确定最优特征数。

    使用 Ridge 回归（快速）来展示趋势。支持缓存。

    Returns:
        feature_counts: list[int]
        cv_rmses: list[float]
    """
    from pathlib import Path
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            import pandas as _pd
            cached = _pd.read_csv(cache_path)
            print(f"    Loaded cached learning curve ({len(cached)} points)")
            return cached['n'].tolist(), cached['rmse'].tolist()

    sorted_features = ensemble_scores.sort_values(ascending=False).index.tolist()
    if counts is None:
        counts = [5, 10, 15, 20, 30, 50, 80, len(sorted_features)]
    counts = [c for c in counts if c <= len(sorted_features)]

    cv_rmses = []
    for n in counts:
        top_n = sorted_features[:n]
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0)),
        ])
        cv = cross_val_score(
            pipe, X[top_n], y, cv=3,
            scoring='neg_mean_squared_error', n_jobs=-1,
        )
        rmse = np.sqrt(-cv.mean())
        cv_rmses.append(rmse)
        print(f"    n_features={n:3d}  RMSE={rmse:.3f}")

    if cache_path is not None:
        import pandas as _pd
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _pd.DataFrame({'n': counts, 'rmse': cv_rmses}).to_csv(cache_path, index=False)
        print(f"    Cached to {cache_path}")

    return counts, cv_rmses
