import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
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


def _col_tmp34w(c: str) -> bool:
    return c.startswith("nmme-tmp2m-34w__") or c.startswith("nmme0-tmp2m-34w__")


def _col_tmp56w(c: str) -> bool:
    return c.startswith("nmme-tmp2m-56w__")


def _col_precip(c: str) -> bool:
    return (
        c.startswith("nmme-prate-34w__")
        or c.startswith("nmme0-prate-34w__")
        or c.startswith("nmme-prate-56w__")
        or c.startswith("nmme0-prate-56w__")
    )


def _col_wind_uv_merged(c: str) -> bool:
    return (
        c.startswith("wind-uwnd-250-2010-")
        or c.startswith("wind-vwnd-250-2010-")
        or c.startswith("wind-uwnd-925-2010-")
        or c.startswith("wind-vwnd-925-2010-")
    )


def _col_tele(c: str) -> bool:
    return c.startswith("mei__") or c.startswith("mjo1d__")


def _group_definitions():
    """按气象学分组：(名称, 列筛选函数, 固定主成分数回退值)。先匹配的组优先占用列。"""
    return [
        ("nmme_tmp34w", _col_tmp34w, 2),
        ("nmme_tmp56w", _col_tmp56w, 2),
        ("nmme_prate", _col_precip, 3),
        ("hgt10", lambda c: c.startswith("wind-hgt-10-2010-"), 1),
        ("hgt100", lambda c: c.startswith("wind-hgt-100-2010-"), 1),
        ("hgt500", lambda c: c.startswith("wind-hgt-500-2010-"), 1),
        ("hgt850", lambda c: c.startswith("wind-hgt-850-2010-"), 1),
        ("sst", lambda c: c.startswith("sst-2010-"), 2),
        ("icec", lambda c: c.startswith("icec-2010-"), 2),
        ("wind_uv", _col_wind_uv_merged, 3),
        ("tele", _col_tele, 2),
    ]


def build_feature_matrix_for_pca(df, target_col):
    """构建用于分组 PCA 的数值矩阵。

    相对 ``build_feature_matrix``：保留 year/month/day/week/season 等时间数值特征；
    仅排除 index、startdate 与目标列；将 ``climateregions``、``season`` 转为数值编码。
    """
    df = df.copy()
    if "climateregions__climateregion" in df.columns:
        col = df["climateregions__climateregion"]
        if col.dtype == object or str(col.dtype) == "category":
            df["climateregions__climateregion"] = pd.Categorical(col.astype(str)).codes
    if "season" in df.columns:
        scol = df["season"]
        if scol.dtype == object or str(scol.dtype) == "category":
            df["season"] = pd.Categorical(scol.astype(str)).codes

    exclude = {"index", "startdate", target_col}
    num_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in num_cols if c not in exclude]
    subset = feature_cols + [target_col]
    df_clean = df[subset].dropna()
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    return X, y, feature_cols


class GroupPCATransformer(BaseEstimator, TransformerMixin):
    """对 NMME / 位势高度 / SST-海冰 / 风场等子组分别 StandardScaler + PCA，其余列原样保留。

    Parameters
    ----------
    var_threshold : float or None
        若指定 (如 0.95)，每组按"保留 ≥ var_threshold 累积方差"动态决定主成分数；
        若为 None，使用 ``_group_definitions()`` 中的固定值。
    """

    def __init__(self, random_state=None, var_threshold=None):
        self.random_state = random_state
        self.var_threshold = var_threshold
        self._group_defs = _group_definitions()
        self._fits = []
        self._pca_input_cols = frozenset()

    def _n_comp(self, n_features: int, n_samples: int, requested: int) -> int:
        max_k = min(n_features, max(1, n_samples - 1))
        return max(1, min(requested, max_k))

    def _n_comp_by_variance(self, Z, n_features):
        """根据累积解释方差自动确定主成分数。"""
        pca_probe = PCA(random_state=self.random_state)
        pca_probe.fit(Z)
        cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
        n = int(np.searchsorted(cumvar, self.var_threshold) + 1)
        return max(1, min(n, n_features))

    def fit(self, X, y=None):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        assigned = set()
        self._fits = []
        all_cols = list(X.columns)
        eps = 1e-15

        for name, pred, n_req in self._group_defs:
            cols = [c for c in all_cols if c not in assigned and pred(c)]
            if not cols:
                continue
            variances = X[cols].var(axis=0, ddof=0)
            cols_active = [c for c in cols if variances.get(c, 0.0) > eps]
            if not cols_active:
                continue
            assigned.update(cols)

            Z = X[cols_active].astype(np.float64).values

            if self.var_threshold is not None:
                n_comp = self._n_comp_by_variance(Z, len(cols_active))
            else:
                n_comp = self._n_comp(len(cols_active), len(X), n_req)

            pca = PCA(n_components=n_comp, random_state=self.random_state)
            pca.fit(Z)
            self._fits.append(
                {
                    "name": name,
                    "cols": cols_active,
                    "scaler": None,
                    "pca": pca,
                    "n_out": n_comp,
                    "explained_var": float(pca.explained_variance_ratio_.sum()),
                }
            )

        self._pca_input_cols = frozenset(assigned)
        self._other_cols_ = [c for c in all_cols if c not in self._pca_input_cols]
        return self

    def transform(self, X):
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        parts = [X[self._other_cols_]]

        for block in self._fits:
            cols = block["cols"]
            Z = X[cols].astype(np.float64).values
            P = block["pca"].transform(Z)
            colnames = [f"pca_{block['name']}_{j + 1}" for j in range(P.shape[1])]
            parts.append(pd.DataFrame(P, columns=colnames, index=X.index))

        out = pd.concat(parts, axis=1)
        return out

    def get_output_columns(self):
        names = list(self._other_cols_)
        for block in self._fits:
            names += [f"pca_{block['name']}_{j + 1}" for j in range(block["n_out"])]
        return names


def select_features(df, target_col, top_n=None, use_group_pca=True):
    """特征工程入口。

    默认 ``use_group_pca=True``：分组 PCA + 保留未参与 PCA 的原始列（含 lat/lon/海拔/时间编码等）。

    原「集成投票」特征选择已停用，若需恢复请设 ``use_group_pca=False`` 并取消下方注释块内的调用。
    """
    if use_group_pca:
        X, y, _ = build_feature_matrix_for_pca(df, target_col)
        transformer = GroupPCATransformer(random_state=RANDOM_STATE)
        transformer.fit(X)
        X_out = transformer.transform(X)
        names = list(X_out.columns)
        print(f"  Group PCA: {len(transformer._fits)} blocks, "
              f"{len(transformer._pca_input_cols)} cols → PCs; "
              f"{len(transformer._other_cols_)} other cols kept. Output dim={len(names)}")
        return X_out, y, names

    from src.preprocessing import build_feature_matrix

    X, y, all_features = build_feature_matrix(df, target_col)

    if top_n is None or top_n >= len(all_features):
        return X, y, all_features

    # ---------- 原集成投票特征选择（Lasso+ENet+RF top_n 投票）；默认已关闭 ----------
    # print(f"  Computing ensemble feature importance on {len(all_features)} features...")
    # scores, _ = compute_ensemble_importance(X, y)
    # selected = scores.head(top_n).index.tolist()
    # print(f"  Selected top {top_n} features (best: {selected[0]})")
    # return X[selected], y, selected

    print("  [select_features] use_group_pca=False：集成选择代码已注释，改为保留全部 build_feature_matrix 特征。")
    print("  若需恢复投票选择，请取消 feature_engineering.select_features 内注释并调用 compute_ensemble_importance。")
    return X, y, all_features


# ---------------------------------------------------------------------------
# 补充特征工程（仿 supplement.ipynb，所有统计仅在 train 上 fit）
# ---------------------------------------------------------------------------

def add_supplementary_features(X_train, X_test, y_train):
    """在 PCA 降维结果基础上追加领域特征，所有统计量仅从训练集计算。

    新增特征：
      1. 周期性时间编码 (sin/cos: month, week, dayofyear)
      2. 气候区 × 月份 目标统计 (mean / std / median)
      3. 经纬度网格 × 月份 目标统计
      4. 滚动窗口目标均值 (14d / 30d，shift(1) 防泄露)
      5. 经纬度交叉特征

    Returns
    -------
    X_train, X_test : pd.DataFrame（新列已拼接，索引不变）
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    tr_idx, te_idx = X_train.index.copy(), X_test.index.copy()

    # ── 1. 周期性时间编码 ──────────────────────────────────────
    if "month" in X_train.columns:
        for df in (X_train, X_test):
            df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
            df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

    if "week" in X_train.columns:
        for df in (X_train, X_test):
            df["sin_week"] = np.sin(2 * np.pi * df["week"] / 52)
            df["cos_week"] = np.cos(2 * np.pi * df["week"] / 52)

    if all(c in X_train.columns for c in ("year", "month", "day")):
        for df in (X_train, X_test):
            doy = pd.to_datetime(
                df[["year", "month", "day"]].rename(
                    columns={"year": "year", "month": "month", "day": "day"}
                )
            ).dt.dayofyear
            df["dayofyear"] = doy
            df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
            df["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    # ── 2. 气候区 × 月份 目标统计 ──────────────────────────────
    cr_col = "climateregions__climateregion"
    if cr_col in X_train.columns and "month" in X_train.columns:
        _tmp = pd.DataFrame({
            "cr": X_train[cr_col].values,
            "month": X_train["month"].values,
            "target": y_train.values,
        })
        grp = _tmp.groupby(["cr", "month"])["target"].agg(
            climate_month_mean="mean",
            climate_month_std="std",
            climate_month_median="median",
        )
        for stat in ("climate_month_mean", "climate_month_std", "climate_month_median"):
            mi_tr = pd.MultiIndex.from_arrays(
                [X_train[cr_col].values, X_train["month"].values]
            )
            mi_te = pd.MultiIndex.from_arrays(
                [X_test[cr_col].values, X_test["month"].values]
            )
            X_train[stat] = grp[stat].reindex(mi_tr).values
            X_test[stat] = grp[stat].reindex(mi_te).values
        print("  + 气候区×月份目标统计 (mean/std/median)")

    # ── 3. 经纬度网格 × 月份 目标统计 ──────────────────────────
    if all(c in X_train.columns for c in ("lat", "lon", "month")):
        for df in (X_train, X_test):
            df["lat_grid"] = (df["lat"] // 2 * 2).astype(int)
            df["lon_grid"] = (df["lon"] // 2 * 2).astype(int)

        _tmp2 = pd.DataFrame({
            "lg": X_train["lat_grid"].values,
            "log": X_train["lon_grid"].values,
            "month": X_train["month"].values,
            "target": y_train.values,
        })
        grp2 = _tmp2.groupby(["lg", "log", "month"])["target"].agg(
            grid_month_mean="mean",
            grid_month_std="std",
            grid_month_median="median",
        )
        for stat in ("grid_month_mean", "grid_month_std", "grid_month_median"):
            mi_tr = pd.MultiIndex.from_arrays([
                X_train["lat_grid"].values,
                X_train["lon_grid"].values,
                X_train["month"].values,
            ])
            mi_te = pd.MultiIndex.from_arrays([
                X_test["lat_grid"].values,
                X_test["lon_grid"].values,
                X_test["month"].values,
            ])
            X_train[stat] = grp2[stat].reindex(mi_tr).values
            X_test[stat] = grp2[stat].reindex(mi_te).values
        print("  + 经纬度网格×月份目标统计 (mean/std/median)")

    # ── 4. 滚动窗口目标均值 (按 lat+lon 分组) ─────────────────
    if all(c in X_train.columns for c in ("lat", "lon")):
        _roll = pd.DataFrame({
            "lat": X_train["lat"].values,
            "lon": X_train["lon"].values,
            "target": y_train.values,
        })
        for window in (14, 30):
            col_name = f"rolling_{window}d_mean"
            _roll[col_name] = (
                _roll.groupby(["lat", "lon"])["target"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            X_train[col_name] = _roll[col_name].values

        last_rolling = (
            _roll.groupby(["lat", "lon"])[["rolling_14d_mean", "rolling_30d_mean"]]
            .last().reset_index()
        )
        X_test = X_test.reset_index(drop=True).merge(
            last_rolling, on=["lat", "lon"], how="left"
        )
        X_test.index = te_idx
        print("  + 滚动窗口目标均值 (14d / 30d)")

    # ── 5. 经纬度交叉特征 ─────────────────────────────────────
    if all(c in X_train.columns for c in ("lat", "lon")):
        for df in (X_train, X_test):
            df["lat_lon_interact"] = df["lat"] * df["lon"]
            df["dist_equator"] = df["lat"].abs()
        print("  + 经纬度交叉特征")

    # ── 6. 填充新特征中的 NaN ─────────────────────────────────
    for col in X_train.columns:
        if X_train[col].isnull().any():
            fill = X_train[col].median()
            X_train[col] = X_train[col].fillna(fill)
            X_test[col] = X_test[col].fillna(fill)

    X_train.index = tr_idx
    X_test.index = te_idx
    return X_train, X_test
