"""
生成 Kaggle 提交文件 data/final/submission_ensemble.csv

流程：
1. 在 100% 训练数据上 fit GroupPCATransformer（特征变换）
2. 用同一个 transformer transform 测试数据
3. catboost: 加载 generate_submission.py 已保存的 catboost_full.pkl（不重复训练）
   elasticnet: 100% 数据训练（秒级）
4. 从 ensemble_history.csv 读取最优 blending 权重 α
5. α·catboost + (1-α)·elasticnet → 输出 data/final/submission_ensemble.csv

注意：请先运行 generate_submission.py 生成 catboost_full.pkl
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_RAW, DATA_TEST, TARGET, MODEL_DIR, RANDOM_STATE
from src.feature_engineering import GroupPCATransformer, build_feature_matrix_for_pca
from src.preprocessing import load_and_clean
from src.models import discover_models

OUTPUT_DIR = Path(__file__).parent / "data" / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = OUTPUT_DIR / "submission_ensemble.csv"
ENSEMBLE_HISTORY_PATH = MODEL_DIR.parent / "ensemble_history.csv"

RUN_ID = "20260404_021054"
run_dir = MODEL_DIR / RUN_ID

# ── 1. 读取最优 blending 权重 α ─────────────────────────────────────────
print("步骤 1：从 ensemble_history.csv 读取最优 blending 权重")
ens_df = pd.read_csv(ENSEMBLE_HISTORY_PATH)
ens_df = ens_df.dropna(subset=["stacking_members"])
ens_blend = ens_df[ens_df["stacking_blend_method"] == "weighted_average"]
if ens_blend.empty:
    raise ValueError("ensemble_history.csv 中没有 weighted_average 记录，请先运行 04_ensemble.ipynb")
best_row = ens_blend.loc[ens_blend["stacking_rmse"].astype(float).idxmin()]

coefs = json.loads(best_row["stacking_meta_coefs"])
alpha_cat = coefs[0]
alpha_enet = coefs[1]
print(f"  α(catboost)={alpha_cat:.2f}, α(elasticnet)={alpha_enet:.2f}")
print(f"  验证集 blending RMSE={best_row['stacking_rmse']}")

# ── 2. 读取特征列表 ──────────────────────────────────────────────────────
with open(run_dir / "selected_features.json") as f:
    selected_features = json.load(f)
print(f"\n步骤 2：读取 selected_features：{len(selected_features)} 个特征")

# ── 3. 在 100% 训练数据上 fit GroupPCATransformer ────────────────────────
print("\n步骤 3：在全量训练数据上 fit GroupPCATransformer")
df_train = load_and_clean(DATA_RAW)
df_train = df_train.sort_values("startdate").reset_index(drop=True)
X_raw_train, y_train, _ = build_feature_matrix_for_pca(df_train, TARGET)

gpca = GroupPCATransformer(random_state=RANDOM_STATE, var_threshold=0.95)
gpca.fit(X_raw_train)
X_train_full = gpca.transform(X_raw_train)
X_train_selected = X_train_full[selected_features]
print(f"  训练集维度：{X_train_selected.shape}（100% 数据）")

# ── 4. 变换测试数据 ──────────────────────────────────────────────────────
print("\n步骤 4：加载测试数据并做相同的特征变换")
df_test = load_and_clean(DATA_TEST)

df_test_proc = df_test.copy()
if "climateregions__climateregion" in df_test_proc.columns:
    col = df_test_proc["climateregions__climateregion"]
    if col.dtype == object or str(col.dtype) == "category":
        df_test_proc["climateregions__climateregion"] = pd.Categorical(col.astype(str)).codes
if "season" in df_test_proc.columns:
    scol = df_test_proc["season"]
    if scol.dtype == object or str(scol.dtype) == "category":
        df_test_proc["season"] = pd.Categorical(scol.astype(str)).codes

num_cols = df_test_proc.select_dtypes(include=[np.number]).columns
X_raw_test = df_test_proc[[c for c in num_cols if c not in {"index", "startdate", TARGET}]]

for c in X_raw_train.columns:
    if c not in X_raw_test.columns:
        X_raw_test = X_raw_test.copy()
        X_raw_test[c] = 0.0
X_raw_test = X_raw_test[X_raw_train.columns]

X_test_full = gpca.transform(X_raw_test)
X_test_selected = X_test_full[selected_features]
print(f"  测试集维度：{X_test_selected.shape}")

# ── 5. catboost: 加载已有 pkl / elasticnet: 训练（秒级） ─────────────────
print("\n步骤 5：加载 catboost + 训练 elasticnet")

catboost_full_path = OUTPUT_DIR / "catboost_full.pkl"
if not catboost_full_path.exists():
    raise FileNotFoundError(
        f"未找到 {catboost_full_path}\n请先运行 generate_submission.py 生成 catboost_full.pkl"
    )
cat_pipe = joblib.load(catboost_full_path)
cat_pred = cat_pipe.predict(X_test_selected)
print(f"  catboost   (加载 catboost_full.pkl) 预测范围：[{cat_pred.min():.2f}, {cat_pred.max():.2f}]")

registry = discover_models()
enet_pipe = registry["elasticnet"].build_pipeline()
enet_pipe.fit(X_train_selected, y_train)
enet_pred = enet_pipe.predict(X_test_selected)
print(f"  elasticnet (100% 数据训练) 预测范围：[{enet_pred.min():.2f}, {enet_pred.max():.2f}]")

# ── 6. Blending 并保存 ──────────────────────────────────────────────────
final_pred = alpha_cat * cat_pred + alpha_enet * enet_pred
print(f"\n步骤 6：Blending（{alpha_cat:.2f}·catboost + {alpha_enet:.2f}·elasticnet）")
print(f"  预测范围：[{final_pred.min():.2f}, {final_pred.max():.2f}]  均值={final_pred.mean():.2f}")

print(f"\n保存 → {SUBMISSION_PATH}")
submission = pd.DataFrame({
    "index": df_test["index"].astype(int).values,
    "contest-tmp2m-14d__tmp2m": final_pred,
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"  共 {len(submission)} 行")
print(f"\n预览：\n{submission.head()}")
print("\n完成！")
