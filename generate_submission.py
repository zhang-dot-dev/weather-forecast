"""
生成 Kaggle 提交文件 data/final/submission.csv

流程：
1. 在 100% 训练数据上 fit GroupPCATransformer（特征变换）
2. 用同一个 transformer transform 测试数据
3. 用 100% 训练数据重新训练 catboost（沿用已有超参数）
4. 预测测试集，输出 data/final/submission.csv
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
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

MODEL_NAME = "catboost"
RUN_ID = "20260404_021054"
run_dir = MODEL_DIR / RUN_ID

# ── 1. 读取特征列表 ──────────────────────────────────────────────────────
with open(run_dir / "selected_features.json") as f:
    selected_features = json.load(f)
print(f"步骤 1：读取 selected_features：{len(selected_features)} 个特征")

# ── 2. 在 100% 训练数据上 fit GroupPCATransformer ────────────────────────
print("\n步骤 2：在全量训练数据上 fit GroupPCATransformer")
df_train = load_and_clean(DATA_RAW)
df_train = df_train.sort_values("startdate").reset_index(drop=True)
X_raw_train, y_train, _ = build_feature_matrix_for_pca(df_train, TARGET)

gpca = GroupPCATransformer(random_state=RANDOM_STATE, var_threshold=0.95)
gpca.fit(X_raw_train)
X_train_full = gpca.transform(X_raw_train)
print(f"  训练集维度：{X_train_full.shape}（100% 数据）")

# ── 3. 变换测试数据 ──────────────────────────────────────────────────────
print("\n步骤 3：加载测试数据并做相同的特征变换")
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

# ── 4. 用 100% 训练数据重新训练 catboost ─────────────────────────────────
print(f"\n步骤 4：用全量训练数据训练 {MODEL_NAME}...")
registry = discover_models()
pipe = registry[MODEL_NAME].build_pipeline()

X_train_selected = X_train_full[selected_features]
pipe.fit(X_train_selected, y_train)
print(f"  训练完成（{len(X_train_selected)} 行 × {len(selected_features)} 特征）")

catboost_full_path = OUTPUT_DIR / "catboost_full.pkl"
joblib.dump(pipe, catboost_full_path)
print(f"  已保存 100% 模型 → {catboost_full_path}")

# ── 5. 预测并保存 ────────────────────────────────────────────────────────
final_pred = pipe.predict(X_test_selected)
print(f"\n步骤 5：保存 → {SUBMISSION_PATH}")
print(f"  预测范围：[{final_pred.min():.2f}, {final_pred.max():.2f}]  均值={final_pred.mean():.2f}")

submission = pd.DataFrame({
    "index": df_test["index"].astype(int).values,
    "contest-tmp2m-14d__tmp2m": final_pred,
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"  共 {len(submission)} 行")
print(f"\n预览：\n{submission.head()}")
print("\n完成！")
