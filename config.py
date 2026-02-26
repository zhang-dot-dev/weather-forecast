from pathlib import Path

# ─── 路径 ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_RAW   = BASE_DIR / "data/raw/train_data.csv"
DATA_TEST  = BASE_DIR / "data/raw/test_data.csv"
DATA_PROC  = BASE_DIR / "data/processed/train_processed.csv"
MODEL_DIR  = BASE_DIR / "outputs/models"
FIG_DIR    = BASE_DIR / "outputs/figures"
PRED_DIR   = BASE_DIR / "outputs/predictions"

# ─── 目标变量 ─────────────────────────────────────────────────
TARGET = "contest-tmp2m-14d__tmp2m"

# ─── 列分组（供 EDA / 特征工程复用） ──────────────────────────
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Fall']
SEASON_PALETTE = {
    'Winter': '#5b9bd5', 'Spring': '#70ad47',
    'Summer': '#ed7d31', 'Fall':   '#a5a5a5',
}
TIME_DERIVED_COLS = ['year', 'month', 'day', 'week', 'season']
DROP_COLS = ['index', 'startdate'] + TIME_DERIVED_COLS

# ─── 模型超参数 ───────────────────────────────────────────────
LASSO_ALPHA     = 0.1
ENET_ALPHA      = 0.1
ENET_L1_RATIO   = 0.5
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = 10
RANDOM_STATE    = 42
CV_FOLDS        = 5