# 模型开发指南

## 项目结构

```
src/models/
├── base.py                # 基类（请勿修改）
├── __init__.py            # 自动发现机制（请勿修改）
├── lasso.py               # Lasso 回归
├── elasticnet.py          # ElasticNet 回归
├── random_forest.py       # Random Forest
├── catboost_model.py      # CatBoost
├── xgboost_model.py       # XGBoost
└── lightgbm_model.py      # LightGBM
```

## 环境配置

```bash
# 1. 克隆/拉取最新代码
git pull

# 2. 创建虚拟环境（首次）
python -m venv .venv

# 3. 激活虚拟环境
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 4. 安装依赖
pip install -r requirements.txt
```

## 数据准备

### 运行预处理 notebook

首次拉取代码后，需要先执行 `notebooks/02_preprocessing.ipynb`（选择 `.venv` kernel，Run All）。

该 notebook 会依次完成：
1. 加载原始数据（`data/raw/train_data.csv`，251 列）
2. 数据清洗 & 缺失值处理
3. **特征选择**：使用 Lasso + ElasticNet + RandomForest 三模型投票（≥2/3），从 242 个特征中筛选出 **25 个最重要的特征**（具体逻辑见 `notebooks/01_eda.ipynb` 中的特征选择部分）
4. 80/20 划分训练集和测试集

执行完成后，`data/processed/` 目录下会生成三个文件：

| 文件 | 说明 |
|------|------|
| `train_processed.csv` | 全量数据（25 特征 + 目标变量，375,734 行） |
| `train.csv` | **训练集**（80%，约 300,588 行） |
| `test.csv` | **测试集**（20%，约 75,148 行） |

### 模型训练使用哪些数据

- 训练时使用 **`data/processed/train.csv`**
- 评估时使用 **`data/processed/test.csv`**
- 所有人使用相同的训练集和测试集，保证模型对比公平

> 注意：如果 `data/processed/` 下已有这三个文件（别人已经生成过），可以跳过此步直接开始开发。

---

## 开发流程

### Step 0：创建自己的分支

每人在自己的分支上开发，避免直接修改 main 分支：

```bash
# 从 main 创建并切换到自己的分支（用你的模型名命名）
git checkout main
git pull
git checkout -b model/你的模型名

# 例如：
git checkout -b model/xgboost
git checkout -b model/catboost
git checkout -b model/lightgbm
```

开发完成后提交 Pull Request 合并到 main。

### Step 1：找到你负责的模型文件

每人只需编辑 `src/models/` 下自己负责的 `.py` 文件，例如负责 XGBoost 的同学编辑 `xgboost_model.py`。

### Step 2：修改模型参数

每个文件里有两个方法需要关注：

| 方法 | 作用 | 是否必须 |
|------|------|----------|
| `build_pipeline()` | 定义你的模型 Pipeline | 是 |
| `param_grid()` | 定义超参搜索空间 | 可选 |

示例（以 XGBoost 为例）：

```python
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    name = "xgboost"

    def build_pipeline(self):
        return Pipeline([
            ('scaler', self.default_scaler()),
            ('model', XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=42, verbosity=0,
            )),
        ])

    def param_grid(self):
        return {
            'model__n_estimators': [300, 500, 800],
            'model__max_depth': [4, 6, 8],
            'model__learning_rate': [0.03, 0.1, 0.2],
        }
```

### Step 3：运行 notebook 验证

打开 `notebooks/03_modeling.ipynb`，选择 `.venv` 虚拟环境作为 kernel，然后 Run All。

notebook 会自动发现所有模型并输出交叉验证对比表，无需手动注册。

### Step 4：提交代码并发起 PR

```bash
git add src/models/你的模型文件.py
git commit -m "update: 调整 xxx 模型参数"
git push -u origin model/你的模型名
```

然后在 GitHub 上发起 Pull Request，合并到 main 分支。

## 注意事项

1. **Pipeline 最后一步必须命名为 `'model'`**，超参键名使用 `model__` 前缀（如 `model__max_depth`）
2. **只修改自己的模型文件**，不要改动 `base.py`、`__init__.py` 和 notebook
3. 每人只提交自己的文件，不会产生 git 冲突
4. 如果需要添加新的 pip 依赖，请同步更新 `requirements.txt`
