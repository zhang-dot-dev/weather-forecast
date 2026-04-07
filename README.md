# Weather Temperature Forecast

基于多源气象特征预测未来 14 天 2 米气温（`contest-tmp2m-14d__tmp2m`）。Kaggle 竞赛项目，评估指标为 RMSE。

## 项目目标

利用 NMME 数值预报集成、历史气候基准、海温（SST）、海冰（ICEC）、气候指数（ENSO/MJO）等 240+ 维特征，训练回归模型预测美国本土站点未来两周的平均气温。

## 数据概况

| 项目 | 说明 |
|------|------|
| 训练集 | `data/raw/train_data.csv`（375,734 行 × 246 列） |
| 测试集 | `data/raw/test_data.csv`（31,354 行） |
| 时间范围 | 2014-09-01 ~ 2016-08-31 |
| 目标变量 | `contest-tmp2m-14d__tmp2m`（未来 14 天 2 米气温，单位 °C） |
| 评估指标 | RMSE |

## 项目结构

```
weather-forecast/
├── config.py                          # 全局配置（路径、常量、超参数）
├── requirements.txt                   # Python 依赖
│
├── src/                               # 共享函数库
│   ├── preprocessing.py               #   数据加载、清洗、时间特征、缺失值填充
│   ├── feature_engineering.py         #   特征重要性、分组 PCA（GroupPCATransformer）
│   ├── evaluate.py                    #   交叉验证、评估指标、学习曲线
│   └── models/                        #   模型注册中心与各模型定义
│       ├── base.py                    #     BaseModel 抽象基类
│       ├── catboost_model.py          #     CatBoost
│       ├── lightgbm_model.py          #     LightGBM
│       ├── xgboost_model.py           #     XGBoost
│       ├── elasticnet.py              #     ElasticNet
│       ├── lasso.py                   #     Lasso
│       └── random_forest.py           #     RandomForest
│
├── notebooks/                         # 分析 & 实验（按顺序执行）
│   ├── 01_eda.ipynb                   #   探索性数据分析
│   ├── 02_preprocessing.ipynb         #   数据预处理 & 特征工程
│   ├── 03_modeling.ipynb              #   单模型训练、调参、保存 .pkl
│   └── 04_ensemble.ipynb              #   Catboost + ElasticNet Blending
│
├── generate_submission.py             # 生成 Kaggle 提交文件（catboost 单模型）
├── generate_submission_ensemble.py    # 生成 Kaggle 提交文件（blending 集成）
├── generate_report.py                 # 生成项目报告
│
├── data/
│   ├── raw/                           # 原始数据（不修改）
│   ├── processed/                     # 预处理后的特征矩阵
│   └── final/                         # Kaggle 提交文件
│       ├── submission.csv             #   catboost 单模型预测
│       └── submission_ensemble.csv    #   blending 集成预测
│
├── outputs/
│   ├── models/<run_id>/               # 各次运行保存的模型 .pkl 和 selected_features.json
│   ├── figures/                       # 可视化图片
│   ├── run_history.csv                # 单模型历史运行记录
│   └── ensemble_history.csv           # ensemble 历史运行记录
│
├── report/
│   └── report.tex                     # LaTeX 报告源文件
│
└── tests/
    └── test_preprocessing.py          # 预处理单元测试
```

## 模型 & 方法

### 特征工程

- **时间特征**：从 `startdate` 派生 year/month/day/week/season
- **缺失值填充**：数值列中位数填充
- **分组 PCA**（`GroupPCATransformer`）：按气象学含义将 246 维原始特征分组（NMME 温度/降水、位势高度、SST、海冰、风场、气候指数），每组独立做 PCA（保留 95% 累积方差），最终降维至 92 维

### 训练的模型

| 模型 | 验证集 RMSE（80/20 切分） |
|------|--------------------------|
| **CatBoost** | **1.0959** |
| LightGBM | 1.2001 |
| XGBoost | 1.2466 |
| ElasticNet | 1.5153 |
| Lasso | 1.6224 |
| RandomForest | 1.7799 |

### 集成策略

在 `04_ensemble.ipynb` 中尝试了 Catboost + ElasticNet 加权 Blending：

```
pred = α · catboost + (1-α) · elasticnet
```

在 20% 验证集上搜索最优 α。由于 catboost 单模型已显著优于其他模型，blending 对最终 RMSE 的提升有限。

## 环境配置

**Python >= 3.11**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

主要依赖：pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, xgboost, lightgbm, catboost, joblib

## 快速运行

```bash
# 按 notebook 逐步执行（推荐）
# 打开 notebooks/ 下的 01 → 02 → 03 → 04 依次运行

# 生成 Kaggle 提交文件（catboost 单模型，100% 数据训练）
python generate_submission.py

# 生成 Kaggle 提交文件（catboost + elasticnet blending）
python generate_submission_ensemble.py
```

## 当前进度

- [x] 项目框架搭建（config / src / notebooks）
- [x] EDA 探索性数据分析（`01_eda.ipynb`）
- [x] 数据预处理 & 特征工程（`02_preprocessing.ipynb`）
- [x] 分组 PCA 降维：246 维 → 92 维
- [x] 模型训练 & 评估：6 个模型（`03_modeling.ipynb`）
- [x] Ensemble 集成实验（`04_ensemble.ipynb`）
- [x] Kaggle 提交文件生成
