# Weather Temperature Forecast

基于多源气象特征预测未来 14 天 2 米气温（`contest-tmp2m-14d__tmp2m`）。

## 项目目标

利用 NMME 数值预报集成、历史气候基准、海温（SST）、海冰（ICEC）、气候指数（ENSO/MJO）等 240+ 维特征，训练回归模型预测美国本土站点未来两周的平均气温。

## 数据概况

| 项目 | 说明 |
|------|------|
| 训练集 | `data/raw/train_data.csv`（375,734 行 × 246 列） |
| 测试集 | `data/raw/test_data.csv` |
| 时间范围 | 2014-09-01 ~ 2016-08-31 |
| 目标变量 | `contest-tmp2m-14d__tmp2m`（未来 14 天 2 米气温，单位 °C） |

## 项目结构

```
weather-forecast/
├── config.py                       # 全局配置（路径、常量、超参数）
├── main.py                         # 命令行一键运行入口
├── requirements.txt                # Python 依赖
│
├── src/                            # 共享函数库
│   ├── preprocessing.py            #   数据加载、清洗、缺失值填充
│   ├── feature_engineering.py      #   特征重要性计算、特征选择
│   ├── models.py                   #   模型定义（Lasso / ElasticNet / RF）
│   └── evaluate.py                 #   交叉验证、评估指标、学习曲线
│
├── notebooks/                      # 分析 & 实验（每人负责一个）
│   ├── 01_eda.ipynb                #   探索性数据分析
│   ├── 02_preprocessing.ipynb      #   数据预处理 & 特征工程
│   ├── 03_modeling.ipynb           #   模型训练 & 调参
│   └── 04_evaluation.ipynb         #   模型评估 & 预测
│
├── data/
│   ├── raw/                        # 原始数据（不修改）
│   └── processed/                  # 预处理后的特征矩阵
│
└── outputs/
    ├── models/                     # 保存的模型文件 (.pkl)
    ├── figures/                    # EDA & 评估可视化图片
    └── predictions/                # 测试集预测结果
```

## 环境配置

**Python >= 3.11**

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

主要依赖：pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, joblib, pyarrow

## 快速运行

```bash
# 方式一：一键运行完整 pipeline
python main.py

# 方式二：按 notebook 逐步执行
# 打开 notebooks/ 下的 01 → 02 → 03 → 04 依次运行
```

## 当前进度

- [x] 项目框架搭建（config / src / notebooks）
- [x] EDA 探索性数据分析（`01_eda.ipynb`）
- [ ] 数据预处理 & 特征工程（`02_preprocessing.ipynb`）
- [ ] 模型训练 & 超参调优（`03_modeling.ipynb`）
- [ ] 模型评估 & 测试集预测（`04_evaluation.ipynb`）
