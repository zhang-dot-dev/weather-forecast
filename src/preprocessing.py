import pandas as pd
import numpy as np


def load_and_clean(path):
    """加载 CSV 并完成基础清洗：时间拆分 + 缺失值填充。"""
    df = pd.read_csv(path)
    df = add_time_features(df)
    df = fill_missing(df)
    return df


def add_time_features(df):
    """从 startdate 派生 year / month / day / week / season。"""
    dt = pd.to_datetime(df['startdate'], format='%m/%d/%y')
    month = dt.dt.month
    week = dt.dt.isocalendar().week.to_numpy(dtype='int32', na_value=0)
    time_cols = pd.DataFrame({
        'year':   dt.dt.year,
        'month':  month,
        'day':    dt.dt.day,
        'week':   week,
        'season': month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3:  'Spring', 4: 'Spring', 5: 'Spring',
            6:  'Summer', 7: 'Summer', 8: 'Summer',
            9:  'Fall',  10: 'Fall',  11: 'Fall',
        }),
    }, index=df.index)
    df = pd.concat([df.assign(startdate=dt), time_cols], axis=1)
    return df


def fill_missing(df, strategy='median'):
    """数值列缺失值填充。strategy: 'median' | 'mean' | 'zero'。"""
    num_cols = df.select_dtypes(include='number').columns
    if strategy == 'median':
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif strategy == 'mean':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy == 'zero':
        df[num_cols] = df[num_cols].fillna(0)
    return df


def get_column_groups(df):
    """按前缀/关键词将列分组，返回字典。"""
    from config import TARGET
    return {
        'nmme_tmp':    [c for c in df.columns if 'nmme' in c and 'tmp2m' in c],
        'nmme_prate':  [c for c in df.columns if 'nmme' in c and 'prate' in c],
        'contest':     [c for c in df.columns if c.startswith('contest') and c != TARGET],
        'sst':         [c for c in df.columns if c.startswith('sst')],
        'icec':        [c for c in df.columns if c.startswith('icec')],
        'climate_idx': ['mjo1d__phase', 'mjo1d__amplitude',
                        'mei__mei', 'mei__meirank', 'mei__nip'],
    }


def build_feature_matrix(df, target_col, drop_cols=None):
    """从 DataFrame 中分离 X, y，只保留数值列。"""
    from config import DROP_COLS
    exclude = set(DROP_COLS + (drop_cols or []) + [target_col])
    num_cols = df.select_dtypes(include='number').columns
    feature_cols = [c for c in num_cols if c not in exclude]
    df_clean = df[feature_cols + [target_col]].dropna()
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    return X, y, feature_cols