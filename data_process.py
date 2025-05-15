import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('../data/PM25数据.csv')

# 1. 将时间列转换为时间戳（假设时间列名为'time'）
# 修改时间解析格式
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y/%m/%d %H:%M:%S').astype('int64') // 10**9

# 2. 处理缺失值 - 线性插值
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].interpolate(method='linear').round(2)

# 3. 处理异常值 - 使用3σ原则
for col in numeric_cols:
    if col != 'timestamp':  # 排除时间戳列
        mean = df[col].mean()
        std = df[col].std()
        df[col] = np.where(
            (df[col] < mean - 3*std) | (df[col] > mean + 3*std),
            np.nan,
            df[col]
        )
        # 对替换为NaN的异常值再次进行线性插值
        df[col] = df[col].interpolate(method='linear').round(2)

# 按时间顺序划分测试集（最后10%）和训练集（前90%）
split_idx = int(len(df) * 0.9)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
train.to_csv('../data/PM25_process.csv', index=False)
test.to_csv('../data/test.csv', index=False)

