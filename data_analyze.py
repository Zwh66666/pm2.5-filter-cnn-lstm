import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

# 读取处理后的数据
df = pd.read_csv('../data/PM25_process.csv')

# 将时间戳转换为datetime格式
df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
# 设置全局字体为仿宋
plt.rcParams['font.sans-serif'] = ['FangSong']  # 设置仿宋字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. PM2.5时序图
plt.figure(figsize=(15, 6))
plt.plot(df['datetime'], df['PM2.5'], color='red', linewidth=1)
plt.title('PM2.5 时间序列', fontsize=14)  # 标题改为中文
plt.xlabel('日期', fontsize=12)  # x轴标签改为中文
plt.ylabel('PM2.5 (微克/立方米)', fontsize=12)  # y轴标签改为中文
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('../output/pm25_timeseries.png', dpi=300)
plt.close()

# 2. 热力图分析
corr_cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', '气温2m(℃)', '地面气压(hPa)','相对湿度(%)','蒸发量(mm)','降水量(mm)','地表温度(℃)','经向风速(V,m/s)','纬向风速(U,m/s)','总太阳辐射度(down,J/m2)']
corr_df = df[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5, annot_kws={'size': 10})
plt.title('空气质量参数相关性热力图', fontsize=14)  # 标题改为中文
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('../output/pm25_correlation_heatmap.png', dpi=300)
plt.close()

# 3. PM2.5与PM10的散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PM10', y='PM2.5', data=df, alpha=0.6, color='blue')
plt.title('PM2.5与PM10关系', fontsize=14)  # 标题改为中文
plt.xlabel('PM10 (微克/立方米)', fontsize=12)  # x轴标签改为中文
plt.ylabel('PM2.5 (微克/立方米)', fontsize=12)  # y轴标签改为中文
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('../output/pm25_vs_pm10.png', dpi=300)
plt.close()

# 4. PM2.5与SO2的散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SO2', y='PM2.5', data=df, alpha=0.6, color='green')
plt.title('PM2.5与SO2关系', fontsize=14)  # 标题改为中文
plt.xlabel('SO2 (微克/立方米)', fontsize=12)  # x轴标签改为中文
plt.ylabel('PM2.5 (微克/立方米)', fontsize=12)  # y轴标签改为中文
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('../output/pm25_vs_so2.png', dpi=300)
plt.close()

print("分析完成，图表已保存到output目录")
