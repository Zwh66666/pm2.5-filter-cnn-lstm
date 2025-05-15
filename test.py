import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 设置中文字体为仿宋
rcParams['font.family'] = 'FangSong'
rcParams['font.size'] = 12

# 设置随机种子以确保结果可重复
np.random.seed(42)
tf.random.set_seed(42)

# 加载自定义层，以便正确加载模型
class LearnableFourierFilter(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LearnableFourierFilter, self).__init__(**kwargs)

    def build(self, input_shape):
        # 创建可学习的频域滤波器参数
        # input_shape: (batch_size, sequence_length, features)
        self.filter_weights = self.add_weight(
            name='filter_weights',
            shape=(input_shape[1], input_shape[2]),
            initializer='ones',  # 初始化为1，相当于不过滤
            trainable=True,  # 设置为可训练
        )

        # 添加幅度调节参数
        self.amplitude_weights = self.add_weight(
            name='amplitude_weights',
            shape=(input_shape[2],),
            initializer='ones',
            trainable=True,
        )

        # 添加相位调节参数
        self.phase_weights = self.add_weight(
            name='phase_weights',
            shape=(input_shape[2],),
            initializer='zeros',
            trainable=True,
        )

        self.built = True

    def call(self, inputs):
        # 对每个特征进行傅里叶变换
        # 将实数输入转换为复数
        complex_inputs = tf.cast(inputs, tf.complex64)

        # 对序列维度进行FFT
        fft = tf.signal.fft(complex_inputs)

        # 应用可学习的滤波器（需要将滤波器权重转换为复数）
        complex_weights = tf.cast(self.filter_weights, tf.complex64)

        # 创建复数形式的幅度和相位调节
        amplitude = tf.cast(self.amplitude_weights, tf.float32)
        phase = tf.cast(self.phase_weights, tf.float32)
        complex_adjust = tf.complex(
            amplitude * tf.cos(phase),
            amplitude * tf.sin(phase)
        )

        # 应用滤波器权重和幅度/相位调节
        filtered_fft = fft * tf.expand_dims(complex_weights, 0)
        filtered_fft = filtered_fft * tf.reshape(complex_adjust, [1, 1, -1])

        # 进行逆傅里叶变换
        ifft = tf.signal.ifft(filtered_fft)

        # 取实部作为输出
        output = tf.math.real(ifft)

        return output


# 创建保存结果的目录
if not os.path.exists('../results/test'):
    os.makedirs('../results/test')

# 加载测试数据
# 注意：请确保测试数据的格式与训练数据相同
test_data = pd.read_csv('../data/test.csv')  # 测试数据路径

# 将datetime转换为日期时间格式
test_data['datetime'] = pd.to_datetime(test_data['datetime'], unit='s')
test_data.set_index('datetime', inplace=True)

# 选择特征 - 与训练时相同
features =['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', '气温2m(℃)', '相对湿度(%)','降水量(mm)','经向风速(V,m/s)','纬向风速(U,m/s)','总太阳辐射度(down,J/m2)'] #['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', '气温2m(℃)', '地面气压(hPa)', '蒸发量(mm)']
test_data_selected = test_data[features]

# 加载训练数据以获取相同的缩放参数
train_data = pd.read_csv('../data/PM25_process.csv')
train_data['datetime'] = pd.to_datetime(train_data['datetime'], unit='s')
train_data.set_index('datetime', inplace=True)
train_data_selected = train_data[features]

# 使用训练数据的缩放参数进行标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_data_selected)
scaled_test_data = scaler.transform(test_data_selected)
scaled_test_data = pd.DataFrame(scaled_test_data, columns=test_data_selected.columns, index=test_data_selected.index)

# 创建时间序列数据集
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    return np.array(X)

# 设置参数
sequence_length = 24  # 与训练时相同
target_col_idx = features.index('PM2.5')  # PM2.5的索引

# 准备测试数据
X_test_seq = create_sequences(scaled_test_data.values, sequence_length)
y_test_actual = scaled_test_data.values[sequence_length:, target_col_idx]

print(f"测试数据形状: {X_test_seq.shape}")

# 加载训练好的模型
custom_objects = {'LearnableFourierFilter': LearnableFourierFilter}
model = load_model('../results/cnn_lstm_pm25_model.h5', custom_objects=custom_objects)

# 在测试集上进行预测
y_pred = model.predict(X_test_seq)

# 反归一化预测结果和真实值以获得实际的PM2.5值
y_test_actual_denorm = y_test_actual * (scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]) + scaler.data_min_[target_col_idx]
y_pred_actual = y_pred.flatten() * (scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]) + scaler.data_min_[target_col_idx]

# 计算评估指标
mse = mean_squared_error(y_test_actual_denorm, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual_denorm, y_pred_actual)
r2 = r2_score(y_test_actual_denorm, y_pred_actual)

print(f"测试集评估结果:")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 绘制预测结果与实际值的对比
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual_denorm, label='实际值')
plt.plot(y_pred_actual, label='预测值')
plt.title('PM2.5测试集预测结果对比')
plt.xlabel('时间步')
plt.ylabel('PM2.5浓度')
plt.legend()
plt.tight_layout()
plt.savefig('../results/test/test_prediction_comparison.png')

# 绘制散点图比较预测值和实际值
plt.figure(figsize=(8, 8))
plt.scatter(y_test_actual_denorm, y_pred_actual, alpha=0.5)
plt.plot([min(y_test_actual_denorm), max(y_test_actual_denorm)], 
         [min(y_test_actual_denorm), max(y_test_actual_denorm)], 
         'r--')
plt.title('PM2.5预测值与实际值散点图')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.grid(True)
plt.savefig('../results/test/test_scatter_plot.png')

# 计算预测误差
errors = y_test_actual_denorm - y_pred_actual

# 绘制误差分布直方图
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, alpha=0.75)
plt.title('预测误差分布')
plt.xlabel('预测误差')
plt.ylabel('频率')
plt.grid(True)
plt.savefig('../results/test/test_error_distribution.png')

# 保存测试结果到CSV文件
test_dates = test_data_selected.index[sequence_length:]
results_df = pd.DataFrame({
    'datetime': test_dates,
    'actual_pm25': y_test_actual_denorm,
    'predicted_pm25': y_pred_actual,
    'error': errors
})
results_df.to_csv('../results/test/test_results.csv', index=False)

print("测试完成，结果已保存到 '../results/test/' 目录")