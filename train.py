import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout, BatchNormalization, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.layers import Layer
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体为仿宋
rcParams['font.family'] = 'FangSong'  # 或者使用 'SimFang'
rcParams['font.size'] = 12  # 可以调整字体大小


# 创建可学习的傅里叶滤波器层
class LearnableFourierFilter(Layer):
    def __init__(self, **kwargs):
        super(LearnableFourierFilter, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # 创建可学习的频域滤波器参数
        # input_shape: (batch_size, sequence_length, features)
        self.filter_weights = self.add_weight(
            name='filter_weights',
            shape=(input_shape[1], input_shape[2]),
            initializer='ones',  # 初始化为1，相当于不过滤
            trainable=True,      # 设置为可训练
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


# 设置随机种子以确保结果可重复
np.random.seed(42)
tf.random.set_seed(42)

# 加载数据
data = pd.read_csv('../data/PM25_process.csv')

# 将datetime转换为日期时间格式
data['datetime'] = pd.to_datetime(data['datetime'], unit='s')
data.set_index('datetime', inplace=True)

# 选择特征
features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', '气温2m(℃)', '相对湿度(%)','降水量(mm)','经向风速(V,m/s)','纬向风速(U,m/s)','总太阳辐射度(down,J/m2)']#['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2', '气温2m(℃)', '地面气压(hPa)', '蒸发量(mm)']
data_selected = data[features]

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_selected)
scaled_data = pd.DataFrame(scaled_data, columns=data_selected.columns, index=data_selected.index)


# 创建时间序列数据集
def create_sequences(data, target_col_idx, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_col_idx])
    return np.array(X), np.array(y)


# 设置参数
sequence_length = 24  # 使用24小时的数据预测下一个时刻
target_col_idx = features.index('PM2.5')  # PM2.5的索引
X, y = create_sequences(scaled_data.values, target_col_idx, sequence_length)

# 划分训练集和测试集
train_size = int(len(X) * 0.75)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"训练集形状: {X_train.shape}, {y_train.shape}")
print(f"测试集形状: {X_test.shape}, {y_test.shape}")


# 构建带有傅里叶滤波器的CNN-LSTM模型
def build_cnn_lstm_model(input_shape):
    model = Sequential([
        # 添加可学习的傅里叶滤波器层
        LearnableFourierFilter(input_shape=input_shape),

        # CNN层
        Conv1D(filters=64, kernel_size=3, activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),  # 降低Dropout率

        Conv1D(filters=128, kernel_size=3, activation='relu',
               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),  # 降低Dropout率

        # LSTM层
        LSTM(100, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),
        LSTM(50, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),

        # 全连接层
        Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(1)
    ])

    # 使用带学习率衰减的优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


# 创建并训练模型
model = build_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
model.summary()

# 早停策略
early_stopping = EarlyStopping(monitor='val_loss',patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,  # 增大batch size
    validation_split=0.2,
    callbacks=[
        early_stopping,
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5)
    ],
    verbose=1
)

# 评估模型
y_pred = model.predict(X_test)

# 反归一化预测结果和真实值以获得实际的PM2.5值
y_test_actual = y_test * (scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]) + scaler.data_min_[
    target_col_idx]
y_pred_actual = y_pred * (scaler.data_max_[target_col_idx] - scaler.data_min_[target_col_idx]) + scaler.data_min_[
    target_col_idx]

# 计算评估指标
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

from tensorflow.keras.utils import plot_model

# 创建保存结果的目录
if not os.path.exists('../results'):
    os.makedirs('../results')

# 绘制模型结构图
plot_model(
    model,
    to_file='../results/model_architecture.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir='TB',  # 图形方向: TB (从上到下), LR (从左到右)
    dpi=96,
    expand_nested=True  # 显示嵌套层的细节
)

# 绘制训练历史
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='训练MAE')
plt.plot(history.history['val_mae'], label='验证MAE')
plt.title('平均绝对误差')
plt.xlabel('轮次')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig('../results/training_history.png')

# 绘制预测结果与实际值的对比
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='实际值')
plt.plot(y_pred_actual, label='预测值')
plt.title('PM2.5预测结果对比')
plt.xlabel('时间步')
plt.ylabel('PM2.5浓度')
plt.legend()
plt.savefig('../results/prediction_comparison.png')

# 保存模型
model.save('../results/cnn_lstm_pm25_model.h5')

print("模型训练和评估完成，结果已保存。")