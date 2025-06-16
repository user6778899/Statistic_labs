import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, acf


data = yf.download('AAPL', start='2020-01-01', end='2024-12-31')
ts = data['Close']

# Визуализация временного ряда
plt.figure(figsize=(10, 4))
plt.plot(ts)
plt.title('Цена закрытия AAPL')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.grid(True)
plt.show()

# Разностные ряды 1-го и 2-го порядка
ts_diff1 = ts.diff().dropna()
ts_diff2 = ts.diff().diff().dropna()

plt.figure(figsize=(10, 4))
plt.plot(ts_diff1)
plt.title('1-я разность')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(ts_diff2)
plt.title('2-я разность')
plt.grid(True)
plt.show()

# Автокорреляционная функция
acf_values = acf(ts.dropna(), nlags=40)

plt.figure(figsize=(10, 4))
plt.stem(acf_values)
plt.title('АКФ исходного ряда')
plt.grid(True)
plt.show()


# Тест Дики-Фуллера
def test_adf(series, name):
    result = adfuller(series.dropna())
    print(f'ADF тест для {name}')
    print('ADF статистика:', result[0])
    print('p-значение:', result[1])
    print('Критические значения:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    print('')

test_adf(ts, 'исходного ряда')
test_adf(ts_diff1, '1-й разности')
test_adf(ts_diff2, '2-й разности')

# Экспоненциальное сглаживание
def exponential_smoothing(series, alpha):
    result = [series.iloc[0]]
    for i in range(1, len(series)):
        result.append(alpha * series.iloc[i] + (1 - alpha) * result[i-1])
    return pd.Series(result, index=series.index)

for alpha in [0.2, 0.5, 0.8]:
    smoothed = exponential_smoothing(ts, alpha)
    plt.figure(figsize=(10, 4))
    plt.plot(ts, label='Исходный')
    plt.plot(smoothed, label=f'Сглаженный (alpha={alpha})')
    plt.legend()
    plt.title(f'Экспоненциальное сглаживание, alpha={alpha}')
    plt.grid(True)
    plt.show()


















