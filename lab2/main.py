import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


np.random.seed(42)
# Дискретное распределение — Пуассон (λ=3)
poisson_sample = np.random.poisson(lam=3, size=1000)

# Непрерывное распределение — Нормальное (μ=0, σ=1)
normal_sample = np.random.normal(loc=0, scale=1, size=1000)



# Описательные статистики
def describe_sample(sample):
    q1 = np.percentile(sample, 25)
    q2 = np.median(sample)
    q3 = np.percentile(sample, 75)

    mean = np.mean(sample)
    median = np.median(sample)
    mode = stats.mode(sample, keepdims=False).mode

    range_ = np.max(sample) - np.min(sample)
    iqr = q3 - q1
    var = np.var(sample)
    std = np.std(sample)
    cv = std / mean if mean != 0 else np.nan
    mad = np.mean(np.abs(sample - mean))

    skew = stats.skew(sample)
    kurt = stats.kurtosis(sample)

    moments = {
        f"moment_{i}": stats.moment(sample, moment=i)
        for i in range(1, 6)}

    return {
        'Q1': q1, 'Q2 (median)': q2, 'Q3': q3,
        'mean': mean, 'median': median, 'mode': mode,
        'range': range_, 'IQR': iqr, 'variance': var,
        'std': std, 'CV': cv, 'MAD': mad,
        'skewness': skew, 'kurtosis': kurt,
        **moments
        }

poisson_stats = describe_sample(poisson_sample)
normal_stats = describe_sample(normal_sample)

##print(poisson_stats)
##print(normal_stats)

# Графики
# CDF и tCDF
def plot_ecdf(sample, dist_name, dist_params):
    x = np.sort(sample)
    y = np.arange(1, len(x)+1) / len(x)

    plt.figure()
    plt.step(x, y, label='eCDF')

    # Теоретическая CDF
    if dist_name == 'normal':
        cdf = stats.norm.cdf(x, *dist_params)
    elif dist_name == 'poisson':
        cdf = stats.poisson.cdf(x, *dist_params)
    plt.plot(x, cdf, label='CDF теоретическая')
    plt.title(f"eCDF vs CDF — {dist_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_ecdf(normal_sample, 'normal', (0, 1))
plot_ecdf(poisson_sample, 'poisson', (3,))

# Гистограммы (ePDF vs PDF) — только для непрерывного
def plot_hist_with_pdf(sample, bins='auto'):
    x = np.linspace(min(sample), max(sample), 1000)
    pdf = stats.norm.pdf(x, loc=0, scale=1)

    plt.figure()
    plt.hist(sample, bins=bins, density=True, alpha=0.5, label='ePDF')
    plt.plot(x, pdf, 'r', label='PDF теоретическая')
    plt.title("ePDF vs PDF — нормальное распределение")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_hist_with_pdf(normal_sample)
plot_hist_with_pdf(normal_sample, bins=30)
plot_hist_with_pdf(normal_sample, bins=10)

# Многоугольник вероятностей — дискретное
def plot_poisson_polygon(sample):
    values, counts = np.unique(sample, return_counts=True)
    probs = counts / len(sample)

    x = np.arange(min(values), max(values)+1)
    theo_probs = stats.poisson.pmf(x, mu=3)

    plt.figure()
    plt.plot(values, probs, 'o-', label='эмпирическая')
    plt.plot(x, theo_probs, 's--', label='теоретическая')
    plt.title("Многоугольник вероятностей — Пуассон")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_poisson_polygon(poisson_sample)

# Boxplot-ы
def plot_boxplot(sample, title):
    plt.figure()
    sns.boxplot(x=sample)
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_boxplot(normal_sample, "Boxplot — нормальное распределение")
plot_boxplot(poisson_sample, "Boxplot — Пуассон")


# Устойчивость характеристик (для нормального)
# Генерация выбросов: 5% от размера выборки
n = len(normal_sample)
percentages = np.linspace(0, 0.05, 11)
variability_measures = {'std': [], 'MAD': [], 'CV': []}

for p in percentages:
    k = int(p * n)
    outliers = np.random.uniform(10, 20, k)
    sample_with_outliers = np.concatenate([normal_sample, outliers])

    std = np.std(sample_with_outliers)
    mad = np.mean(np.abs(sample_with_outliers - np.mean(sample_with_outliers)))
    mean = np.mean(sample_with_outliers)
    cv = std / mean if mean != 0 else np.nan

    variability_measures['std'].append(std)
    variability_measures['MAD'].append(mad)
    variability_measures['CV'].append(cv)

plt.figure()
for label, values in variability_measures.items():
    plt.plot(percentages * 100, values, label=label)
plt.title("Устойчивость мер вариабельности")
plt.xlabel("% выбросов")
plt.ylabel("Значение")
plt.grid(True)
plt.legend()
plt.show()







