import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
x = np.random.normal(loc=0, scale=1, size=500)

# 1. Точечные оценки
mean_x = np.mean(x)
med_x = np.median(x)
var_x = np.var(x, ddof=1)
iqr_x = np.percentile(x, 75) - np.percentile(x, 25)

print("Среднее:", mean_x)
print("Медиана:", med_x)
print("Дисперсия:", var_x)
print("IQR:", iqr_x)

# 2. Гистограммы + KDE
plt.figure(figsize=(15, 4))
bins_list = [10, 30, 50]
for i, b in enumerate(bins_list):
    plt.subplot(1, 3, i + 1)
    sns.histplot(x, bins=b, kde=True, color='skyblue')
    plt.title(f'{b} бинов')
plt.tight_layout()
plt.show()

# 3. Бутстрап
B = 1000
n = len(x)
boot_means = []
boot_medians = []
boot_vars = []
boot_iqrs = []

for _ in range(B):
    sample = np.random.choice(x, size=n, replace=True)
    boot_means.append(np.mean(sample))
    boot_medians.append(np.median(sample))
    boot_vars.append(np.var(sample, ddof=1))
    iqr = np.percentile(sample, 75) - np.percentile(sample, 25)
    boot_iqrs.append(iqr)

# Гистограммы бутстрап-оценок
plt.figure(figsize=(12, 8))
for i, (stat, title) in enumerate(zip(
    [boot_means, boot_medians, boot_vars, boot_iqrs],
    ["Среднее", "Медиана", "Дисперсия", "IQR"])):
    plt.subplot(2, 2, i + 1)
    sns.histplot(stat, kde=True, color='lightgreen')
    plt.axvline([mean_x, med_x, var_x, iqr_x][i], color='red', label='Исходная оценка')
    plt.title(title)
    plt.legend()
plt.tight_layout()
plt.show()

# 4. Доверительные интервалы
def ci(data, alpha):
    low = np.percentile(data, 100 * alpha / 2)
    high = np.percentile(data, 100 * (1 - alpha / 2))
    return low, high

for alpha in [0.1, 0.05, 0.01]:
    ci_mean = ci(boot_means, alpha)
    ci_med = ci(boot_medians, alpha)
    print(f"{100*(1-alpha):.0f}% CI для среднего: {ci_mean}")
    print(f"{100*(1-alpha):.0f}% CI для медианы: {ci_med}")

# 5. Влияние размера выборки
Ns = [50, 100, 200, 500, 1000]
widths = []

for N in Ns:
    x_small = np.random.normal(0, 1, N)
    boot = []
    for _ in range(B):
        s = np.random.choice(x_small, N, replace=True)
        boot.append(np.mean(s))
    low, high = ci(boot, 0.05)
    widths.append(high - low)

plt.plot(Ns, widths, marker='o')
plt.title("Ширина 95% CI от размера выборки")
plt.xlabel("Размер выборки")
plt.ylabel("Ширина интервала")
plt.grid()
plt.show()

# 6. Влияние числа итераций
Bs = [100, 200, 400, 1600, 3200]
widths_b = []

for B2 in Bs:
    boot = []
    for _ in range(B2):
        s = np.random.choice(x, n, replace=True)
        boot.append(np.mean(s))
    low, high = ci(boot, 0.05)
    widths_b.append(high - low)

plt.plot(Bs, widths_b, marker='o')
plt.title("Ширина 95% CI от числа итераций")
plt.xlabel("Число бутстрап-итераций")
plt.ylabel("Ширина интервала")
plt.grid()
plt.show()

# 7. Проверка покрытия
Ns = [50, 100, 200, 500, 1000]
Bs = [100, 200, 400, 1600, 3200]
cover = np.zeros((len(Ns), len(Bs)))

for i, N in enumerate(Ns):
    for j, B2 in enumerate(Bs):
        count = 0
        for _ in range(100):
            x1 = np.random.normal(0, 1, N)
            boot = []
            for _ in range(B2):
                s = np.random.choice(x1, N, replace=True)
                boot.append(np.mean(s))
            low, high = ci(boot, 0.05)
            if 0 >= low and 0 <= high:
                count += 1
        cover[i, j] = count / 100

# Тепловая карта
plt.figure(figsize=(8, 5))
sns.heatmap(cover, annot=True, xticklabels=Bs, yticklabels=Ns, cmap="YlGnBu", fmt=".2f")
plt.xlabel("B (итераций бутстрапа)")
plt.ylabel("N (размер выборки)")
plt.title("Покрытие 95% интервалов")
plt.show()
