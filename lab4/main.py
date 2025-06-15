import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


np.random.seed(42)

# Выборка 1: близкие мат. ожидания, одинаковые дисперсии
group1_a = np.random.normal(5, 1, 30)
group1_b = np.random.normal(5.2, 1, 30)
group1_c = np.random.normal(5.1, 1, 30)

# Выборка 2: разные мат. ожидания
group2_a = np.random.normal(5, 1, 30)
group2_b = np.random.normal(7, 1, 30)
group2_c = np.random.normal(9, 1, 30)


# Ядерные графики
def plot_kde(groups, title):
    plt.figure(figsize=(8, 5))
    for g in groups:
        sns.kdeplot(g, fill=True)
    all_data = np.concatenate(groups)
    sns.kdeplot(all_data, fill=False, color='black', linestyle='--', label='Объединение')
    plt.title(title)
    plt.legend(['Группа 1', 'Группа 2', 'Группа 3', 'Объединение'])
    plt.show()

plot_kde([group1_a, group1_b, group1_c], "KDE для выборки 1")
plot_kde([group2_a, group2_b, group2_c], "KDE для выборки 2")


# Парные тесты
# Известные дисперсии (z-тест)
def z_test(x, y, var):
    n = len(x)
    z = (np.mean(x) - np.mean(y)) / np.sqrt(var / n + var / n)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

# Неизвестные, но равные дисперсии (t-тест)
def t_test_equal_var(x, y):
    t, p = stats.ttest_ind(x, y, equal_var=True)
    return t, p

def test_all_pairs(groups, name):
    print(f"\nПроверка парных тестов для {name}:")
    for i in range(3):
        for j in range(i+1, 3):
            print(f"\nГруппа {i+1} и Группа {j+1}:")
            var = 1  # известно
            z, pz = z_test(groups[i], groups[j], var)
            print(f"Z-тест: z = {z:.2f}, p = {pz:.4f}")

            t, pt = t_test_equal_var(groups[i], groups[j])
            print(f"T-тест: t = {t:.2f}, p = {pt:.4f}")

test_all_pairs([group1_a, group1_b, group1_c], "Выборка 1")
test_all_pairs([group2_a, group2_b, group2_c], "Выборка 2")

# ANOVA
def do_anova(groups, name):
    f, p = stats.f_oneway(*groups)
    print(f"\nANOVA для {name}: F = {f:.2f}, p = {p:.4f}")

do_anova([group1_a, group1_b, group1_c], "Выборка 1")
do_anova([group2_a, group2_b, group2_c], "Выборка 2")


















