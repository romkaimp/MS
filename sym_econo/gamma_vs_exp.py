import numpy as np
import matplotlib.pyplot as plt

# Параметры
k = 2.5   # Параметр формы (>1 дает положительное последействие)
theta = 2.0 # Параметр масштаба
size = 10000 # Количество сгенерированных значений

# Генерация данных из Гамма-распределения
data_with_memory = np.random.gamma(k, theta, size)

# Сравним с показательным распределением с тем же средним
mean_exp = k * theta
data_memoryless = np.random.exponential(mean_exp, size)

# Построим гистограммы
plt.figure(figsize=(12, 5))
plt.hist(data_with_memory, bins=100, alpha=0.7, label=f'Гамма (k={k}, θ={theta})', density=True)
plt.hist(data_memoryless, bins=100, alpha=0.7, label='Показательное (без памяти)', density=True)
plt.legend()
plt.title('Сравнение Гамма и Показательного распределений')
plt.xlabel('Время')
plt.ylabel('Плотность вероятности')
plt.show()