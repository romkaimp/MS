import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True) 
np.set_printoptions(threshold=np.inf) 
np.set_printoptions(precision=4)

N = 3
n = 63
# равномерная сетка А и центрально равномерная сетка B
A = np.linspace(0, 2, 41)
B = (A[1:] + A[:-1]) / 2

print("A:", A)
print("B:", B)
K = np.matrix([[0.0] * 40] * 40)
# Аналитические выражения компонент матрицы через узлы коллокаций и шаг сетки в общем случае:
for i in range(40):
    for j in range(40):
        if i < j:
            K[i, j] = B[i] * (2 * A[j + 1] - ((A[j + 1]) ** 2) / 2 - 2 * A[j] + ((A[j]) ** 2) / 2)
        if i == j:
            K[i, j] = (2 - B[i]) * (B[i] ** 2 - A[i] ** 2) / 2 + B[i] * (2 * A[j + 1] - ((A[j + 1]) ** 2) / 2 - 2 * B[j] + ((B[j]) ** 2) / 2)
        if i > j:
            K[i, j] = (2 - B[i]) * (A[j + 1] ** 2 - A[j] ** 2) / 2
np.set_printoptions(suppress=True, threshold=5)
print("K:\n", K)

print('-----------------------------------------------------------------------------------')
lambd = (-1)**N * (N + 1) / (N * (67 - n))
F = np.eye(40)
F -= lambd * K
print("F:\n", F)
print('-----------------------------------------------------------------------------------')

def f(x):
    return (-1)**N * (N + 4) / N * np.sin((N + 1) / N * np.pi * x) + (N + 4) / (N + 1) * x + 64 - n

y = (-1)**N * (N + 4) / N * np.sin((N + 1) / N * np.pi * B) + (N + 4) / (N + 1) * B + 64 - n
print("y:\n")
for el in y:
    print(round(el, 4))
print('-----------------------------------------------------------------------------------')

print("x:\n")
x = np.linalg.inv(F) @ y
for el in x:
    print(round(el, 4))
print('-----------------------------------------------------------------------------------')


# аналитическое решение
r = sqrt(2 / 3)
def f_analytical(x):
    return 0.8592 * np.e**(r * x) + 0.1408 * np.e**(-r * x) - 2.248 * np.sin(4*np.pi/3 * x)

anal = np.linspace(0, 2, 40)
plt.plot(B, x, 'ro', label='метод коллокаций')
plt.plot(anal, f_analytical(anal), label='аналитический метод')
plt.legend(frameon=False)
plt.show()

print("Absolute error: ", max(x - f_analytical(B)))

#приближённое решение уравнения, имеющее вид частичной суммы ряда Фурье
def furie(x):
    return (-1)**N * (N + 4) / N * np.sin((N + 1) / N * np.pi * x) + (N + 4) / (N + 1) * x + 64 - n\
            +lambd*(2.36875 * np.sin(np.pi/2 * x) + 0.368456 * np.sin(np.pi * x) + 0.076598 * np.sin(3*np.pi/2 * x))



# Построение графика для сравнения
anal = np.linspace(0, 2, 40)
plt.plot(B, furie(B), 'o', label='метод Фурье')
plt.plot(anal, f_analytical(anal), label='аналитический метод')
plt.legend(frameon=False)
plt.show()

print("Absolute error: ", max(furie(B) - f_analytical(B)))
print("f(x):\n", furie(B))