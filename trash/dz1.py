#from sympy import symbols, exp, I, residue, simplify, roots, pi
#from sympy.abc import w, t
import numpy as np
np.set_printoptions(suppress=True, precision=3)

# a, b = 50, 70
#
# x = np.array([[1, 5],
#               [6, 7]])
# d = np.array([1, 7])
# v = np.array([1, 7])

a, b = 70, 10
x = np.array([[1, 4],
               [5, 3]])
d = np.array([3, 7])
v = np.array([5, 5])

print("1) Суммы x в строках +d:", x.sum(axis=1) + d)
A = x / (x.sum(axis=1) + d)
print("2) Матрица Леонтьева A=", A)
print("det(A) =", np.linalg.det(A))
print(np.eye(A.shape[0]) - A)
H = np.linalg.inv(np.eye(A.shape[0]) - A)
print("3) Матрица H:", H)
print("Проверка (E-A)^(-1) @ H=", (np.eye(A.shape[0]) - A)@H)
print("Проверка x=Hd=", H@d, "x=", x)
d_ = np.array([d[0]*(1 + a/100), d[1]*(1 - b/100)])
x_ = H@d_
print("4) (x'=) d'=", d_)
print("5) новый Валовый выпуск:", x_)

X = x.sum(axis=1) + d
delta_1 = (x_[0] - X[0])/ X[0]
delta_2 = (x_[1] - X[1])/X[1]
print("6) delta_1=", delta_1)
print("   delta_2=", delta_2)

p = H.T @ v
print("7) p=", p)