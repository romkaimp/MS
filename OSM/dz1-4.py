import numpy as np
import matplotlib.pyplot as plt

n = 63
N = 3
alpha = 4 * (n-60) * np.sqrt(2)
beta = N
h = 0.05
tau = 0.05
L = 1.0
T = 1.0
x = np.arange(0, L + h, h)
t = np.arange(0, T + tau, tau)
nx = len(x)
nt = len(t)


# Аналитическое решение (предполагаемое)
def analytic_solution(t, x):
    return alpha * beta * x * (1 - x) * np.cos(np.pi * t / 2) + 2 * beta * (1 - t)


# Правая часть
def f(t, x):
    return (beta / 2) * (alpha * np.cos(np.pi * t / 2) - alpha * np.pi * (x - x ** 2) * np.sin(np.pi * t / 2) - 4)


# Явная схема
def explicit_scheme():
    phi = np.zeros((nt, nx))
    # Начальное условие
    phi[0, :] = alpha * beta * x * (1 - x) + 2 * beta
    # Граничные условия
    phi[:, 0] = 2 * beta * (1 - t)
    phi[:, -1] = 2 * beta * (1 - t)

    r = tau / (4 * h ** 2)

    for j in range(nt - 1):
        for i in range(1, nx - 1):
            phi[j + 1, i] = phi[j, i] + r * (phi[j, i + 1] - 2 * phi[j, i] + phi[j, i - 1]) + tau * f(t[j], x[i])

    return phi


# Неявная схема (метод прогонки)
def implicit_scheme():
    phi = np.zeros((nt, nx))
    # Начальное условие
    phi[0, :] = alpha * beta * x * (1 - x) + 2 * beta
    # Граничные условия
    phi[:, 0] = 2 * beta * (1 - t)
    phi[:, -1] = 2 * beta * (1 - t)

    r = tau / (4 * h ** 2)

    for j in range(nt - 1):
        # Коэффициенты для прогонки
        a = np.full(nx - 2, -r)
        b = np.full(nx - 2, 1 + 2 * r)
        c = np.full(nx - 2, -r)
        d = phi[j, 1:-1] + tau * f(t[j], x[1:-1])

        # Учет граничных условий
        d[0] += r * phi[j + 1, 0]
        d[-1] += r * phi[j + 1, -1]

        # Прогонка
        phi[j + 1, 1:-1] = thomas_algorithm(a, b, c, d)

    return phi


def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n - 1):
        c_[i] = c[i] / (b[i] - a[i - 1] * c_[i - 1])

    for i in range(1, n):
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i - 1])

    x = np.zeros(n)
    x[-1] = d_[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


# Вычисление решений
phi_explicit = explicit_scheme()
phi_implicit = implicit_scheme()

# Аналитическое решение в нужные моменты времени
t_05_idx = int(0.5 / tau)
t_1_idx = int(1.0 / tau)

phi_analytic_05 = analytic_solution(0.5, x)
phi_analytic_1 = analytic_solution(1.0, x)

# Погрешности
error_explicit_05 = np.abs(phi_explicit[t_05_idx, :] - phi_analytic_05)
error_implicit_05 = np.abs(phi_implicit[t_05_idx, :] - phi_analytic_05)
error_explicit_1 = np.abs(phi_explicit[t_1_idx, :] - phi_analytic_1)
error_implicit_1 = np.abs(phi_implicit[t_1_idx, :] - phi_analytic_1)

max_err = -10
for i in range(0, nt):
    err = max(np.abs(phi_implicit[i, :] - analytic_solution(i*tau, x)))
    if err > max_err:
        max_err = err

print("max_err", max_err)
# Графики
plt.figure(figsize=(12, 8))

# t=0.5
plt.subplot(2, 2, 1)
plt.plot(x, phi_explicit[t_05_idx, :], label='Явная схема')
plt.plot(x, phi_analytic_05, '--', label='Аналитическое')
plt.title('t=0.5 (явная схема)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, phi_implicit[t_05_idx, :], label='Неявная схема')
plt.plot(x, phi_analytic_05, '--', label='Аналитическое')
plt.title('t=0.5 (неявная схема)')
plt.grid(True)
plt.legend()

# t=1.0
plt.subplot(2, 2, 3)
plt.plot(x, phi_explicit[-1, :], label='Явная схема')
plt.plot(x, phi_analytic_1, '--', label='Аналитическое')
plt.title('t=1.0 (явная схема)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x, phi_implicit[-1, :], label='Неявная схема')
plt.plot(x, phi_analytic_1, '--', label='Аналитическое')
plt.title('t=1.0 (неявная схема)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Погрешности
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x, error_explicit_05, label='Явная, t=0.5')
plt.plot(x, error_implicit_05, label='Неявная, t=0.5')
plt.title('Погрешности при t=0.5')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, error_explicit_1, label='Явная, t=1.0')
plt.plot(x, error_implicit_1, label='Неявная, t=1.0')
plt.title('Погрешности при t=1.0')
plt.legend()

plt.tight_layout()
plt.show()

 #--------------------------------------------------------------

tau_small = 0.00125
t_small = np.arange(0, T + tau_small, tau_small)
nt_small = len(t_small)


def explicit_scheme_small_tau():
    phi = np.zeros((nt_small, nx))
    # Начальное условие
    phi[0, :] = alpha * beta * x * (1 - x) + 2 * beta
    # Граничные условия
    phi[:, 0] = 2 * beta * (1 - t_small)
    phi[:, -1] = 2 * beta * (1 - t_small)

    r = tau_small / (4 * h ** 2)

    for j in range(nt_small - 1):
        for i in range(1, nx - 1):
            phi[j + 1, i] = phi[j, i] + r * (phi[j, i + 1] - 2 * phi[j, i] + phi[j, i - 1]) + tau_small * f(t_small[j],
                                                                                                            x[i])

    return phi


phi_explicit_small = explicit_scheme_small_tau()

# Индексы для t=0.5 и t=1.0
t_05_idx_small = int(0.5 / tau_small)
t_1_idx_small = int(1.0 / tau_small)
# Погрешности
error_explicit_small_05 = np.abs(phi_explicit_small[t_05_idx_small, :] - phi_analytic_05)
error_explicit_small_1 = np.abs(phi_explicit_small[t_1_idx_small, :] - phi_analytic_1)

print(max(error_explicit_small_05), max(error_explicit_small_1))
# Графики
plt.figure(figsize=(12, 4))
print("Явная схема t=0.5:", phi_explicit_small[t_05_idx_small, :])
print("Явная схема t=1:", phi_explicit_small[t_1_idx_small, :])
print("Аналитическая схема t=0.5:", phi_analytic_05)
print("Аналитическая схема t=1:", phi_analytic_1)
plt.subplot(1, 2, 1)
plt.plot(x, phi_explicit_small[t_05_idx_small, :], label='Явная схема (малый τ)')
plt.plot(x, phi_analytic_05, '--', label='Аналитическое')
plt.grid(True)
plt.title('t=0.5 (явная схема с τ=0.005)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, phi_explicit_small[t_1_idx_small, :], label='Явная схема (малый τ)')
plt.plot(x, phi_analytic_1, '--', label='Аналитическое')
plt.grid(True)
plt.title('t=1.0 (явная схема с τ=0.005)')
plt.legend()

plt.tight_layout()
plt.show()

# Сравнение погрешностей
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(x, error_explicit_05, label='τ=0.05')
plt.plot(x, error_explicit_small_05, label='τ=0.005')
plt.title('Погрешности при t=0.5')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, error_explicit_1, label='τ=0.05')
plt.plot(x, error_explicit_small_1, label='τ=0.005')
plt.title('Погрешности при t=1.0')
plt.legend()

plt.tight_layout()
plt.show()