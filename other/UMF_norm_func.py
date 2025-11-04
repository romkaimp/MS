from sympy import symbols, integrate, sqrt, oo, summation, pi, sin

# Определяем переменные
x, j = symbols('x j')

# Определяем функции
f1 = summation(sin(pi * j * x/3), (j, 0, 100))
f2 = sin(pi * x/3)
# Ваша первая функция, например, sin(x)
# Добавьте больше функций по необходимости

# Определите интервал
a = 0  # Начало интервала
b = 3 # Конец интервала

# Вычисляем сумму квадратов функций
sum_of_squares = f2**2 # + добавьте больше, если есть

# Интегрирование суммы квадратов на интервале [a, b]
integral_value = integrate(sum_of_squares, (x, a, b))

# Извлекаем квадратный корень из результата интеграла
euclidean_norm = sqrt(integral_value)

# Вывод результата
print(euclidean_norm)
