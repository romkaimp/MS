import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import os
from sympy import diff, symbols, integrate, Rational, solve, Pow, limit, pi, cos, sin, summation, expand, simplify

os.environ['ffmpeg'] = 'E:/ffmpeg/ffmpeg-7.0.2-essentials_build/bin/ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = 'E:/ffmpeg/ffmpeg-7.0.2-essentials_build/bin/ffmpeg'
plt.rcParams['savefig.bbox'] = 'tight'

x_range = np.linspace(0, 3, 30)
t_range = np.linspace(0, 10, 100)

x, t, j = symbols("x t j")
a = 1
#y = (((1 - 3/(a**2 * pi**3 * 100 - 3*pi))*cos(a*pi*t/3) + 3*cos(t/10)/(a**2*pi**3*100-3*pi))*sin(pi*x/3) +
#     summation( ((- 3/(a**2 * (j*pi)**3 * 100 - 3*pi*j)) * cos(a*pi*t*j/3) + 3*cos(t/10)/(a**2*(pi*j)**3*100-3*pi*j)
#                 )*sin(pi*x*j/3) , (j, 2, oo)) +
#cos(t/10) - x/3*cos(t/10))
y = (((1 - 3/(a**2 * pi**3 * 100 - 3*pi))*cos(a*pi*t/3) + 3*cos(t/10)/(a**2*pi**3*100-3*pi))*sin(pi*x/3) +
     summation( ((- 3/(a**2 * (j*pi)**3 * 100 - 3*pi*j)) * cos(a*pi*t*j/3) + 3*cos(t/10)/(a**2*(pi*j)**3*100-3*pi*j)
                )*sin(pi*x*j/3) , (j, 2, 20)) +
cos(t/10) - x/3*cos(t/10))

y_t = diff(y, t)
y_tt = diff(y_t, t)
y_x = diff(y, x)
y_xx = diff(y_x, x)
print(y_tt.evalf(n=6, subs={x: 2, t: 1}), y_xx.evalf(n=6, subs={x: 2, t:1}))
print(simplify(y_tt - y_xx))
print(y.subs({x:0}))
print(y.subs({x:3}))
print(y.subs({t:0}))
print(y_t.subs({t:0}))
