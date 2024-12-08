import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import os
from sympy import symbols, pi, cos, sin, summation, exp, sinh, sqrt

os.environ['ffmpeg'] = 'E:/ffmpeg/ffmpeg-7.0.2-essentials_build/bin/ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = 'E:/ffmpeg/ffmpeg-7.0.2-essentials_build/bin/ffmpeg'
plt.rcParams['savefig.bbox'] = 'tight'

x_range = np.linspace(0, 3, 30)
t_range = np.linspace(0, 10, 100)

x, t, k = symbols("x t k")
l = 3
b = pi**2/l**2
a = 1

y = ( 30*(1-exp(-t/5))*sinh(sqrt(b)/a*(l-x))/sinh(sqrt(b)/a*l) +
     summation(
         (2000*(cos(pi*k)-1)/(pi*k*(k**2-4))*exp(-0.005*(k**2+1)*t) -
          12*pi*k*a**2/(b*l**2+pi**2*k**2*a**2) * (1/(pi**2*k**2/l**2+b-1/5)) * (exp(-t/5) - exp(-(pi**2*k**2/l**2+b)*t)) ) * sin(pi*k*x/l)
         , (k, 1, 1)) +
     summation(
         (2000*(cos(pi*k)-1)/(pi*k*(k**2-4))*exp(-0.005*(k**2+1)*t) -
          12*pi*k*a**2/(b*l**2+pi**2*k**2*a**2) * (1/(pi**2*k**2/l**2+b-1/5)) * (exp(-t/5) - exp(-(pi**2*k**2/l**2+b)*t)) ) * sin(pi*k*x/l)
         , (k, 3, 100)))

print(y.evalf(n=5, subs={t:0, x: 2}), 500*sin(pi*2/l)*sin(pi*2/l))
print(y.evalf(n=5, subs={x:0, t:5}))
print(y.evalf(n=5, subs={x:l, t:5}))
