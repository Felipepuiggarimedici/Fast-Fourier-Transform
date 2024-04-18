import numpy as np
from FFT import fft
from scipy.integrate import odeint
from matplotlib import pyplot as plt 

L = 90
N = 2**8
dx = L/N
x = np.arange(0,L, dx)
a = 0.0001
b = 0.001
dev = 15
mean = 40
dt = 0.25
t = np.arange(0, 100*dt, dt)
m = a*x**2+b*x


kappa = 2*np.fft.fftfreq(N, dx)
p_0 = 1/(dev*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*dev**2))

def dp_dt(p, t, kappa, m):
    p_aHat = -1j*kappa*fft(p + 0j)
    p_a = fft(p_aHat, inverse=True)
    dp_dt = -(p_a + m*p)
    return dp_dt.real

p = odeint(dp_dt, p_0, t, args=(kappa,m))

plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for j in range(0,len(t), 5):
    ax.plot(np.full_like(x, t[j]), x, p[j])
ax.set_xlabel('t')
ax.set_ylabel('a')
ax.set_zlabel('Population density')
ax.set_title('3D Line Plot')
plt.show()