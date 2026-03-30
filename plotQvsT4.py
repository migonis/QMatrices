import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ManyQs.dat', delimiter=',')

ts = data[0,:]
Q1s_1 = data[1,:]
Q2s_1 = data[12,:]

fig = plt.figure(figsize = (9,6))
plt.semilogx(ts, Q1s_1, '.', color = 'green', label = 'Qs', ms=15)
plt.xlabel('$t_4$')
plt.ylabel('$Q^{(1)}_{B850}$')
plt.title('$Q_1$ vs time', fontsize = 20)
plt.legend(loc = 2)
plt.grid()
plt.show()
fig.savefig('Q1vst.png',dpi=400)

fig = plt.figure(figsize = (9,6))
plt.semilogx(ts, Q2s_1, '.', color = 'green', label = 'Qs', ms=15)
plt.xlabel('$t_4$')
plt.ylabel('$Q^{(2)}_{B850}$')
plt.title('$Q_2$ vs time', fontsize = 20)
plt.legend(loc = 2)
plt.grid()
plt.show()
fig.savefig('Q2vst.png',dpi=400)

fig = plt.figure(figsize = (9,6))
plt.semilogx(ts, Q1s_1, '.', color = 'blue', label = 'Q1s', ms=15)
plt.semilogx(ts, Q2s_1, 'x', color = 'green', label = 'Q2s', ms=15)
plt.xlabel('$t_4$')
plt.ylabel('$Q^{(1)/(2)}_{B850}$')
plt.title('$Q_1$ and $Q_2$ vs time', fontsize = 20)
plt.legend(loc = 2)
plt.grid()
plt.show()
fig.savefig('Q1nQ2vst.png',dpi=400)

fig = plt.figure(figsize = (9,6))
plt.plot(ts[:-3], Q1s_1[:-3], '.', color = 'blue', label = 'Q1s', ms=15)
plt.plot(ts[:-3], Q2s_1[:-3], 'x', color = 'green', label = 'Q2s', ms=15)
plt.xlabel('$t_4$')
plt.ylabel('$Q^{(1)/(2)}_{B850}$')
plt.title('$Q_1$ and $Q_2$ vs time', fontsize = 20)
plt.legend(loc = 'lower right')
plt.grid()
plt.show()
fig.savefig('Q1nQ2vst-not_log.png',dpi=400)