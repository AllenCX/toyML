import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, num=51)
#Before adding noise, y==x
y = np.linspace(0, 10, num=51)
t = np.random.normal(0, 0.5, 51)
y += t
plt.scatter(x, y)
plt.show()
