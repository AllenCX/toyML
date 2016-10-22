import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
x = np.linspace(0, 10, num=51).reshape(51, 1)
y = np.linspace(0, 10, num=51).reshape(51, 1)
t = np.random.normal(0, 2, (51, 1))
y += t
beta_hat = inv(x.T.dot(x)).dot(x.T).dot(y)

fig, ax = plt.subplots()
ax.scatter(x, y, 'b')
ax.plot(x, beta_hat*x, 'r')
plt.show()

