import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
def f(x, y):
  return x**2 + 2*y**2
def f_dx(x):
  return 2*x

def f_dy(y):
  return 4*y
"""

def f(x, y):
  return -np.cos(x)*np.cos(y)*np.exp(-((x-np.pi)**2 + (y-np.pi)**2))
"""
"""

def f_dx(x, y):
  h = 0.000000001
  return (f(x+h, y) - f(x, y))/h

def f_dy(x, y):
  h = 0.000000001
  return (f(x, y+h) - f(x, y))/h


def gradient_descent(lr, patience, x, y, pat_thresh):
  # loop ( tetha' = tetha - lr * f_dx/tetha )
  tetha = [x, y]

  fs = []
  while patience > 0:
    tetha[0] = x
    tetha[1] = y
    tetha[0] = tetha[0] - lr * f_dx(tetha[0], tetha[1])
    tetha[1] = tetha[1] - lr * f_dy(tetha[0], tetha[1])
    x = tetha[0]
    y = tetha[1]

    # patience
    f_res = f(x, y)
    fs.append(f_res)

    #if f_res < pat_thresh:
    if len(fs) > 10:
      if min(fs[:-10]) - min(fs[-10:]) < pat_thresh:
        patience -= 1

    #print(f_res)

  return min(fs)


N = 10
lr_range = np.linspace(1e-4, 1e-2, N)
pat_range = np.linspace(1e-6, 1e-3, N)
interval = [-10, 10]

M = np.zeros((N, N))
for i,lr in tqdm(enumerate(lr_range), total=len(lr_range)):
  for j,pat_thresh in enumerate(pat_range):
    avg_f_min = []
    for _ in range(100):
      X = np.random.uniform(low=interval[0], high=interval[1], size=(1,))
      Y = np.random.uniform(low=interval[0], high=interval[1], size=(1,))

      f_min = gradient_descent(lr, 10, X[0], Y[0], pat_thresh)
      avg_f_min.append(f_min)

    M[i, j] = sum(avg_f_min)/len(avg_f_min)

# quantitize values between 0 and 255
#M_quant = np.zeros((N, N))
#M_quant = (M/np.max(M) * 255).astype(np.uint8)

#fig, ax = plt.subplots(1,1)
#img = ax.imshow(M_quant, extent=[min(lr_range), max(lr_range), min(pat_range), max(pat_range)], aspect='auto')
#M = M / N

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
xv, yv = np.meshgrid(lr_range, pat_range)
ax.plot_surface(xv, yv, M, cmap='viridis', edgecolor='none')
ax.set_xlabel('Learning rate')
ax.set_ylabel('Patience threshold')

#fig.colorbar(img)
plt.show()