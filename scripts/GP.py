
"""
@author: <alberto.suarez@uam.es>
         <Antonio Coín Castro>
         <Luis Antonio Ortega Andrés>
"""

# Load packages
from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_gp(
    t: np.ndarray,
    mean_fn: Callable[[np.ndarray], np.ndarray],
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    M: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a Gaussian process.
        X(t) ~ GP(mean_fn,kernel_fn)
    Parameters
    ----------
    t :
        Times at which the process is monitored.
    mean_fn:
        Mean function of the Gaussian process (vectorized).
    kernel_fn:
        Covariance functions of the Gaussian process (vectorized).
    M :
        Number of trajectories that are simulated.
    Returns
    -------
    X:
        Simulated trajectories as an np.ndarray with M rows and len(t) columns.
        Each trajectory is a row of the matrix consisting of the
        values of the process for each value of t.
    mean_vector:
        Vector with the values of the mean for each value of t.
        It is a np.ndarray with len(t) columns.
    kernel_matrix:
        Kernel matrix as an np.ndarray with len(t) rows and len(t)  columns.
    """
    # Compute kernel matrix using auxiliary function
    kernel_matrix = kernel_fn(t, t, gamma = 10)
    print(kernel_matrix.shape)
    # SVD decomposition and transform s to matrix
    U, s, Vh = np.linalg.svd(kernel_matrix)
    S = np.diag(s)

    # Sample from standard Gaussian
    Z = np.random.randn(M, len(t))

    # Compute mean of Gaussian process at each time
    mu = mean_fn(t)
    print(mu.shape)
    # Generate Gaussian process samples using SVD decomposition
    X = Z@np.sqrt(S)@U.T + mu.T

    return X, mu, kernel_matrix



def mean_fn(t):
    return np.zeros(np.shape(t))

M, N  = (100, 1000)
t0, t1 = (0.0, 1.0)
t = np.linspace(t0, t1, N).reshape(-1,1)

print(t[900])
BB, _, _ = simulate_gp(t, mean_fn, rbf_kernel, M)
gauss = BB[:, 900] 


# fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[2,1]})

# ax1.plot(t[900]*np.ones(M), BB[:, 900], "bo")
# ax1.plot(t[500]*np.ones(M), BB[:, 500], "bo" , c = "orange")
# ax1.plot(t, BB.T)
# ax1.set_xlabel('t')
# ax1.axvline(x = t[900])
# ax1.axvline(x = t[500], c = "orange")
# gauss2 = BB[:, 500]
# #ax2.hist(gauss, density = True, orientation = "horizontal")
# ax2 = sns.kdeplot(y = gauss, ax = ax2)
# sns.kdeplot(y = gauss2, ax =ax2, color="orange")
# plt.show()


# plt.plot(t, BB.T, lw = 0.8)
# plt.plot(t, BB.T[:,5], lw = 2, color = "black")
# a = [100, 300, 400, 600, 700, 800]
# print(t[a])
# plt.plot(t[a], BB.T[a, 5], "bo")
# plt.show()


plt.plot(t, BB.T[:,5], lw = 1)
a = [100, 300, 400, 600, 700, 800]
b = [200, 500, 650]
plt.plot(t[a], BB.T[a, 5], "bo", color = "blue", label = "Known observations")
plt.plot(t[b], BB.T[b, 5], "bo", color = "orange", label = "Unknown inducing points")
plt.legend()
plt.show()



