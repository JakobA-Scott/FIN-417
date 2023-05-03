import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rnd

#region functions for task 1 a-b --------------------
def geom_pmf(k,p):
    return p*(1-p)**k

def VaR_geom(alpha, p = 0.5):
    return np.ceil(np.log(1-alpha)/np.log(1-p) - 1)

#endregion ------------------

#1a)

var_095 = VaR_geom(0.95)
print(var_095)

#1b)
alphas = np.arange(0.9,1, 0.01)
VaR_alphas = VaR_geom(alphas)

plt.step(alphas, VaR_alphas)
plt.xlabel(r"Level of significance $\alpha$")
plt.ylabel(r"$VaR_\alpha$")
plt.grid()
plt.show()

x_axis = np.array(range(0, 8))
fig, ax = plt.subplots()

plt.step(x_axis, geom_pmf(x_axis, 0.5), label = r"$f_L(x)$", where="post")

#Draw a vline for every unique VaR_alpha
colors = ["red", "blue", "green", "purple", "pink"]
for i in range(len(set(VaR_alphas))):
    val = list(VaR_alphas).index(list(set(VaR_alphas))[i])
    interval = np.where(VaR_alphas == VaR_alphas[val])
    plt.vlines(VaR_alphas[val], 0 , geom_pmf(VaR_alphas[val], 0.5), colors=colors[i], linestyles="dashed", label = r"$\alpha\in$" + "[" + str(alphas[interval][0])[0:4] + "," + str(alphas[interval][-1])[0:4]  + "]")
plt.xlabel(r"$x$")
plt.ylabel(r"$f_L(x)$")
plt.grid()
plt.legend()
plt.show()

#2a) 

#region functions -------------
def fac(n):
    if n == 0:
        return 1
    return n*fac(n-1)

def poisson_pdf(x, lamb):
    return np.exp(-lamb)*lamb**x/fac(x)

def poisson_cdf(x, lamb):
    if x < 0:
        return 0
    cumsum = 0
    for i in range(int(x) + 1):
        cumsum += poisson_pdf(i, lamb)
    return cumsum

def VaR_poisson(alpha, lamb):
    part = None
    if not isinstance(alpha, list) and not isinstance(alpha, np.ndarray):
        alpha = [alpha]
    if len(alpha) != 1:
        part = VaR_poisson(alpha[1:], lamb)

    x = 0
    while(poisson_cdf(x, lamb) < alpha[0]):
        x += 1

    if part == None:
        return [x]

    return [x] + part

def plot_VaR_poisson(alpha, lamb):
    VaR_alphas = VaR_poisson(alpha, lamb)
    plt.step(alphas, VaR_alphas, where = "mid")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$VaR_\alpha$")
    plt.grid()
    plt.show()

#endregion ---------------------

alphas = np.arange(0.9, 1, 0.01)

plot_VaR_poisson(alphas, 1)
plot_VaR_poisson(alphas, 2)
plot_VaR_poisson(alphas, 3)




