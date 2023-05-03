import numpy.random as rnd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def func1(x, p):
    return norm.cdf(x)*p + norm.cdf(x - 10)*(1-p)

def func2(x, p):
    scl = 2
    return norm.cdf(x, scale = scl)*p**2 + 2*norm.cdf(x - 10, scale = scl)*p*(1-p) + norm.cdf(x - 20 , scale = scl)*(1-p)**2

def binomial_search(function,target,a,b,n, p):
    mid = (a+b)/2
    if n==0:
        return mid
    
    eval = function(mid, p)
    if eval < target:
        return binomial_search(function, target, mid, b, n-1, p)
    else:
        return binomial_search(function, target, a, mid, n-1, p)


target = 0.99; z_1 = 0; z_2 = 10; n_bisections = 100; p = 0.991

ans = binomial_search(func1, target, z_1, z_2, n_bisections, p)
ans2 = binomial_search(func2, target, z_1, z_2, n_bisections, p)

print("Value for the first split:", ans, "sanity check (should be close to " + str(target)[0:5] + ":", func1(ans, p))
print("Value for the first split:", ans2, "sanity check (should be close to " + str(target)[0:5] + ":", func2(ans2, p))
print("VaR subadditive:", ans2 <= 2*ans)

