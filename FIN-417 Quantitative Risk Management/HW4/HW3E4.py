import numpy as np
import pandas as pd
import numpy.random



N = 10000
A = np.matrix([[1, 0, 0, 0],
    [1, 1, 0, 0],
    [-1, 2, 3, 0],
    [1, -1, 1, 1]])

x = numpy.random.standard_t(df = 5, size = (N, 4))
X = pd.DataFrame((A@(x.T)).T)
L = np.sum(X, axis = 1)

# Quesiton 1
VaR_095 = np.sort(L, axis = 0)[int(N*0.95)]
print("Value-at-Risk: ",VaR_095)

#Question 2
C = np.cov(X, rowvar=False)
evals, evecs = np.linalg.eig(C)
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]
evals = evals[idx]
print("First PC:")
print("Eigenvector:",evecs[:,0])
print("Eigenvalue:", evals[0])
print("Covariance matrix:\n", np.cov(X.T))

#Question 3
mu = np.mean(X.to_numpy())
Y = evecs.T@(X.to_numpy().T - mu) #estimate Y

X_approx =  mu + evecs[:,0:2]@Y[0:2,:] #Use the two first principal components with Y
L_approx = np.sum(X_approx, axis = 0)

VaR_095 = np.sort(L_approx, axis = 0)[int(N*0.95)]
print("Value-at-Risk: ",VaR_095)










