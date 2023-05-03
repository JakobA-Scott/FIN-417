import pandas as pd
import random
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
df= pd.read_csv('stock_prices_SP500_2007_2021.csv')
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

T = 250
alpha1 = .95
alpha2 = .99
stocks = df.drop(columns=['time'])
stocks = stocks.dropna(axis=1)  # Drop stocks with NaN-values
diff = np.zeros((len(stocks.iloc[:, 0]) - 1, len(stocks.iloc[0, :])))
no_factors = 5
q = np.random.rand(len(stocks.iloc[0, :]), 1)
q = q / np.sum(q)  # Random portfolio where the elements sum to 1.

"""Calculate loss"""
for i in range(len(stocks.iloc[:, 0]) - 1):
    diff[i, :] = (np.log(stocks.iloc[i, :]) - np.log(stocks.iloc[i + 1, :]))
VaR1 = []
VaR2 = []

"""Calculate VaR based on the factor approach with top 5 PCA elements as factors."""

for i in range(len(diff[:, 0]) - T):
    cov_matrix = np.cov(diff[i:T + i, :].T)
    eigenValues, eigenVectors = np.linalg.eig(cov_matrix)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]  # This matrix turns out to be orthonormal
    factors_diff = diff[i:i + T] @ eigenVectors[:, 0:no_factors]  # Factors are the principal components

    """From here, we do the same as in the previous part."""
    omega = np.cov(factors_diff.T)
    F = np.hstack((np.ones((T, 1)), factors_diff))
    B = np.linalg.inv(F.T @ F) @ F.T @ diff[i:i + T, :]  # OLS estimate of regressors
    a = B[0, :].T
    beta = B[1:, :].T
    residual = diff[i:i + T, :] - F @ B
    res_cov = np.cov(residual.T)
    res_cov = np.diagonal(res_cov) * np.identity(
        len(res_cov[:, 0]))  # We only want diagonal elements according to the assignment
    portfolio_sigma = beta @ omega @ beta.T + res_cov
    mean = np.mean(diff[i:i + T, :], axis=0)
    VaR1.append((norm.ppf(alpha1) * q.T @ portfolio_sigma @ q) ** .5)  # Calculates VaR as on page 48 of lecture 3
    VaR2.append((norm.ppf(alpha2) * q.T @ portfolio_sigma @ q) ** .5)  # Calculates VaR as on page 48 of lecture 3

VaR1 = np.array(VaR1)
VaR2 = np.array(VaR2)
plt.title('PCA-method')
plt.plot(VaR1[:, 0, 0], label='VaR95')
plt.plot(VaR2[:, 0, 0], label='VaR99')
plt.legend()
plt.grid(True)
plt.show()
plt.title('Portfolio returns')
plt.plot(diff @ q)
plt.grid(True)
plt.show()


"""Calculates number of breaches"""

portfolio_loss=diff@q
VaR1=np.array(VaR1)
VaR1=VaR1.reshape(-1,1)
portfolio_loss=np.array(portfolio_loss)
breaches=portfolio_loss[T:,:]>VaR1
print(np.sum(breaches))
print(np.sum(breaches)/len(VaR1))


VaR2=np.array(VaR2)
VaR2=VaR2.reshape(-1,1)
portfolio_loss=np.array(portfolio_loss)
breaches=portfolio_loss[T:,:]>VaR2
print(np.sum(breaches))
print(np.sum(breaches)/len(VaR2))