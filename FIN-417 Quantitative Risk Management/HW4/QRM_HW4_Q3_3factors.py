import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


alpha = .95
T = 250
theta = 1

df = pd.read_csv('stock_prices_SP500_2007_2021.csv')
factors = df[['SPY', 'VXX', 'GLD']]
stocks = df.drop(columns=['time'])
stocks = stocks.dropna(axis=1)  # Drop stocks with NaN-values
regressors = pd.concat([factors, stocks], axis=1,
                       sort=False)  # Create a matrix with all factors and stocks. The first 3 columns are the independent variables in the regression model.
regressors = regressors.dropna(
    axis=0)  # Drop all days with any NaN values. This removes about 1/4 of the data in the beginning of the period
diff = np.zeros((len(regressors.iloc[:, 0]) - 1, len(regressors.iloc[0, :])))

q = np.random.rand(len(stocks.iloc[0, :]), 1)
q = q / np.sum(q)  # Random portfolio where the elements sum to 1.

"""Calculates loss"""
for i in range(len(regressors.iloc[:, 0]) - 1):
    diff[i, :] = (np.log(regressors.iloc[i, :]) - np.log(regressors.iloc[i + 1, :]))

stocks_diff = diff[:, 3:]
factors_diff = diff[:, 0:3]
VaR = []

"""Calculates VaR. Formulas are largely obtained from page 22 onwards in the lectures notes of lecture 4."""

for i in range(len(diff[:, 0]) - T):
    omega = np.cov(factors_diff[i:i + T, :].T)
    F = np.hstack((np.ones((T, 1)), factors_diff[i:i + T, :]))
    B = np.linalg.inv(F.T @ F) @ F.T @ stocks_diff[i:i + T, :]  # OLS estimate of regressors
    a = B[0, :].T
    beta = B[1:, :].T
    residual = stocks_diff[i:i + T, :] - F @ B
    res_cov = np.cov(residual.T)
    res_cov = np.diagonal(res_cov) * np.identity(
        len(res_cov[:, 0]))  # We only want diagonal elements according to the assignment
    portfolio_sigma = beta @ omega @ beta.T + res_cov
    mean = np.mean(stocks_diff[i:i + T, :], axis=0)
    VaR.append((norm.ppf(alpha) * q.T @ portfolio_sigma @ q) ** .5)  # Calculates VaR as on page 48 of lecture 3

VaR = np.array(VaR)
plt.plot(VaR[:, 0, 0])
plt.show()

"""Calculates number of breaches"""

portfolio_loss=stocks_diff@q
VaR=np.array(VaR)
VaR=VaR.reshape(-1,1)
portfolio_loss=np.array(portfolio_loss)
breaches=portfolio_loss[T:,:]>VaR
print('The number of breaches is {0}'.format(np.sum(breaches)))