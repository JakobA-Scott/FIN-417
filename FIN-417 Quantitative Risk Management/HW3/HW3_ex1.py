import pandas as pd
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import datetime
df= pd.read_csv('stock_prices_SP500_2007_2021.csv')
pd.set_option('display.max_rows',None)
dates=df.iloc[:,0]

"""Calculate rolling mean and covariance matrix over a T=250 window."""
T = 250
theta = 0.5
stocks = df.drop(columns=['time'])
stocks = stocks.dropna(axis=1)
# stocks = stocks.iloc[:, [j for j, c in enumerate(stocks.columns) if j != 69]]
diff = np.zeros((len(dates) - 1, len(stocks.iloc[0, :])))
for i in range(len(dates) - 1):
    diff[i, :] = (np.log(stocks.iloc[i, :]) - np.log(stocks.iloc[i + 1, :]))
    # diff[i,:]=((stocks.iloc[i,:])-(stocks.iloc[i+1,:]))
rolling_cov = []
rolling_mean = []
for i in range(len(dates) - T - 1):
    rolling_mean.append(np.mean(diff[i:i + T, :], axis=0))
    rolling_cov.append(np.cov(diff[i:i + T, :].T))
    N = len(rolling_cov[0][0, :])
    diag_cov = (np.diagonal(rolling_cov[i]) * np.identity(N)) ** .5
    rho = np.linalg.inv(diag_cov) @ rolling_cov[i] @ np.linalg.inv(diag_cov)
    rho = (1 - theta) * rho + theta * np.identity(N)
    rolling_cov[i] = diag_cov @ rho @ diag_cov

rolling_cov = np.array(rolling_cov)
rolling_mean = np.array(rolling_mean)
rolling_cov.shape

"""Create portfolio and calculate its mean loss and variance to estimate VaR."""
q=np.random.rand(len(stocks.iloc[0,:]),1)
q=q/np.sum(q)
mean_loss=rolling_mean@q
mean_loss_var=[]
alpha=.05
for i in range(len(mean_loss)):
    mean_loss_var.append(q.T@rolling_cov[i]@q)
mean_loss_var=np.array(mean_loss_var)
VaR=[]
for i in range(len(mean_loss)):
    VaR_i=mean_loss_var[i,0,0]**.5*norm.ppf(1-alpha)#+mean_loss[i,0]
    VaR.append(VaR_i)
plt.plot(VaR)
plt.show()
plt.plot(stocks@q)
plt.show()

portfolio_loss=diff@q
VaR=np.array(VaR)
VaR=VaR.reshape(-1,1)
portfolio_loss=np.array(portfolio_loss)
breaches=portfolio_loss[T:,:]>VaR
print(np.sum(breaches))