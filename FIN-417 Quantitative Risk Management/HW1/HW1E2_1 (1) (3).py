import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
degrees=3
N=10000
S_t=100
r_t=.05
sigma_t=.2
T=1
K=100
Delta=1/252
Cov=np.array([[.01**2,0,-.5*.01*10**(-3)],
     [0,10**(-8),0],
     [-.5*.01*10**(-3),0,10**(-6)]])
X=np.random.multivariate_normal([0,0,0],Cov,N)
C=np.zeros((N,2))
L=np.zeros((N,1))
r=np.zeros((N,2))
S=np.zeros((N,2))
sigma=np.zeros((N,2))
r[:,0]=r_t
S[:,0]=S_t
sigma[:,0]=sigma_t
S[:,1]=S[:,0]*np.exp(X[:,0])
r[:,1]=abs(r[:,0]+X[:,1])
sigma[:,1]=abs(sigma[:,0]+X[:,2])
def d1d2(S,K,r,sigma,T):
    d1=(np.log(S/K)+(r+1/2*sigma**2)*T)/(sigma*T**.5)
    d2=d1-sigma*T**.5
    return d1,d2
d_11,d_21=d1d2(S[:,0],K,r[:,0],sigma[:,0],T)
d_12,d_22=d1d2(S[:,1],K,r[:,1],sigma[:,1],T-Delta)
C[:,0]=S[:,0]*norm.cdf(d_11)-K*np.exp(-T)*norm.cdf(d_21)
C[:,1]=S[:,1]*norm.cdf(d_11)-K*np.exp(-(T-Delta))*norm.cdf(d_21)
L=-(C[:,1]-C[:,0])
plt.hist(L,bins=100,density=True, label="Histogram of L")
plt.title("Simulation of L when X is multivariate normal")
plt.ylabel("Density")
plt.xlabel("L")
plt.legend(loc="upper left")
plt.show()
