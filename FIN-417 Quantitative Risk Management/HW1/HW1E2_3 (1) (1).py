import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
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

def d1d2(S,K,r,sigma,T):
    d1=(np.log(S/K)+(r+1/2*sigma**2)*T)/(sigma*T**.5)
    d2=d1-sigma*T**.5
    return d1,d2

d_11,d_21=d1d2(S_t,K,r_t,sigma_t,T)
Theta=-S_t*sigma_t*stats.norm.pdf(d_11, 0, 1)/(2*np.sqrt(T))-K*r_t*np.exp(-T*r_t)*stats.norm.cdf(d_21, 0, 1)
delta=stats.norm.cdf(d_11, 0, 1)
rho=T*K*np.exp(-r_t*T)*stats.norm.cdf(d_21, 0, 1)
vega=np.sqrt(T)*S_t*stats.norm.pdf(d_11, 0, 1)

L_delta=-(Theta*Delta+S_t*delta*X[:,0]+rho*X[:,1]+vega*X[:,2])
risk1=S_t*delta*X[:,0]
risk2=rho*X[:,1]
risk3=vega*X[:,2]
print(np.std(risk1))
print(np.std(risk2))
print(np.std(risk3))
plt.hist(L_delta,bins=100,density=True, label="Histogram of L")
plt.title("Simulation of linearized L when X is multivariate normal")
plt.ylabel("Density")
plt.xlabel("Linearized L")
plt.legend(loc="upper left")
plt.show()
