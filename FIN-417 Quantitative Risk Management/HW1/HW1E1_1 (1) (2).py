import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
d=1
llambda=1
S_t=100
N=10000
#degrees=10
std=.01
M=1
#alpha=np.sqrt(degrees/(degrees-2))
S=np.zeros((N,2))
S[:,0]=S_t
L=np.zeros((N,1))
X=std*np.random.normal(0,1,(N,1))
#X=1/alpha*std*X

S[:,1]=S[:,0]*np.exp(X[:,0])
L[:,0]=-(S[:,1]-S[:,0])
#plt.title("Simulation for "+str(degrees)+" degrees of freedom")
plt.title("Simulation for the normal distribution")
plt.ylabel("Density")
plt.xlabel("L(t,t+1)")
def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)
x=np.linspace(-5,5,100)
plt.hist(L,bins=100,density=True, label="Histogram of L")
plt.axis([-5,5,0,.8])
plt.plot(x,stats.norm.pdf(x, np.mean(L), np.std(L)),label="Fitted normal PDF")
plt.legend(loc="upper left")
plt.show()