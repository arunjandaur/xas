from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import scipy.ndimage as im

def gaussian(x, amp, mu, sigma):
    """
    x - point along normal 
    amp - amplitude
    mu - mean
    sigma - standard deviation
    """
    #return amp*np.exp(-np.power(((x-mu)/sigma), 2)/2)
    #return amp/(sigma*np.sqrt(2*np.pi))*np.exp(-np.power(((x-mu)/sigma), 2)/2)
    return amp * stats.norm.pdf(x,loc=mu, scale= sigma)

#amp1 = lambda x, mu, sigma: gaussian(x, 1, mu, sigma)
mean_func = lambda x, a, b : a + b*x
func = lambda x, amp, mu_a, mu_b, sigma: gaussian(x, amp, mean_func(x,mu_a, mu_b) , sigma)

xdata =  np.linspace(0,1000,100000)
ydata = gaussian(xdata, 10000, 400, 30)
noisy_xdata = xdata + np.random.normal(size = len(xdata)) * (xdata[1])*0.05
noisy_ydata = ydata + 0.10*np.amax(ydata)*np.random.normal(size = len(ydata))

fitParams, fitCovariance = curve_fit(gaussian, noisy_xdata, noisy_ydata, p0= (1,1,1000)) 

plt.plot(xdata, ydata, label = "gaussian data" )
plt.plot(noisy_xdata, noisy_ydata, label = "noisy gaussian data")
plt.plot(xdata, gaussian(xdata, *fitParams), label = "fitted curve")
plt.legend()
plt.show()

