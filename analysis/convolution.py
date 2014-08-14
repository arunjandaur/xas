from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import scipy.ndimage as im
import scipy

def gaussian(x, amp, mu, sigma):
    """
    x - point along normal 
    amp - amplitude
    mu - mean
    sigma - standard deviation
    """
    #return amp*np.exp(-np.power(((x-mu)/sigma), 2)/2)
    #return amp/(sigma*np.sqrt(2*np.pi))*np.exp(-np.power(((x-mu)/sigma), 2)/2)
    return amp *stats.norm.pdf(x,loc=mu, scale= sigma)

x = np.linspace(0,100,10000)
y = gaussian(x, 10, 45,15)

filtered = im.gaussian_filter1d(y, 3,order = 0)
filt_deriv = im.filters.prewitt(filtered)
f2d = im.gaussian_filter1d(y, 3,order = 2)
#zero_cross = np.where(filtered
plt.subplot(211)
plt.plot(x,y,label= "original")
plt.plot(x,filt_deriv, label= "filtered")
plt.legend()
plt.title("gaussian data")
plt.subplot(212)
plt.plot(x,f2d, label= "filtered 2nd derivative")
plt.title("second derivative")
plt.legend()
plt.show()

