import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cmaps
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift


def reconstruct(n, meff, noffset, moffset):
    #m = meff/2.
    
    filename = 'gate0_numblock_{0}_meff_{1}_offsetn_{2}_offsetm_{3}_uc.npy'.format(n, meff, noffset, moffset)
    data = np.load(filename)
    
    #doppler = (np.arange(m)-m/2)*1./(6729.*m/660.)*1e3
    #delay = (np.arange(n)-n/2)*1./(8.0e6*1000./16384.)*1e3
    
    e = np.zeros([n,meff], complex)
    for i in range(len(data)/meff):
        e[i,:] = data[i*meff:(i+1)*meff,0]
        
    e = np.concatenate((e[n*(2/4.):n,meff*(3/4.):meff], e[n*(2/4.):n, 0:meff/4.]), axis=1)
    e = np.flipud(e)
    e = np.concatenate((np.zeros([e.shape[0], e.shape[1]]), e), axis=0)
    
    #y = delay[len(delay)/2:3*len(delay)/4]
    #x = doppler[:-1]
    
    return e

def correlation(data1, data2):
    data1[data1==0]=1e6
    data2[data2==0]=1e6
    temp = data1.real*data2.real + data1.imag*data2.imag
    temp2 = np.sqrt(np.abs(data1)**2*np.abs(data2)**2)
    return temp/temp2