import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cmaps
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift


def reconstruct(n, meff, noffset, moffset):
    #m = meff/2.
    
    filename = 'results/gate0_numblock_{0}_meff_{1}_offsetn_{2}_offsetm_{3}_uc.npy'.format(n, meff, noffset, moffset)
    data = np.load(filename)
    
    #doppler = (np.arange(m)-m/2)*1./(6729.*m/660.)*1e3
    #delay = (np.arange(n)-n/2)*1./(8.0e6*1000./16384.)*1e3
    
    e = np.zeros([n,meff], complex)
    for i in range(len(data)/meff):
        e[i,:] = data[i*meff:(i+1)*meff,0]
        
    e = np.concatenate((e[int(n*2/4.):n,int(meff*3/4.+0.5):meff], e[int(n*2/4.):n, 0:int(meff/4.+0.5)]), axis=1)
    print (int(n*2/4.), n,int(meff*3/4.), meff)
    print (int(n*2/4.),n, 0,int(meff/4.))
    e = np.flipud(e)
    if n%2 != 0:
        e = np.concatenate((np.zeros([e.shape[0]-1, e.shape[1]]), e), axis=0)
    else:
        e = np.concatenate((np.zeros([e.shape[0], e.shape[1]]), e), axis=0)
    print (e.shape)
    
    #y = delay[len(delay)/2:3*len(delay)/4]
    #x = doppler[:-1]
    
    return e

def correlation(data1, data2):
    data1[data1==0]=1e6
    data2[data2==0]=1e6
    temp = data1.real*data2.real + data1.imag*data2.imag
    temp2 = np.sqrt(np.abs(data1)**2*np.abs(data2)**2)
    return temp/temp2