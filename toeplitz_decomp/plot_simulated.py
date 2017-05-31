import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cmaps
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift
import re
import os.path
import glob
from toeplitz_scint import *

matplotlib.rcParams.update({'font.size': 13})

n = str(sys.argv[1])
m=int(sys.argv[2]) # frequency
offsetn=int(sys.argv[3]) # time
offsetm=int(sys.argv[4]) # offset in freq

n = 21
m = 21
offsetn = 0
offsetm = 0

if m%2 == 0:
    xticks = np.arange(-m/2+0.5, m/2+1, 2)
    xtickslabels= np.arange(-m/2, m/2+1, 2)
    yticks = np.arange(-n/2+0.5, n/2+1, 2)
    ytickslabels= np.arange(-n/2, n/2+1, 2)
else:
    xticks = np.arange(-m/2+0.5, m/2+1, 2)
    xtickslabels= np.arange(-m/2+1, m/2+1, 2)
    yticks = np.arange(-n/2+0.5, n/2+1, 2)
    ytickslabels= np.arange(-n/2+1, n/2+1, 2)
extent = [-n/2, n/2, -m/2, m/2]


retrieved = (reconstruct(n, m*2, offsetn, offsetm))

plt.figure(figsize=(18,12))
plt.subplot(2,3,1)
im=plt.imshow(retrieved.real, interpolation='nearest', origin='lower', 
              cmap='viridis', extent = extent) 
plt.title(r"Retrieved RE $\tilde{E}(\tau, f_\nu)$", color='white')
plt.xlabel(r"$f_\nu$")
plt.ylabel(r"$\tau$")
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)

#ax.tick_params(axis='x', colors='white')
#ax.tick_params(axis='y', colors='white')
cbar = plt.colorbar()
#cbar.ax.tick_params(axis='y', colors='white')

plt.subplot(2,3,2)
im=plt.imshow(retrieved.imag, interpolation='nearest', origin='lower', 
              cmap='viridis',  extent = extent)
plt.title(r"Retrieved IM $\tilde{E}(t, \nu)$", color='white')
plt.xlabel(r"$f_\nu$")
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)

#ax.tick_params(axis='x', colors='white')
#ax.tick_params(axis='y', colors='white')
cbar = plt.colorbar()
#cbar.ax.tick_params(axis='y', colors='white')

e = np.load('e_field_fourier_nonoise_tfd.npy')
         
plt.subplot(2,3,4)
plt.imshow(e.real, extent = extent, origin='lower',cmap='viridis', interpolation='nearest')
plt.title(r"Simulated RE $\tilde{E}(\tau,f_D)$", color='white')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)

#ax.tick_params(axis='x', colors='white')
#ax.tick_params(axis='y', colors='white')
cbar = plt.colorbar()
#cbar.ax.tick_params(axis='y', colors='white')

plt.subplot(2,3,5)
plt.imshow(e.imag, extent = extent, origin='lower',cmap='viridis', interpolation='nearest')
plt.title(r"Simulated RE $\tilde{E}(\tau,f_D)$", color='white')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)

#ax.tick_params(axis='x', colors='white')
#ax.tick_params(axis='y', colors='white')
cbar = plt.colorbar()
#cbar.ax.tick_params(axis='y', colors='white')

retrieved = ifft2(ifftshift(retrieved))*retrieved.size
e = ifft2(ifftshift(e))*e.size
         
plt.subplot(2,3,3)
plt.imshow(np.angle(retrieved), extent = extent, origin='lower',cmap='viridis', interpolation='nearest')
plt.title(r"Retrieved Phase ${E}(f,\nu)$", color='white')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)

#ax.tick_params(axis='x', colors='white')
#ax.tick_params(axis='y', colors='white')
cbar = plt.colorbar()
#cbar.ax.tick_params(axis='y', colors='white')

plt.subplot(2,3,6)
plt.imshow(np.angle(e), extent = extent, origin='lower',cmap='viridis', interpolation='nearest')
plt.title(r"Simulated Phase ${E}(f,\nu)$", color='white')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)

#ax.tick_params(axis='x', colors='white')
#ax.tick_params(axis='y', colors='white')
cbar = plt.colorbar()
#cbar.ax.tick_params(axis='y', colors='white')

plt.savefig('retreieved_offset_freq_domain.png', transparent=False, bbox_inches='tight')