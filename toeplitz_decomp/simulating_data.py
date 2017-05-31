import sys
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
#import matplotlib.cm as cmaps
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift
import os.path
from scipy import signal
from scipy import ndimage

matplotlib.rcParams.update({'font.size': 13})

# defining electric field in fourier domain
e = np.zeros((61,21),dtype=np.complex_)

e[30,10]=200.+0j
e[32,8]=200+0j
#e[34,13]=200+0j

#e = np.zeros((1,21),dtype=np.complex_)
#
#e[0,10]=200.+0j
#e[0,14]=200+0j

e = np.zeros((21,21),dtype=np.complex_)
for i in range(21):
    for j in range(1):
        im = np.random.uniform(-0.1,0.1)
        e[i,j] = np.random.uniform(-0.1,0.1)+im*1j
e[10,10]= 200.+0j
e[12,8]= 200.+50.j
e[16,13]= 200.+50.j
#e[0,0]=200.+5j
#e[14,13]=200.+5j
#e[13,13]=200+0j

#e =  np.zeros((41,21))
#for i in range(41):
#    for j in range(21):
#        e[i,j] = np.random.uniform(0,0.1)
#
#e[20,10]=2.
#e[24,8]=1.5

#e[24,12]=1.5+0.j
#e[28,7]=1.+0.j
#e[28,13]=1.+0.j
#e = 10**e

if e.shape[0]%2 == 0:
    n = e.shape[0]
    m = e.shape[1]
    xticks = np.arange(-m/2+0.5, m/2+1, 2)
    xtickslabels= np.arange(-m/2, m/2+1, 2)
    yticks = np.arange(-n/2+0.5, n/2+1, 2)
    ytickslabels= np.arange(-n/2, n/2+1, 2)
else:
    n = e.shape[0]
    m = e.shape[1]
    xticks = np.arange(-m/2+0.5, m/2+1, 2)
    xtickslabels= np.arange(-m/2+1, m/2+1, 2)
    yticks = np.arange(-n/2+0.5, n/2+1, 2)
    ytickslabels= np.arange(-n/2+1, n/2+1, 2)
extent = [-n/2, n/2, -m/2, m/2]

np.save('e_field_fourier_nonoise_tfd.npy', e)

# defining the grid of the plot
gs = plt.GridSpec(4, 6, wspace=0.4, hspace=0.4)
fig = plt.figure(figsize=(18, 12))

fig.add_subplot(gs[:2, :2])
plt.imshow(e.real, aspect='equal', extent=extent, interpolation='nearest',origin='lower', cmap='viridis')
plt.title(r'Simulated $\tilde{E}(\tau,f_D)$')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)
plt.colorbar()

# electric field in frequency and time
e_fft = ifft2(ifftshift(e))*e.shape[0]*e.shape[1]

fig.add_subplot(gs[:2, 2:4])
plt.imshow(fftshift(e_fft).real, extent=extent, aspect='equal', interpolation='nearest',origin='lower', cmap='viridis')
plt.title(r'Simulated RE ${E}(f,t)$')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)
#plt.xlabel(r'Time')
plt.colorbar()

fig.add_subplot(gs[:2, 4:6])
plt.imshow(fftshift(e_fft).imag, aspect='equal', extent=extent, interpolation='nearest',origin='lower', cmap='viridis')
plt.title(r'Simulated IM ${E}(f,t)$')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)
#plt.xlabel(r'Time')
plt.colorbar()

i = np.abs(((e_fft)))**2
print ('stop', i.min(), i.max())
np.save('simulated_dyn_spec2.npy', i)

fig.add_subplot(gs[2:4, :2])
plt.imshow(i.real, aspect='equal', extent=extent, interpolation='nearest',origin='lower', cmap='viridis')
plt.title(r'Simulated ${E}(f,t)^2$')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)
#plt.xlabel(r'Time')
plt.colorbar()

i = (fft2(i))
#print 'intensity field comparison', fftshift(i)[24,8].real/fftshift(i)[20,10]

fig.add_subplot(gs[2:4, 2:4])
plt.imshow(fftshift(i).real, aspect='equal', extent=extent, interpolation='nearest',origin='lower', cmap='viridis')
plt.title(r'Simulated RE $\tilde{I}(\tau,f_D)$')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)
#plt.xlabel(r'Doppler Freq')
plt.colorbar()

fig.add_subplot(gs[2:4, 4:6])
plt.imshow(fftshift(i).imag, aspect='equal', extent=extent, interpolation='nearest',origin='lower', cmap='viridis')
plt.title(r'Simulated IM $\tilde{I}(\tau,f_D)$')
ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xtickslabels)
ax.set_yticks(yticks)
ax.set_yticklabels(ytickslabels)
#plt.xlabel(r'Doppler Freq')
plt.colorbar()
plt.savefig('simulated_offset_tau_domain.png')
plt.show()
