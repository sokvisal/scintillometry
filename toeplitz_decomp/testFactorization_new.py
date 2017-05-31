
import numpy as np
import os,sys
from scipy.linalg import cholesky
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cmaps

matplotlib.rcParams.update({'font.size': 13})

##Specifications
n=21
m=21

offsetn = 0
offsetm = 0

pad = 1

##Load the processed data 
filename = "gate0_numblock_{}_meff_{}_offsetn_{}_offsetm_{}".format(n,m*2, offsetn, offsetm) 
file = filename + ".dat"
folder = "processedData"
print(folder,filename)


##Load the toepletz file generated from extract_real
toeplitz1 = np.memmap("{0}/{1}".format(folder,file), dtype='complex', mode='r', shape=(2*m,2*n*m), order='F')
print (toeplitz1.shape)

##Creating a square toeplitz matrix
toeplitz = np.zeros((2*m*n*(1 + pad), 2*n*m*(1 + pad)), complex)

##Gonna be messy, but using toeplitz1 to create toeplitz
toeplitz[:2*m, :2*n*m] = toeplitz1

for i in range(0,n*(1 + pad)):
    T_temp = toeplitz[:2*m,2*i*m:2*(i+1)*m]
    for j in range(0,n*(1 + pad) - i):
        k = j + i
        toeplitz[2*j*m:2*(j+1)*m,2*k*m:2*(k+1)*m] = T_temp
        toeplitz[2*k*m:2*(k+1)*m, 2*j*m:2*(j+1)*m] = np.conj(T_temp.T)


print ('begin plot')
##Plotting thr raw toeplitz matrix
fig = plt.figure(figsize=(12,6))
#plt.subplot(1,3,1)
#print '1'
#plt.imshow(np.abs(toeplitz))
#print '2'
#plt.colorbar()
#if not pad:
#        plt.title("Toeplitz matrix with n = {0}, m = {1}".format(n, m*4))
#else:
#        plt.title("Zero padded Toeplitz matrix with n = {0}, m = {1}".format(n, m*4))
##Adding graphing lines
#ax = fig.gca()
#ax.set_xticks(np.arange(0,2*n*m*(1 + pad),2*m))
#ax.set_yticks(np.arange(0,2*n*m*(1 + pad),2*m))
#ax.grid(True, which='both')
#if pad:
#    plt.plot([0, 2*2*n*m], [2*n*m, 2*m*n], '-k')
#    plt.plot([2*n*m, 2*m*n],[0, 2*2*n*m], '-k')
#print 'finish'
#
###Loading the factorized matrix 
folder = "gate0_numblock_{}_meff_{}_offsetn_{}_offsetm_{}".format(n, m*2, offsetn, offsetm)

L = np.zeros((2*m*n*(1 + pad),2*m*n*(1 + pad)), complex)
for i in range(n*(1 + pad)):
    for j in range(n*(1 + pad)): 
        path = "results/{0}/L_{1}-{2}.npy".format(folder, i,j)     
        
        if os.path.isfile(path):
            Ltemp = np.load(path)
            L[2*m*j: 2*m*(j + 1), 2*m*i:2*m*(i + 1)] = Ltemp
#            
##The  factorized matrix using Numpy's Cholesky
#npL = cholesky(toeplitz, True)

print ('begin plot 2')
plt.subplot(1,2,2)

##Graphical lines
ax = fig.gca()
ax.set_xticks(np.arange(0,2*n*m*(1 + pad),4*m))
ax.set_xticklabels(np.arange(0,2*n*m*(1 + pad),4*m) , rotation='vertical')
ax.set_yticks(np.arange(0,2*n*m*(1 + pad),4*m))
ax.grid(True, which='both')
if pad:
    plt.plot([0, 2*2*n*m], [2*n*m, 2*m*n], '-k')
    plt.plot([2*n*m, 2*m*n],[0, 2*2*n*m], '-k')

##Plotting the error difference between our raw and our results multiplied by its transposed conjugate
#toeplitz=np.where(toeplitz > 1e-6, toeplitz, 1000)
plt.imshow(np.abs(toeplitz - L.dot(np.conj(L.T))) )
#ax.tick_params(axis='x', colors='white')
#ax.tick_params(axis='y', colors='white')
cbar = plt.colorbar()
#cbar.ax.tick_params(axis='y', colors='white')
plt.title("Errors on the Toeplitz Matrix")


plt.subplot(1,2,1)
##Graphical lines
ax = fig.gca()
ax.set_xticks(np.arange(0,2*n*m*(1 + pad),4*m))
ax.set_xticklabels(np.arange(0,2*n*m*(1 + pad),4*m) , rotation='vertical')
ax.set_yticks(np.arange(0,2*n*m*(1 + pad),4*m))
ax.grid(True, which='both')
if pad:
        plt.plot([0, 2*2*n*m], [2*n*m, 2*m*n], '-k')
        plt.plot([2*n*m, 2*m*n],[0, 2*2*n*m], '-k')

##Plotting the factorized matrix
plt.title("Factorized Toeplitz Matrix")
plt.imshow(np.abs(L))
#ax.tick_params(axis='x', colors='white')
#ax.tick_params(axis='y', colors='white')
cbar = plt.colorbar()
#cbar.ax.tick_params(axis='y', colors='white')
plt.savefig('padd_toeplitz_err.png', transparent=False, bbox_inches='tight', dpi=400)
#plt.close
plt.show()

