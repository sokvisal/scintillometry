import sys
import numpy as np
import scipy as sp
from scipy import linalg
#from numpy import linalg as LA
#from toeplitz_decomp import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cmaps
import os	
import mmap
import re
from scipy.fftpack import fftshift, fft2, ifft2, ifftshift
import tarfile

matplotlib.rcParams.update({'font.size': 13})

#mm=mmap.mmap(-1,256)
#np.set_printoptions(precision=2, suppress=True, linewidth=5000)
#if len(sys.argv) < 8:
#    print "Usage: %s filename num_rows num_columns offsetn offsetm sizen sizem" % (sys.argv[0])
#    sys.exit(1)

filename = str(sys.argv[1])
num_rows=int(sys.argv[2]) # frequency
num_columns=int(sys.argv[3]) # time
offsetn=int(sys.argv[4]) # offset in freq
offsetm=int(sys.argv[5]) # offset in time
sizen=int(sys.argv[6]) # size of freq = n
sizem=int(sys.argv[7]) # size of freq = m
nump=sizen

## looking for the meff number in filename
#matchObj = re.search('freq_(\d*)',filename) 
#if matchObj:    
#    f=int(matchObj.group(1))
#    print f

if offsetn>num_rows or offsetm>num_columns or offsetn+sizen>num_rows or offsetm+sizem>num_columns:
	print ("Error sizes or offsets don't match")
	sys.exit(1)
 
#a = np.memmap(sys.argv[1], dtype='float32', mode='r', shape=(num_rows,num_columns),order='F')

# load normal array
#data = np.fromfile('C:\Users\Visal LeSok\Desktop\script\simulated_test\dynamic_spectrum_257_freq_00_f0326.5.bin',dtype=np.complex).reshape(-1,660)
a_f = np.load(sys.argv[1])
a = np.copy(a_f)

#a_f = np.fromfile(sys.argv[1],dtype=np.complex).reshape(-1,660)
#print a_f.shape
#a = a_f



##### change to 1 for padding #####
pad=1
pad2=1
debug=0

neff=sizen+sizen*pad
meff=sizem+sizem*pad

meff_f=meff+pad2*meff

a_input=np.zeros(shape=(neff,meff), dtype=complex)
print (a_input.shape)
a_input[:sizen,:sizem]=np.copy(a[offsetn:offsetn+sizen,offsetm:offsetm+sizem])
print (a_input)

plt.figure()
plt.imshow(a_input[offsetn:offsetn+sizen,offsetm:offsetm+sizem].real, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
plt.colorbar()
plt.title(r"Dynamic Spectrum: Input")
plt.ylabel(r"Time")
plt.xlabel(r"Freq")
plt.savefig('dyn_spec_input.png')
plt.close()

plt.figure()
plt.imshow(a_input.real, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
plt.colorbar()
plt.title(r"Dynamic Spectrum: First Padding")
plt.ylabel(r"Time")
plt.xlabel(r"Freq")
plt.savefig('dyn_spec_1padding.png')
plt.close()

##### specifying file directories #####
newdir = "gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s" %(str(sizen),str(meff_f/2),str(offsetn),str(offsetm))
if not os.path.exists("processedData/"+newdir):	
	os.makedirs("processedData/"+newdir)
filen="processedData/"+newdir+"/"+newdir+"_dynamic.npy"
np.save(filen,a_input)

const=int(pad2*meff/2)

##### ensuring positive definite matrix #####
norm = a.shape[0]*a.shape[1]
a_input=np.sqrt(a_input)
if debug:
	print (a_input,"after sqrt")
a_input[:sizen,:sizem]=np.fft.fft2(a_input,s=(sizen,sizem))
if debug:
	print (a_input,"after first fft")
c = a_input

plt.figure(figsize=(12,6))
print (int(round(sizem/2.)))
a_input[0:sizen, meff-int(round(sizem/2.)):meff] =  a_input[0:sizen, int(sizem/2 + 0.5):sizem]
a_input[0:sizen, int(round(sizem/2.)):sizem] = 0+0j
plt.subplot(1,2,1)
plt.imshow((a_input).real, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')

a_input[neff-int(round(sizen/2.)):neff,0:meff] = a_input[int(sizen/2+0.5):sizen, 0:meff]
a_input[int(round(sizen/2.)):sizen, 0:meff] = 0+0j
plt.subplot(1,2,2)
plt.imshow((a_input).real, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
plt.savefig('2.png') 
if debug:
	print (a_input,"after shift")
print ('after transformation')

## inverse Fourier transform 
a_input=np.fft.ifft2(a_input,s=(neff,meff))
if debug:
	print (a_input,"after inverse fft")
a_input=np.power(np.abs(a_input),2)
if debug:
	print (a_input,"after abs^2")
a_input=np.fft.fft2(a_input,s=(neff,meff))
if debug:
	print (a_input,"after third fft")
###############################################

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow((a_input).real, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
plt.colorbar()
plt.title(r"Conjugate Spectrum: Rooting and Squaring")
plt.ylabel(r"Lag")
plt.xlabel(r"Dopp Freq")

plt.subplot(1,2,2)
plt.imshow(np.fft.ifft2(a_input).real, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
plt.colorbar()
plt.title(r"Dynamic Spectrum: Rooting and Squaring")
plt.ylabel(r"Time")
plt.xlabel(r"Freq")
plt.savefig('dyn_spec_2rooting.png')
plt.close()

path="processedData/gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s" %(str(sizen),str(meff_f/2),str(offsetn),str(offsetm))
mkdir="mkdir "+path
    
epsilon=np.identity(int(meff_f/2))  *10e-8
input_f=np.zeros(shape=(int(meff_f/2), int(sizen*meff_f/2)), dtype=complex)


#################### making blocked toeplitz elements #########################################
if neff == 1:
    neff += 1
for j in np.arange(0,int(neff/2)):
    rows = np.append(a_input[j,:meff-const], np.zeros(pad2*meff*0+const))
    cols = np.append(np.append(a_input[j,0], a_input[j,const+1:][::-1]), np.zeros(pad2*meff*0+const))
    input_f[0:int(meff_f/2),j*int(meff_f/2):(j+1)*int(meff_f/2)] = sp.linalg.toeplitz(cols,rows)
    if j==0:
        input_f[0:int(meff_f/2),j*int(meff_f/2):(j+1)*int(meff_f/2)] = sp.linalg.toeplitz(np.conj(np.append(a_input[j,:meff-const],np.zeros(pad2*meff*0+const))))+epsilon
print ("##########################")
if neff == 1:
    neff -= 1
        
tar = tarfile.open('data.tar.gz','w:gz')
for rank in np.arange(0,nump):
    size_node_temp=(sizen//nump)*int(meff_f/2)
    size_node=size_node_temp
    if rank==nump-1:
        size_node = (sizen//nump)*int(meff_f/2)+ (sizen%nump)*int(meff_f/2)
    start = rank*size_node_temp
    file_name=path+'/'+str(rank)+".npy"
    np.save(file_name, np.conj(input_f[:,start:start+size_node].T))
#    tar.add(file_name)
#    os.remove(file_name)
tar.close()
    
# dat file for toeplitz matrix
output_file="processedData/gate0_numblock_%s_meff_%s_offsetn_%s_offsetm_%s.dat" %(str(sizen),str(meff_f/2),str(offsetn),str(offsetm))
output = np.memmap(output_file, dtype='complex', mode='w+', shape=(int(meff_f/2), sizen*int(meff_f/2)),order='F')
output[:,:]=input_f[:meff_f,:]

plt.figure()
plt.imshow(output.real, aspect='auto', interpolation='nearest', origin='lower', cmap='viridis')
plt.colorbar()
plt.savefig('toep.png')
plt.close()

del output


#mm.close()
if debug:
    pad=1
    u=toeplitz_blockschur(input_f[:neff/2*meff_f,:neff/2*meff_f],meff_f,pad)
    print (u[:,(neff/2)*(pad+1)*(meff_f)-meff_f/2-1:(neff/2)*(pad+1)*(meff_f)-meff_f/2])

    
