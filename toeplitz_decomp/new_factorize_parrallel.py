##Give Credits Later

import numpy as np
import scipy as sp
from numpy.linalg import cholesky, inv
from numpy import triu
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir + "/Exceptions")

from ToeplitzFactorizorExceptions import *

from mpi4py import MPI

from GeneratorBlocks import Blocks
from GeneratorBlock import Block

from time import time

MAXTIME = int(60*60*23.5) #23.5 hours in seconds
timePerLoop = []
startTime = time()


SEQ, WY1, WY2, YTY1, YTY2 = "seq", "wy1", "wy2", "yty1", "yty2"
class ToeplitzFactorizor:
    
    def __init__(self, folder, n,m, pad, detailedSave = False):
        self.comm = MPI.COMM_WORLD
        size  = self.comm.Get_size()
        self.size = size
        self.rank = self.comm.Get_rank()
        self.n = n
        self.m = m
        self.pad = pad
        self.folder = folder
        self.blocks = Blocks()
        
        self.detailedSave = detailedSave
        self.numOfBlocks = n*(1 + pad)
        
        kCheckpoint = 0 #0 = no checkpoint
        
        if os.path.exists("processedData/" + folder + "/checkpoint"):
            for k in range(n*(1 + self.pad) - 1, 0, -1):
                if os.path.exists("processedData/{0}/checkpoint/{1}/".format(folder, k)):
                    path, dirs, files = os.walk("processedData/{0}/checkpoint/{1}/".format(folder, k)).next()
                    file_count = len(files)
                    if file_count == 2*self.numOfBlocks:
                        kCheckpoint = k 
                        if self.rank == 0: print ("Using Checkpoint #{0}".format(k))
                        break
        else:
            if self.rank == 0:
                os.makedirs("processedData/{0}/checkpoint/".format(folder))
        self.kCheckpoint = kCheckpoint
        if not os.path.exists("results"):
            if self.rank == 0:
                os.makedirs("results")
        if not os.path.exists("results/{0}".format(folder)):
            if self.rank == 0:
                os.makedirs("results/{0}".format(folder))   

        if self.rank==0:
            if not os.path.exists("results/{0}".format(folder + "_uc.npy")):
                uc = np.zeros((m*n,1), dtype=complex)
                np.save("results/{0}".format(folder + "_uc.npy"), uc)
        ## So that the creation of files and directories are complete before the rest of the nodes continue        
        initDone=False
        initDone = self.comm.bcast(initDone, root=0)
        
        
    def addBlock(self, rank):
        folder = self.folder
        b = Block(rank)
        k = self.kCheckpoint
        if k!= 0:
            A1 = np.load("processedData/{0}/checkpoint/{1}/{2}A1.npy".format(folder, k, rank))
            A2 = np.load("processedData/{0}/checkpoint/{1}/{2}A2.npy".format(folder, k, rank))
            b.setA1(A1)
            b.setA2(A2)
        else:
            if rank >= self.n:
                m = self.m
                b.createA(np.zeros((m,m), complex))
                
            else:
                T = np.load("processedData/{0}/{1}.npy".format(folder,rank))
                b.setT(T)
        b.setName("results/{0}_uc.npy".format(folder))
        self.blocks.addBlock(b)     
        return 

    def fact(self, method, p):
        if method not in np.array([SEQ, WY1, WY2, YTY1, YTY2]):
            raise InvalidMethodException(method)
        if p < 1 and method != SEQ:
            raise InvalidPException(p)
        
        
        pad = self.pad
        m = self.m
        n = self.n
        
        folder = self.folder
        
        if self.kCheckpoint==0:
            self.__setup_gen()

            for b in self.blocks:
                if not pad and b.rank == n*(1 + pad) - 1:
                    b.updateuc(b.rank)
                    
        if (self.detailedSave):
            for b in self.blocks:        
                np.save("results/{0}/L_{1}-{2}.npy".format(folder, 0, b.rank), b.getA1())
        
        for k in range(self.kCheckpoint + 1,n*(1 + pad)):
            self.k = k
            if self.rank == 1:
                print ("Loop {0}".format(k))
            ##Build generator at step k [A1(:e1, :) A2(s2:e2, :)]
            s1, e1, s2, e2 = self.__set_curr_gen(k, n)
            if method==SEQ:
                self.__seq_reduc(s1, e1, s2, e2)
            else:
                self.__block_reduc(s1, e1, s2, e2, m, p, method, k)
            
                
            ##Save results immediately if we reached the end of the loop
            for b in self.blocks:
                if b.rank <=e1 and b.rank + k == n*(1 + pad) - 1:
                    b.updateuc(k%self.n)
                if b.rank <= e1 and self.detailedSave:
                    np.save("results/{0}/L_{1}-{2}.npy".format(folder, k, b.rank + k), b.getA1())
                
            ##CheckPoint
            saveCheckpoint = False
            if self.rank==0:
                timePerLoop.append(time() - sum(timePerLoop) - startTime)
                
                elapsedTime = time() - startTime
                if elapsedTime + max(timePerLoop) >= MAXTIME: ##Max instead of np.mean, just to be safe
                    print ("Saving Checkpoint #{0}".format(k))
                    if not os.path.exists("processedData/{0}/checkpoint/{1}/".format(folder, k)):
                        try:
                            os.makedirs("processedData/{0}/checkpoint/{1}/".format(folder, k))
                        except: pass
                    saveCheckpoint = True
            saveCheckpoint = self.comm.bcast(saveCheckpoint, root=0)
            
            if saveCheckpoint:
                for b in self.blocks:
                    ##Creating Checkpoint
                    A1 = np.save("processedData/{0}/checkpoint/{1}/{2}A1.npy".format(folder, k, b.rank), b.getA1())
                    A2 = np.save("processedData/{0}/checkpoint/{1}/{2}A2.npy".format(folder, k, b.rank), b.getA2())
                exit()
                    
            

    ##Private Methods
    def __setup_gen(self):
        n = self.n
        m = self.m
        pad = self.pad
        A1 = np.zeros((m, m),complex)
        A2 = np.zeros((m, m), complex)
        cinv = None
        
        ##The root rank will compute the cholesky decomposition
        if self.blocks.hasRank(0) :
#            print self.blocks.getBlock(0).getT(), 'getBlock'
            c = cholesky(self.blocks.getBlock(0).getT())
            c = np.conj(c.T)
            cinv = inv(c)
#            print cinv, 'cinv'
        cinv = self.comm.bcast(cinv, root=0)
        for b in self.blocks:
            if b.rank < self.n:
                b.createA(b.getT().dot(cinv))
#                print 'A1', A1.shape, b.rank
#                print 'A2', A2.shape, b.rank
            
        ##We are done with T. We shouldn't ever have a reason to use it again
        for b in self.blocks:
            b.deleteT()
        
        return A1, A2

    def __set_curr_gen(self, k, n):
        s1 = 0
        e1 = min(n, (n*(1 + self.pad) - k)) -1
        s2 = k
        e2 = e1 + s2
        
        for b in self.blocks:
            if s1 <= b.rank <=e1:
                b.setWork1(b.rank + k)
            else:
                b.setWork1(None)
            if e2 >= b.rank >= s2:
                b.setWork2(b.rank - k)
            else:
                b.setWork2(None)
        return s1, e1, s2, e2
    
    def __temp_Comm(self, k, n, b):
        s1 = 0
        e1 = min(n, (n*(1 + self.pad) - k)) -1
        s2 = k
        e2 = e1 + s2
        
        N = self.size # number of processes
        n = np.arange(0,N)
        
        # find processes that are needed
        temp = np.where( np.logical_and( n >=  s1, n <= e1) )
        temp2 = np.where( np.logical_and( n >=  s2, n <= e2, n==0) )
        union = np.union1d(temp, temp2)
#        if self.rank == 0: print union
        
        exclusion = np.setxor1d(n, union) # find processes that are not needed
        newrank = np.arange(0, union.size)
        
        # making a new sub communicator between the processes that are needed
        group = self.comm.Get_group()
        newgroup = group.Excl(exclusion)
        newcomm = self.comm.Create(newgroup)
        
        # renaming new comm size and ranking scheme
        if self.rank in exclusion:
            assert newcomm == MPI.COMM_NULL
        else:
            assert newcomm.size == self.size-exclusion.size
#            print newrank[np.where(self.rank == union )], self.rank
            assert newcomm.rank == newrank[np.where(self.rank == union )][0]
        
        if self.rank not in exclusion:
            newcomm.Bcast(b.getTemp(), root=0) 
            
        group.Free(); newgroup.Free()
        if newcomm: newcomm.Free()
        return union

    def __block_reduc(self, s1, e1, s2, e2, m, p, method, k):
        n = self.n
       
        X2_list = np.zeros((m, m+1), complex)
        for sb1 in range (0, m, p):
            
            for b in self.blocks:
                b.setWork(None, None)
                if b.rank==0: b.setWork1(s2)
                if b.rank==s2: b.setWork2(0)
            #print k, b.rank, b.getWork1(), b.getWork2()
        
            sb2 = s2*m + sb1
            eb1 = min(sb1 + p, m) #next j
            eb2 = s2*m + eb1
            u1 = eb1
            u2 = eb2
            p_eff = min(p, m - sb1)
            
            temp =  np.zeros((p_eff, m+1), complex)
            if method == WY1 or method == WY2:
                S = np.array([np.zeros((m,p)),np.zeros((m,p))], complex)
            elif method == YTY1 or YTY2:
                S = np.zeros((p, p), complex)
            
            #b.createTemp(np.zeros((m+1), complex))
            for j in range(0, p_eff):
                j1 = sb1 + j
                j2 = sb2 + j
                data= self.__house_vec(j1, s2, j, b) ##s2 or sb2?
  
                temp[j] = data
                X2 = data[:self.m]
                beta = data[-1]
                self.__seq_update(X2, beta, eb1, eb2, s2, j1, m, n)

            XX2 = temp[:,:m]
            if b.rank == s2 or b.rank == 0:
                S = self.__aggregate(S, XX2, beta, m, j, p_eff, method)
                self.__set_curr_gen(s2, n) ## Updates work
                self.__new_block_update(XX2, sb1, eb1, u1, e1, s2,  sb2, eb2, u2, e2, S, m, p_eff)
            X2_list[sb1:sb1+p_eff,:] = temp
        
        b.createTemp(np.zeros((m, m+1), complex))
        b.setTemp(X2_list)
        if b.getCond()[0]:
            pass
        else:
            self.comm.Bcast(b.getTemp(), root=s2)
            
        temp = b.getTemp()
        for sb1 in range (0, m, p):
            
            for b in self.blocks:
                b.setWork(None, None)
                if b.rank==0: b.setWork1(s2)
                if b.rank==s2: b.setWork2(0)
        
            sb2 = s2*m + sb1
            eb1 = min(sb1 + p, m) #next j
            eb2 = s2*m + eb1
            u1 = eb1
            u2 = eb2
            p_eff = min(p, m - sb1)
            
            temp2 = temp[sb1:sb1+p_eff,:]
            XX2 = temp2[:,:m]
            beta = temp2[-1,-1]
            if method == YTY1 or YTY2:
                S = np.zeros((p, p), complex)
            if b.rank != s2 or b.rank != 0:
                S = self.__aggregate(S, XX2, beta, m, j, p_eff, method)
                self.__set_curr_gen(s2, n) ## Updates work
                self.__block_update(XX2, sb1, eb1, u1, e1, s2,  sb2, eb2, u2, e2, S, method)
        return
    
    def __new_block_update(self, X2, sb1, eb1, u1, e1,s2, sb2, eb2, u2, e2, S, m, p_eff):
        for b in self.blocks:
            num = self.numOfBlocks
            invT = S
            if b.rank == s2:
                s = u1
                A2 = b.getA2()
                B2 = A2[s:, :m].dot(np.conj(X2[:p_eff, :m]).T)
                self.comm.Send(B2, dest=b.getWork2()%self.size, tag=3*num + b.getWork2())
                del A2
                
            if b.rank == 0:
                s=u1
                
                A1 = b.getA1()
                B1 = A1[s:, sb1:eb1]
                
                B2 = np.empty((m - s, p_eff), complex)
                self.comm.Recv(B2, source=b.getWork1()%self.size, tag=3*num + b.rank)  
                M = B1 - B2
                M = M.dot(inv(invT[:p_eff,:p_eff]))
                
                #print 's2', s, M.shape
                self.comm.Send(M, dest=b.getWork1()%self.size, tag=4*num + b.rank)
                A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M
                del A1   
    
            if b.rank == s2:
                s = u1
                M = np.empty((m - s, p_eff), complex)
                self.comm.Recv(M, source=b.getWork2()%self.size, tag=4*num + b.getWork2())
                
                A2 = b.getA2()
                A2[s:, :m] = A2[s:,:m] + M.dot(X2)
                del A2 
        return 
    
    def __block_update(self, X2, sb1, eb1, u1, e1,s2, sb2, eb2, u2, e2, S, method):
        def yty2():
            invT = S
            for b in self.blocks:
#                print b.rank
                #print s2,b.rank, b.work1, b.work2
                if b.work2 == None: 
                    continue
#                print s2,b.rank, b.work1, b.work2, 'a'
                s = 0 
                if b.rank == s2:
                    continue
#                    print u1, 'u1'
                A2 = b.getA2()
                B2 = A2[s:, :m].dot(np.conj(X2[:p_eff, :m]).T)
#                B2 = A2[s:, :m].dot(np.conj(X2[:, :m]).T)
                #print 's1', s, A2[s:, :m].shape
                self.comm.Send(B2, dest=b.getWork2()%self.size, tag=3*num + b.getWork2())
                
                del A2
                
            for b in self.blocks:
                if b.work1 == None: continue
                s = 0
#                print s2,b.rank, b.work1, b.work2,  'b'
                if b.rank == 0:
                    continue
                
                A1 = b.getA1()
                B1 = A1[s:, sb1:eb1]
#                B1 = A1[s:, :]
                
                B2 = np.empty((m - s, p_eff), complex)
#                B2 = np.empty((m - s, m), complex)
                self.comm.Recv(B2, source=b.getWork1()%self.size, tag=3*num + b.rank)  
                M = B1 - B2
#                print S, b.rank
                M = M.dot(inv(invT[:p_eff,:p_eff]))
#                M = M.dot(inv(invT[:,:]))
                
                #print 's2', s, M.shape
                self.comm.Send(M, dest=b.getWork1()%self.size, tag=4*num + b.rank)
                A1[s:, sb1:eb1] = A1[s:, sb1:eb1] + M
#                A1[s:, :] = A1[s:, :] + M
                del A1   
            for b in self.blocks:
                if b.work2 == None: 
                    continue
                s = 0 
                if b.rank == s2:
                    continue
                M = np.empty((m - s, p_eff), complex)
#                M = np.empty((m - s, m), complex)
                self.comm.Recv(M, source=b.getWork2()%self.size, tag=4*num + b.getWork2())
                
                A2 = b.getA2()
                f = time()
                A2[s:, :m] = A2[s:,:m] + M.dot(X2)
                g = time()
                #print 's3', s, M.shape, g-f
                del A2 
            return 
        
        
        m = self.m
        n = self.n
        nru = e1*m - u1
        p_eff = eb1 - sb1 
        num = self.numOfBlocks
        
        if method == WY1:
            return wy1()
        elif method == WY2:
            return wy2()
        elif method ==YTY1:
            return yty1()
        elif method == YTY2:
            return yty2()
        
    def __aggregate(self,S,  X2, beta, m, j, p_eff, method):
        invT = S
        invT[:p_eff,:p_eff] = triu(X2[: p_eff, :m].dot(np.conj(X2)[: p_eff, :m].T))
        for jj in range(p_eff):
            invT[jj,jj] = (invT[jj,jj] - 1.)/2.
        return invT         

#    def __aggregate(self,S,  X2, beta, p, j, j1, j2, p_eff, method):
#        def yty2():
#            invT = S
##            print S.shape, p_eff, 's'
#            #log("old invT = " + str(invT))
#            if j == p_eff - 1:
#                invT[:p_eff,:p_eff] = triu(X2[: p_eff, :m].dot(np.conj(X2)[: p_eff, :m].T))
#                for jj in range(p_eff):
#                    invT[jj,jj] = (invT[jj,jj] - 1.)/2.
#            return invT           
#        m = self.m
#        n = self.n
#        sb1 = j1 - j
#        sb2 = j2 - j
#        v = np.zeros(m*(n + 1), complex) 
#        #log("sb1, sb2 = {0}, {1}".format(sb1, sb2)) 
#        if method == WY1:
#            return wy1()
#        if method == WY2:
#            return wy2()
#        if method == YTY1:
#            return yty1()
#        if method == YTY2:
#            return yty2()

    def __seq_reduc(self, s1, e1, s2, e2):
        n = self.n
        m = self.m
        for j in range (0, self.m):
            X2, beta = self.__house_vec(j, s2)
            
            self.__seq_update(X2, beta, e1*m, e2*m, s2, j, m, n)

    def __seq_update(self,X2, beta, e1, e2, s2, j, m, n):
        #X2 = np.array([X2])
        u = j + 1
        num = self.numOfBlocks
        
        nru = e1*m - (s2*m + j + 1)  
        for b in self.blocks:
            if b.work2 == None: 
                continue
#            print s2,b.rank, b.work1, b.work2,  'a'
            B1 = np.dot(b.getA2(), np.conj(X2.T))
            start = 0
            end = m
            if b.rank == s2:
                start = u
            if b.rank == e2/m:
                end = e2 % m or m
            B1 = B1[start:end]
            self.comm.Send(B1, dest=b.getWork2()%self.size, tag=4*num + b.getWork2())

        
        for b in self.blocks:
            if b.work1 == None:
                continue 
#            print s2,b.rank, b.work1, b.work2,  'b'
            start = 0
            end = m
            if b.rank == 0:
                start = u
            if b.rank == e1/m:
                end = e1 % m or m
            B1 = np.empty(end-start, complex)
            
            self.comm.Recv(B1, source=b.getWork1()%self.size, tag=4*num + b.rank)
            A1 = b.getA1()
            B2 = A1[start:end, j]
                
            v = B2 - B1
            self.comm.Send(v, (b.getWork1())%self.size, 5*num + b.getWork1())
            A1[start:end,j] -= beta*v
            del A1

        for b in self.blocks:
            if b.work2 == None: 
                continue
            start = 0
            end = m
            if b.rank == s2:
                start = u
            if b.rank == e2/m :
                end = e2 % m or m
            v = np.empty(end-start,complex)
            self.comm.Recv(v, source=b.getWork2()%self.size, tag=5*num + b.rank)
            A2 = b.getA2()
            A2[start:end,:] -= beta*v[np.newaxis].T.dot(np.array([X2[:]]))
            #A2[start:end,:] -= beta*v.dot(np.conj(X2).T)
            del A2
        
    def __house_vec(self, j, s2, j_count, b):
        isZero = np.array([0])
        b.setFalse(isZero)
#        print b.getCond()
        
        X2 = np.zeros(self.m, complex)
        data = np.zeros(self.m+1, complex)
        beta = 0
        blocks = self.blocks
        n = self.n
        num = self.numOfBlocks
        
        if blocks.hasRank(s2):
            A2 = blocks.getBlock(s2).getA2()
            if np.all(np.abs(A2[j, :]) < 1e-13):
                isZero=np.array([1])
                b.setTrue(isZero)
                self.comm.Bcast(b.getCond(), root=s2%self.size)
            del A2
        
        #isZero = self.comm.bcast(isZero, root=s2%self.size)
#        self.comm.Bcast(b.getCond(), root=s2%self.size)
        if b.getCond()[0]:
            print (isZero)
            data[:self.m] = X2
            data[-1] = beta  
            b.setTemp(data)
            return data
        
        if blocks.hasRank(s2):
            A2 = blocks.getBlock(s2).getA2()
            sigma = A2[j, :].dot(np.conj(A2[j,:]))
            self.comm.send(sigma, dest=0, tag=2*num + s2)
            
            z = self.comm.recv(source=0, tag=3*num + s2)
            beta = self.comm.recv(source=0, tag=4*num + s2)

            X2 = A2[j,:]/z
            A2[j, :] = X2
 
           #print X2.shape, beta, 'main'
            data[:self.m] = X2
            data[-1] = beta  
            b.setTemp(data)
            self.comm.send(data, dest=0, tag=5*num + s2)
            del A2
            
        if blocks.hasRank(0):
            A1 = blocks.getBlock(0).getA1()
            sigma = self.comm.recv(source=s2%self.size, tag=2*num + s2)
            alpha = (A1[j,j]**2 - sigma)**0.5            
            if (np.real(A1[j,j] + alpha) < np.real(A1[j, j] - alpha)):
                z = A1[j, j]-alpha
                A1[j,j] = alpha 
            else:
                z = A1[j, j]+alpha
                A1[j,j] = -alpha
            self.comm.send(z, dest=s2%self.size, tag=3*num + s2)
            beta = 2*z*z/(-sigma + z*z)           
            self.comm.send(beta, dest=s2%self.size, tag=4*num + s2)
            
            data = self.comm.recv(source=s2%self.size, tag=5*num + s2)
#            X2 = data[:self.m]
#            beta = data[-1]
            del A1

        return data#X2, beta

	
