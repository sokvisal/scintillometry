import numpy as np
class Block:
    def __init__(self, rank):
        self.work1 = None
        self.work2 = None
        self.rank = rank
        self.T = None
        
    def setT(self, T):
        self.T = T
        
    def deleteT(self):
        del self.T
        
    def createA(self, A1):
        self.A1 = A1
        self.A2 = 1j*A1
        
    def createTemp(self, temp):
        self.temp = temp
        
    def createCond(self, isZero):
        self.isZero = isZero
        
    def setTemp(self, temp):
        self.temp = temp
        
    def getTemp(self):
        return self.temp
    
    def setTrue(self, isZero):
        self.isZero = isZero
        
    def setFalse(self, isZero):
        self.isZero = isZero
        
    def getCond(self):
        return self.isZero
        
    def setA1(self, A1):
        self.A1 = A1
        
    def setA2(self, A2):
        self.A2 = A2
        
    def setWork1(self, work1 = None):
        self.work1 = work1
    def setWork2(self, work2 = None):
        self.work2 = work2
        
    def setWork(self, work1, work2):
        self.setWork1(work1)
        self.setWork2(work2)
        
    def setName(self, name):
        self.Name = name
        
    def updateuc(self, i):
        uc = np.load(self.Name)
        m = self.A1.shape[0]
        try:
            temp = -np.conj(self.A1).T[0,:m/2]
            temp2 = -np.conj(self.A1).T[1:m/2+1,0][::-1]
            #print temp[0]
            uc[m*i:m*(i+1),0] = np.append(temp,temp2)
            #Ltemp[0,:mlen], Ltemp[1:mlen+1,0][::-1]
        except m == 1:
            uc[m*i:m*(i+1),:] = -np.conj(self.A1).T[:,:]
        
        np.save(self.Name, uc)
    
    def getWork(self):
        return self.work1(), self.work2()
        
    def getWork1(self):
        return self.work1
        
    def getWork2(self):
        return self.work2
    
    def getA1(self):
        return self.A1
    
    def getA2(self):
        return self.A2
    def getT(self):
        return self.T
        
        
