__DESCRIPTION__ ="""
Example:
import numpy as np
t=np.arange(10001,dtype=float)
x=np.sin(2*np.pi*t/398.+np.pi/6.)
LS = LombScargle1D(t,x)
sampleSpc = LS_DFT(1./398./10.,1./398.*15,1./398.*0.02)
sampleSpc.sampleSpectrum(LS)
"""

class dataList :
   def __init__(self,t,data,filterNaN) :
      import numpy as np
      idx=np.where(np.isnan(data)==False)[0]
      if len(idx) == 0 :
         self.t=[]
         self.t0=None
         self.idx=[]
         self.data=[]
      else :
         self.t = t[idx]
         self.idx = idx
         self.data = data[idx]
         self.t0 = self.t[0].min()
         self.t=self.t-self.t0
   def __len__(self) :
      return len(self.t)
   def copy(self) :
      import copy
      return copy.deepcopy(self)
   def replicate(self,n,Tmax) :
      import numpy as np
      out=self.copy()
      for k in range(1,n) :
         out.t=np.concatenate((out.t,self.t+Tmax*k))
         out.idx=np.concatenate((out.idx,self.idx+len(self)))
         out.data=np.concatenate((out.data,self.data))
      return out
   def addNoise(self,sigma) :
      import numpy.random
      self.data+=numpy.random.normal(0.,sigma,len(self))

class LombScargle1D(dataList) :
   """ Class to handle Lomb and Scargle fitting of a single sinusoid
       This class is used in conjunction with LS_DFT
       It gets in input a time base and data.
       Stores them, on request computes lomb scargle at a given frequency nu
   """
   def __init__(self,t,data,filterNaN=True) :
      """ t=times for samples, data=data """
      dataList.__init__(self,t,data,filterNaN)
      import numpy as np
      idx=np.where(np.isnan(data)==False)[0]
      if len(idx) == 0 :
         self.t=[]
         self.t0=None
         self.idx=[]
         self.data=[]
      else :
         self.t = t[idx]
         self.idx = idx
         self.data = data[idx]
         self.t0 = self.t[0].min()
         self.t=self.t-self.t0
   def keys(self,all=False) :
      a=[]
      for k in self.__dict__.keys() :
         if (k != 'data' and k != 't0' and k!='t' and k!='idx') or all : a.append(k)
      return a
   def TimeWindow(self) :
      if self.__len__() == 0 :
         return None
      return None
   def noBaseline(self,nu) :
      """computes LS at a specific frequency nu without to account for baseline 
         all the moments are added to the object
         the most important moments: cosine component, sin component and baseline are returned in output
      """
      import numpy as np
      self.ndof = self.__len__()-2
      self.omega=nu*2.*np.pi
      self.ot = self.omega*self.t
      self.tau = np.arctan2((np.sin(2.*self.ot)).sum(),(np.cos(2.*self.ot)).sum())/(2.*self.omega)
      #
      self.C = np.cos(self.ot -self.omega*self.tau)
      self.S = np.sin(self.ot -self.omega*self.tau)
      self.SUM_CC = (self.C**2).sum()
      self.SUM_SS = (self.S**2).sum()
      self.SUM_CS = (self.C*self.S).sum()
      self.SUM_CD = (self.C*self.data).sum()
      self.SUM_SD = (self.S*self.data).sum()
      self.Delta=self.SUM_CC*self.SUM_SS-self.SUM_CS**2
      #
      self.A = (self.SUM_SS*self.SUM_CD-self.SUM_CS*self.SUM_SD)/self.Delta
      self.B = (-self.SUM_CS*self.SUM_CD+self.SUM_CC*self.SUM_SD)/self.Delta
      self.Baseline=0.
      self.nSigmaA=((self.SUM_SS**2*self.SUM_CC+self.SUM_CS**2*self.SUM_SS)**0.5)/self.Delta
      self.nSigmaB=((self.SUM_CS**2*self.SUM_CC+self.SUM_CC**2*self.SUM_SS)**0.5)/self.Delta
      #
      self.Amplitude=(self.A**2+self.B**2)**0.5
      self.nSigmaAmpbyAmp=((self.A*self.nSigmaA)**2+(self.B*self.nSigmaB)**2)**0.5
      self.Phase=np.arctan2(self.B/self.Amplitude,self.A/self.Amplitude)+self.omega*self.tau
      return self.A,self.B, self.Baseline
   def withBaseLine(self,nu) :
      """computes LS at a specific frequency nu accounting for baseline 
         all the moments are added to the object
         the most important moments: cosine component, sin component and baseline are returned in output
      """
      import numpy as np
      self.ndof = self.__len__()-3
      self.omega=nu*2.*np.pi
      self.ot = self.omega*self.t
      self.tau = np.arctan2((np.sin(2.*self.ot)).sum(),(np.cos(2.*self.ot)).sum())/(2.*self.omega)
      #
      self.C = np.cos(self.ot -self.omega*self.tau)
      self.S = np.sin(self.ot -self.omega*self.tau)
      self.SUM_C = (self.C).sum()
      self.SUM_S = (self.S).sum()
      self.SUM_CC = (self.C**2).sum()
      self.SUM_SS = (self.S**2).sum()
      self.SUM_CS = (self.C*self.S).sum()
      self.SUM_CD = (self.C*self.data).sum()
      self.SUM_SD = (self.S*self.data).sum()
      self.SUM_D = (self.data).sum()
      self.n = float(len(self.data))
      self.M = np.array([[self.SUM_CC,self.SUM_CS,self.SUM_C],[self.SUM_CS,self.SUM_SS,self.SUM_S],[self.SUM_C,self.SUM_S,self.n]])
      self.Delta=np.linalg.det(self.M)
      self.IM=np.linalg.inv(self.M)
      self.V=np.array([self.SUM_CD,self.SUM_SD,self.SUM_D])
      self.A=(self.IM[0,:]*self.V).sum()
      self.B=(self.IM[1,:]*self.V).sum()
      self.Baseline=(self.IM[2,:]*self.V).sum()
      self.Amplitude=(self.A**2+self.B**2)**0.5
      
      self.nSigmaA=( (self.IM[0,0])**2*self.SUM_CC+(self.IM[0,1])**2*self.SUM_SS+(self.IM[0,2])**2*self.n)**0.5
      self.nSigmaB=( (self.IM[1,0])**2*self.SUM_CC+(self.IM[1,1])**2*self.SUM_SS+(self.IM[1,2])**2*self.n)**0.5
      self.nSigmaBase=( (self.IM[2,0])**2*self.SUM_CC+(self.IM[2,1])**2*self.SUM_SS+(self.IM[2,2])**2*self.n)**0.5
      
      self.nSigmaAmpbyAmp=((self.A*self.nSigmaA)**2+(self.B*self.nSigmaB)**2)**0.5

      self.Phase=np.arctan2(self.B/self.Amplitude,self.A/self.Amplitude)+self.omega*self.tau
      return self.A,self.B, self.Baseline
   def goodness(self) :
      """computes goodness of fit for the last calculated LS"""
      self.fit=self.A*self.C+self.B*self.S+self.Baseline
      self.residual=self.data-self.fit
      self.chisq=(self.residual**2).sum()
      return self.chisq
   
class LS_DFT :
   """ class to handle a LombScargle1D object to compute a periodogram """
   def __init__(self,f0,bw1,bw2,step,groups=1) :
      """f0 = central frequency,
         bw1 = lower limit of bandwidth (f0*bw1 <= f)
         bw2 = upper limit of bandwidth (f <= f0*bw2)
         step = step
      """
      import numpy as np
      self.f0=f0
      self.freq1=f0*bw1
      self.freq2=f0*bw2
      self.step=f0*step
      self.groups=groups
      if groups == 1 :
         self.f=np.arange(bw1,bw2+step,step)*self.f0
      else :
         f=np.arange(bw1,bw2+step,step)*self.f0
         fff = f*1
         for n in range(1,groups+1) :
            fff=np.concatenate((fff,f*n))
         self.f=fff*1
         self.freq1=self.f.min()
         self.freq2=self.f.max()
      try :
         self.i0 = np.where(abs(self.f - self.f0)<1e-10)[0].min()
      except :
         None
      self.A = self.f*0
      self.eA = self.f*0
      self.B = self.f*0
      self.eB = self.f*0
      self.Amplitude = self.f*0
      self.eAmp = self.f*0
      self.Baseline=self.f*0
      self.chisq=self.f*0
   def keys(self) :
      return self.__dict__.keys()
   def __len__(self) :
      return len(self.f)
   def sampleSpectrum(self,LS,noBaseline=True) :
      """for a given LS computes the sample spectrum"""
      for k in range(len(self.f)) :
         if noBaseline :
            LS.noBaseline(self.f[k])
         else :
            LS.withBaseLine(self.f[k])
         self.A[k]=LS.A
         self.B[k]=LS.B
         try :
            self.eA[k]=LS.nSigmaA
            self.eB[k]=LS.nSigmaB
            self.eAmp[k]=LS.nSigmaAmpbyAmp/LS.Amplitude
         except :
            None
         self.Baseline[k]=LS.Baseline
         self.Amplitude[k]=LS.Amplitude
         self.chisq[k]=LS.goodness()
         
def Integral_expSinPhi2(Alpha,nsmp=1000) :
   """returns int_0^2pi dphi exp(-Alpha*sin(phi)**2
      for Alpha=0 the analytical result is 2*pi
      with nsmp =3 the accuracy is about 3 decimal position
      uses trapezoidal integration
   """
   import numpy as np
   phi=np.arange(nsmp)/float(nsmp-1)*2.*np.pi
   if type(Alpha) == type(0.) or type(Alpha) == type(0) :
      itg=np.exp(-Alpha*np.sin(phi)**2)
      return np.pi/float(nsmp-1)*(itg[1:]+itg[0:-1]).sum()
   res=np.zeros(len(Alpha))
   for k in range(len(Alpha)) : 
      itg=np.exp(-Alpha[k]*np.sin(phi)**2)
      res[k]=(itg[1:]+itg[0:-1]).sum()
   return np.pi/float(nsmp-1)*res

def Integral_expCosPhi2(Alpha,nsmp=1000) :
   """returns int_0^2pi dphi exp(-Alpha*cos(phi)**2
      for Alpha=0 the analytical result is 2*pi
      with nsmp =3 the accuracy is about 3 decimal position
      uses trapezoidal integration
   """
   import numpy as np
   phi=np.arange(nsmp)/float(nsmp-1)*2.*np.pi
   if type(Alpha) == type(0.) or type(Alpha) == type(0) :
      itg=np.exp(-Alpha*np.cos(phi)**2)
      return np.pi/float(nsmp-1)*(itg[1:]+itg[0:-1]).sum()
   res=np.zeros(len(Alpha))
   for k in range(len(Alpha)) : 
      itg=np.exp(-Alpha[k]*np.cos(phi)**2)
      res[k]=(itg[1:]+itg[0:-1]).sum()
   return np.pi/float(nsmp-1)*res

def Integral_expCosSinPhi2(Alpha,Beta,nsmp=1000) :
   """returns int_0^2pi dphi exp(-Alpha*cos(phi)**2
      for Alpha=0 the analytical result is 2*pi
      with nsmp =3 the accuracy is about 3 decimal position
      uses trapezoidal integration
   """
   import numpy as np
   phi=np.arange(nsmp)/float(nsmp-1)*2.*np.pi
   if type(Alpha) == type(0.) or type(Alpha) == type(0) :
      itg=np.exp(-Alpha*np.cos(phi)**2-Beta*np.sin(phi)**2)
      return np.pi/float(nsmp-1)*(itg[1:]+itg[0:-1]).sum()
   res=np.zeros(len(Alpha))
   for k in range(len(Alpha)) : 
      itg=np.exp(-Alpha[k]*np.cos(phi)**2-Beta[k]*np.cos(phi)**2)
      res[k]=(itg[1:]+itg[0:-1]).sum()
   return np.pi/float(nsmp-1)*res

if __name__=='__main__' :
   import numpy as np
   import time 
   
   t=np.arange(10001,dtype=float)
   x=np.sin(2*np.pi*t/398.+np.pi/6.)
   tic = time.time()
   LS = LombScargle1D(t,x)
   sampleSpc = LS_DFT(1./398./10.,1./398.*15,1./398.*0.02)
   sampleSpc.sampleSpectrum(LS)
   tic = time.time()-tic
   print tic,' seconds'
