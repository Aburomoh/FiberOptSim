import matplotlib.pyplot as plt
from matplotlib import *
from numpy import *
import numpy as np
import cmath
from scipy import special as sp
import scipy.integrate as integrate
from numpy.fft import fft, ifft, fftshift, ifftshift
import tensorflow as tf
from tensorflow import keras as K
from scipy.linalg import dft

############################################################################################################
#random data generation 
def source (L):
 bits=np.random.randint(2,size=L)
 return bits
############################################################################################################
def cos_m (theta):
     out=np.cos(theta)
     tol=1e-8
     if abs(out)<tol:
         out=0
     return out
#####################################################################################################
def dataset_read(signalBeforeHardDecision_XI, signalBeforeHardDecision_XQ,
                 signalBeforeHardDecision_YI, signalBeforeHardDecision_YQ,
                 signalBeforeHardDecision_Ref_XI, signalBeforeHardDecision_Ref_XQ,
                 signalBeforeHardDecision_Ref_YI, signalBeforeHardDecision_Ref_YQ):
    XI_CAP = open(signalBeforeHardDecision_XI)
    num_XI_CAP = XI_CAP.readlines()
    XQ_CAP = open(signalBeforeHardDecision_XQ)
    num_XQ_CAP = XQ_CAP.readlines()
    YI_CAP = open(signalBeforeHardDecision_YI)
    num_YI_CAP = YI_CAP.readlines()
    YQ_CAP = open(signalBeforeHardDecision_YQ)
    num_YQ_CAP = YQ_CAP.readlines()
    a_X_cap = np.zeros(len(num_XI_CAP))
    b_X_cap = np.zeros(len(num_XQ_CAP))
    a_Y_cap = np.zeros(len(num_YI_CAP))
    b_Y_cap = np.zeros(len(num_YQ_CAP))

    XI_SEND = open(signalBeforeHardDecision_Ref_XI)
    num_XI_SEND = XI_SEND.readlines()
    XQ_SEND = open(signalBeforeHardDecision_Ref_XQ)
    num_XQ_SEND = XQ_SEND.readlines()
    YI_SEND = open(signalBeforeHardDecision_Ref_YI)
    num_YI_SEND = YI_SEND.readlines()
    YQ_SEND = open(signalBeforeHardDecision_Ref_YQ)
    num_YQ_SEND = YQ_SEND.readlines()
    a_X_SEND = np.zeros(len(num_XI_SEND))
    b_X_SEND = np.zeros(len(num_XQ_SEND))
    a_Y_SEND = np.zeros(len(num_YI_SEND))
    b_Y_SEND = np.zeros(len(num_YQ_SEND))
    for index in range(len(num_XI_CAP)):
        a_X_cap[index] = np.fromstring(num_XI_CAP[index], dtype=float, sep=' ')
        b_X_cap[index] = np.fromstring(num_XQ_CAP[index], dtype=float, sep=' ')
        a_Y_cap[index] = np.fromstring(num_YI_CAP[index], dtype=float, sep=' ')
        b_Y_cap[index] = np.fromstring(num_YQ_CAP[index], dtype=float, sep=' ')
    for index in range(len(num_XI_SEND)):
        a_X_SEND[index] = np.fromstring(num_XI_SEND[index], dtype=float, sep=' ')
        b_X_SEND[index] = np.fromstring(num_XQ_SEND[index], dtype=float, sep=' ')
        a_Y_SEND[index] = np.fromstring(num_YI_SEND[index], dtype=float, sep=' ')
        b_Y_SEND[index] = np.fromstring(num_YQ_SEND[index], dtype=float, sep=' ')

    n1 = np.where(np.abs(a_X_cap) > 0)
    n2 = np.arange(len(a_X_SEND))

    c_X_cap1 = a_X_cap[n1] + 1j * b_X_cap[n1]
    c_Y_cap1 = a_Y_cap[n1] + 1j * b_Y_cap[n1]

    c_X_cap = c_X_cap1[n2]
    c_Y_cap = c_Y_cap1[n2]

    c_X_SEND = a_X_SEND + 1j * b_X_SEND
    c_Y_SEND = a_Y_SEND + 1j * b_Y_SEND

    c_X_cap = c_X_cap * np.sqrt(np.mean(np.abs(c_X_SEND) ** 2)) / np.sqrt(np.mean(np.abs(c_X_cap) ** 2))
    c_Y_cap = c_Y_cap * np.sqrt(np.mean(np.abs(c_Y_SEND) ** 2)) / np.sqrt(np.mean(np.abs(c_Y_cap) ** 2))

    return c_X_cap, c_Y_cap, c_X_SEND, c_Y_SEND

########################################################

def encode(A,c=np.sqrt(0.4)):
    dim = np.shape(A[0])
    N = dim[0]
    CatXr = np.zeros(dim[:2])
    CatXi = np.zeros(dim[:2])
    CatYr = np.zeros(dim[:2])
    CatYi = np.zeros(dim[:2])
    for i in range(N):
        Xr = A[0][i,:,0]
        Xi = A[1][i,:,0]
        Yr = A[2][i,:,0]
        Yi = A[3][i,:,0]
        CatXr[i] = (Xr > -c)*1 + (Xr > 0)*1 + (Xr > c)*1
        CatXi[i] = (Xi > -c)*1 + (Xi > 0)*1 + (Xi > c)*1
        CatYr[i] = (Yr > -c)*1 + (Yr > 0)*1 + (Yr > c)*1
        CatYi[i] = (Yi > -c)*1 + (Yi > 0)*1 + (Yi > c)*1
    return np.concatenate([np.expand_dims(CatXr,axis=2),np.expand_dims(CatXi,axis=2),np.expand_dims(CatYr,axis=2),np.expand_dims(CatYi,axis=2)],axis=2)


def encode2(B):
    dim = np.shape(B)
    C = np.zeros((dim[0],dim[1],dim[2]*3))
    
    C[:,:,0] = B[:,:,0] >= 1
    C[:,:,1] = B[:,:,0] >= 2
    C[:,:,2] = B[:,:,0] >= 3
    
    C[:,:,3] = B[:,:,1] >= 1
    C[:,:,4] = B[:,:,1] >= 2
    C[:,:,5] = B[:,:,1] >= 3
    
    C[:,:,6] = B[:,:,2] >= 1
    C[:,:,7] = B[:,:,2] >= 2
    C[:,:,8] = B[:,:,2] >= 3
    
    C[:,:,9] = B[:,:,3] >= 1
    C[:,:,10] = B[:,:,3] >= 2
    C[:,:,11] = B[:,:,3] >= 3
    
    return C

########################################################

def decode2(C,isrounded=True):
    dim = np.shape(C)
    D = np.zeros((dim[0],4))
    if isrounded:
        D[:,0] = np.sum(np.round(C[:,0:3]),axis=1)
        D[:,1] = np.sum(np.round(C[:,3:6]),axis=1)
        D[:,2] = np.sum(np.round(C[:,6:9]),axis=1)
        D[:,3] = np.sum(np.round(C[:,9:12]),axis=1)
    else:
        D[:,0] = np.sum(C[:,0:3],axis=1)
        D[:,1] = np.sum(C[:,3:6],axis=1)
        D[:,2] = np.sum(C[:,6:9],axis=1)
        D[:,3] = np.sum(C[:,9:12],axis=1)
    return D

########################################################  
def sin_m (theta):
     out=np.sin(theta)
     tol=1e-8
     if abs(out)<tol:
         out=0
     return out
####################################################################################################
#Symbol generation 
def QAM16(data,p):
    l1=len(data)
    lim1=int(l1/4)
    sym=np.zeros((lim1))+0j
    m=np.array([8,4,2,1])
    for l in range (lim1):
        part=data[4*l:4*l+4]
        dec_val=dot(part,m)
        if dec_val==0:
          sym[l]=-3-3j
        elif dec_val==1:
          sym[l]=-3-1j
        elif dec_val==2:
          sym[l]=-3+3j
        elif dec_val==3:
          sym[l]=-3+1j
        elif dec_val==4:
          sym[l]=-1-3j
        elif dec_val==5:
          sym[l]=-1-1j
        elif dec_val==6:
          sym[l]=-1+3j
        elif dec_val==7:
          sym[l]=-1+1j
        elif dec_val==8:
          sym[l]=3-3j
        elif dec_val==9:
          sym[l]=3-1j
        elif dec_val==10:
          sym[l]=3+3j
        elif dec_val==11:
          sym[l]=3+1j
        elif dec_val==12:
          sym[l]=1-3j
        elif dec_val==13:
          sym[l]=1-1j
        elif dec_val==14:
          sym[l]=1+3j
        elif dec_val==15:
          sym[l]=1+1j
        
    sym=sqrt(p)*(sqrt(16)/sqrt(160))*(sym) 
             
    return sym

###########################################################################################################
def QAM64(data,p):
    l1=len(data)
    lim1=int(l1/6)
    sym=np.zeros((lim1))+0j
    m=np.array([32,16,8,4,2,1])
    for l in range (lim1):
        part=data[6*l:6*l+6]
        dec_val=np.dot(part,m)
        sym[l] = 0
        if dec_val<8:
          sym[l] += -7
        elif dec_val<16:
          sym[l] += -5
        elif dec_val<24:
          sym[l] += -1
        elif dec_val<32:
          sym[l] += -3
        elif dec_val<40:
          sym[l] += 7
        elif dec_val<48:
          sym[l] += 5
        elif dec_val<56:
          sym[l] += 1
        else:
          sym[l] += 3
        
        if np.mod(dec_val,8) == 0:
          sym[l] += -7j
        elif np.mod(dec_val,8) == 1:
          sym[l] += -5j
        elif np.mod(dec_val,8) == 2:
          sym[l] += -1j
        elif np.mod(dec_val,8) == 3:
          sym[l] += -3j
        elif np.mod(dec_val,8) == 4:
          sym[l] += 7j
        elif np.mod(dec_val,8) == 5:
          sym[l] += 5j
        elif np.mod(dec_val,8) == 6:
          sym[l] += 1j
        else:
          sym[l] += 3j
          
    sym = sqrt(p)/sqrt(42)*(sym) 
             
    return sym

###########################################################################################################
def deQAM16(rec_symbs,p):
    const=sqrt(p)/sqrt(10)*np.array([-3-3j,-3-1j,-3+3j,-3+1j,-1-3j,-1-1j,-1+3j,-1+1j,3-3j,3-1j,3+3j,3+1j,1-3j,1-1j,1+3j,1+1j])
    l_r=len(rec_symbs)
    s=np.zeros((l_r))+0j
    dis_M = np.zeros((l_r,16))
    for l in range (l_r):
        dis_M[l]=abs(rec_symbs[l]-const)
    s=const[np.argmin(dis_M,axis=1)]
    return s
###########################################################################################################
def mod(t, s, B, rof=0.0625): #sinc modulation
    dt = t[1] - t[0]
    Nfft = len(t)
    sampling = int(shape(t)[0]/shape(s)[0])
    Ns=len(s)
    
    l1 = -int(floor(Ns/2))
    l2 = int(ceil(Ns/2))
    q = np.zeros(Nfft) + 0j
    for l in range (l1,l2):
        indx = l+int(floor(Ns/2))
        lb = max(0,sampling*(indx-20))
        ub = min(sampling*(indx+20),Nfft)
        i = slice(lb,ub)
        q[i] = q[i] + s[indx]*(rcosine(t[i]-l/B,B,rof))
    return q
#############################################

def rcosine(t, B, rof):
    eps = 1e-8
    num = np.sinc(t*B)*np.cos(np.pi*rof*(t*B))
    den = (1-(2*rof*t*B)**2)
    bad_ind = np.argwhere(abs(den) < eps)
    den[bad_ind] = den[bad_ind+1]
    return num/den
#############################################

def phase_noise(N_sym,linewidth,T_sym,samp_rate):
    return np.cumsum(np.sqrt(2*np.pi*linewidth*T_sym/samp_rate) *np.random.randn(N_sym*samp_rate))

#############################################
def s2b(s):
    l_s=len(s)
    dec=np.zeros((l_s))
    
    for l in range(l_s):
        dec[l] = 0
        if np.real(s[l])==-1:
          dec[l]+=4
        elif np.real(s[l])==3:
          dec[l]+=8
        elif np.real(s[l])==1:
          dec[l]+=12
        
        if np.imag(s[l])==-1:
          dec[l]+=1
        elif np.imag(s[l])==3:
          dec[l]+=2
        elif np.imag(s[l])==1:
          dec[l]+=3
        
    r_data = np.zeros((4*l_s))
    for l in range(l_s):
        r_data[4*l:4*l+4] = d2b(int(dec[l]),4)
    return r_data

#########################################################################################################
def s2bQAM64(s):
    l_s = len(s)
    dec = np.zeros((l_s))
    
    for l in range(l_s):
        dec[l] = 0
        if np.real(s[l])==3:
            dec[l]+=56
        elif np.real(s[l])==1:
            dec[l]+=48
        elif np.real(s[l])==5:
            dec[l]+=40
        elif np.real(s[l])==7:
            dec[l]+=32
        elif np.real(s[l])==-3:
            dec[l]+=24
        elif np.real(s[l])==-1:
            dec[l]+=16
        elif np.real(s[l])==-5:
            dec[l]+=8
          
        if np.real(s[l])==3:
            dec[l]+=7
        elif np.real(s[l])==1:
            dec[l]+=6
        elif np.real(s[l])==5:
            dec[l]+=5
        elif np.real(s[l])==7:
            dec[l]+=4
        elif np.real(s[l])==-3:
            dec[l]+=3
        elif np.real(s[l])==-1:
            dec[l]+=2
        elif np.real(s[l])==-5:
            dec[l]+=1
        
    r_data = np.zeros((6*l_s))
    for l in range(l_s):
        r_data[6*l:6*l+6] = d2b(int(dec[l]),6)
    return r_data

#########################################################################################################

def BER(Qfac):
    return sp.erfc(10**(Qfac/20)/np.sqrt(2))/2

#########################################################################################################
def Qfac(BER):
    return 20*np.log10(sqrt(2)*sp.erfcinv(2*BER))

#########################################################################################################
def ssfm_dpol(signal,l_f,n_span,n_seg,gamma,alpha_db,beta2,B,dt,h,f0,TAU,THETA,PHI):
    #Noise addition
    alpha=(log(10)/10)*(10**(-3))*(alpha_db)
    l_span=(l_f/n_span)
    G_db=alpha_db*(l_span*(10**-3))
    g=10**(G_db/10)     #???
    signal_recx,signal_recy=signal[0],signal[1]
    l_s=int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    step_s=(l_f/((n_seg)*(n_span)))
    dis=np.exp(-(1j/2)*beta2*(w**2)*step_s)#dispersion.
    att=np.exp((-alpha/2)*step_s)#loss
    for j in range(n_span):
        for jj in range(n_seg):
            signal_recfx=fft(signal_recx)
            signal_recfy=fft(signal_recy)
            #add dispersion 
            signal_recfx=signal_recfx*dis
            signal_recfy=signal_recfy*dis
            #Birefringence
            #theta=0
            theta=THETA[j*n_seg+jj]#2*pi*np.random.uniform(0,1)
            signal_recfx_i= cos_m(theta)*signal_recfx + sin_m(theta)*signal_recfy
            signal_recfy_i= -sin_m(theta)*signal_recfx + cos_m(theta)*signal_recfy
            
            #phi=0
            phi=PHI[j*n_seg+jj]#2*pi*np.random.uniform(0,1)
            signal_recfx_i=signal_recfx_i*np.exp(1j*phi/2)
            signal_recfy_i=signal_recfy_i*np.exp(-1j*phi/2)
            #add pmd
            # tau = sqrt((PMD*1e-12)**2*step_s)
            tau = TAU[j*n_seg+jj] #sqrt((PMD*1e-12)**2*(step_s*10**(-3)))*np.random.normal()
            signal_recfx_i=signal_recfx_i*(np.exp(1j*w*tau/2))
            signal_recfy_i=signal_recfy_i*(np.exp(-1j*w*tau/2))
            # #reverce 
            # signal_recfx = cos_m(-theta)*signal_recfx_i+ sin_m(-theta)*signal_recfy_i
            # signal_recfy = -sin_m(-theta)*signal_recfx_i + cos_m(-theta)*signal_recfy_i
            # add loss +nonlinearity 
            signal_recx=ifft(signal_recfx_i)
            signal_recy=ifft(signal_recfy_i)
            signal_recx=signal_recx*att
            signal_recy=signal_recy*att
            signal_recx=signal_recx*np.exp(1j*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s)
            signal_recy=signal_recy*np.exp(1j*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s)
        signal_recx=sqrt(g)*signal_recx
        signal_recy=sqrt(g)*signal_recy
    # signal_rec = np.array([signal_recx,signal_recy])
    return signal_recx,signal_recy

def ssfm_dpolc(signal,l_f,n_span,n_seg,gamma,alpha_db,beta2,B,dt,h,f0,PMD):
    #Noise addition
    alpha=(log(10)/10)*(10**(-3))*(alpha_db)
    l_span=(l_f/n_span)
    G_db=alpha_db*(l_span*(10**-3))
    g=10**(G_db/10)
    signal_recx,signal_recy=signal[0],signal[1]
    l_s=int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    step_s=(l_f/((n_seg)*(n_span)))
    dis=np.exp(-(1j/2)*beta2*(w**2)*step_s)#dispersion.
    att=np.exp((-alpha/2)*step_s)#loss
    for j in range(n_span):
        for jj in range(n_seg):
            signal_recfx=fft(signal_recx)
            signal_recfy=fft(signal_recy)
            #add dispersion 
            signal_recfx=signal_recfx*dis
            signal_recfy=signal_recfy*dis
            #Birefringence
            #theta=0
            theta=2*pi*np.random.uniform(0,1)#random theta for each segmant.
            signal_recfx_i= cos_m(theta)*signal_recfx + sin_m(theta)*signal_recfy
            signal_recfy_i= -sin_m(theta)*signal_recfx + cos_m(theta)*signal_recfy
            
            #phi=0
            phi=2*pi*np.random.uniform(0,1)#random phi for each segmant.
            signal_recfx_i=signal_recfx_i*np.exp(1j*phi/2)
            signal_recfy_i=signal_recfy_i*np.exp(-1j*phi/2)
            #add pmd
            # tau = sqrt((PMD*1e-12)**2*step_s)
            tau =sqrt((PMD*1e-12)**2*(step_s*10**(-3)))*np.random.normal()
            signal_recfx_i=signal_recfx_i*(np.exp(1j*w*tau/2))
            signal_recfy_i=signal_recfy_i*(np.exp(-1j*w*tau/2))
            # #reverce 
            # signal_recfx = cos_m(-theta)*signal_recfx_i+ sin_m(-theta)*signal_recfy_i
            # signal_recfy = -sin_m(-theta)*signal_recfx_i + cos_m(-theta)*signal_recfy_i
            # add loss +nonlinearity 
            signal_recx=ifft(signal_recfx_i)
            signal_recy=ifft(signal_recfy_i)
            signal_recx=signal_recx*att
            signal_recy=signal_recy*att
            signal_recx_l=signal_recx
            signal_recy_l=signal_recy
            signal_recx=signal_recx*np.exp(1j*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s)
            signal_recy=signal_recy*np.exp(1j*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s)
        signal_recx=sqrt(g)*signal_recx
        signal_recy=sqrt(g)*signal_recy
        signal_recx_l=sqrt(g)*signal_recx_l
        signal_recy_l=sqrt(g)*signal_recy_l
    # signal_rec = np.array([signal_recx,signal_recy])
    return signal_recx,signal_recy,signal_recx_l,signal_recy_l

def ssfm_dpol_nonsymm(signal,l_span,gamma,alpha_db,beta2,dt,TAU,THETA,PHI,B,NF_db,h,f0,N_s,t):
    alpha = (np.log(10)/10)*(10**(-3))*(alpha_db)
    G_db = alpha_db*(l_span)
    g = 10**(G_db/10)
    nf=10**(NF_db/10)
    var = (g[0]*g[1]-1)*h*f0*(nf)*B

    signal_recx,signal_recy = signal[0],signal[1]
    l_s = int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * np.fft.fftshift(np.arange(-l_s/2,l_s/2))
    step_s = 1e3
    
    att = np.exp((-alpha/2)*step_s)#loss
    n_span = int(len(l_span)/2)
    c = 0
    for j in range(n_span):
        for jj in range(l_span[j*2]):
            signal_recfx=np.fft.fft(signal_recx)
            signal_recfy=np.fft.fft(signal_recy)
            
            dis = np.exp(-(1j/2)*beta2[2*j]*(w**2)*step_s)
            signal_recfx=signal_recfx*dis
            signal_recfy=signal_recfy*dis
            #Birefringence
            theta=THETA[c]#2*pi*np.random.uniform(0,1)
            signal_recfx_i= cos_m(theta)*signal_recfx + sin_m(theta)*signal_recfy
            signal_recfy_i= -sin_m(theta)*signal_recfx + cos_m(theta)*signal_recfy
            #phi=0
            phi=PHI[c]#2*pi*np.random.uniform(0,1)
            signal_recfx_i=signal_recfx_i*np.exp(1j*phi/2)
            signal_recfy_i=signal_recfy_i*np.exp(-1j*phi/2)
            # tau = sqrt((PMD*1e-12)**2*step_s)
            tau = TAU[c] #sqrt((PMD*1e-12)**2*(step_s*10**(-3)))*np.random.normal()
            signal_recfx_i=signal_recfx_i*(np.exp(1j*w*tau/2))
            signal_recfy_i=signal_recfy_i*(np.exp(-1j*w*tau/2))
            # add loss +nonlinearity 
            signal_recx=np.fft.ifft(signal_recfx_i)
            signal_recy=np.fft.ifft(signal_recfy_i)
            signal_recx=signal_recx*att
            signal_recy=signal_recy*att
            signal_recx=signal_recx*np.exp(1j*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s)
            signal_recy=signal_recy*np.exp(1j*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s)
            c += 1
        signal_recx=np.sqrt(g[2*j])*signal_recx
        signal_recy=np.sqrt(g[2*j])*signal_recy
        for jj in range(l_span[j*2+1]):
            signal_recfx=fft(signal_recx)
            signal_recfy=fft(signal_recy)
            
            dis = np.exp(-(1j/2)*beta2[2*j+1]*(w**2)*step_s)
            signal_recfx=signal_recfx*dis
            signal_recfy=signal_recfy*dis
            theta=THETA[c]#2*pi*np.random.uniform(0,1)
            signal_recfx_i= cos_m(theta)*signal_recfx + sin_m(theta)*signal_recfy
            signal_recfy_i= -sin_m(theta)*signal_recfx + cos_m(theta)*signal_recfy
            
            phi=PHI[c]#2*pi*np.random.uniform(0,1)
            signal_recfx_i=signal_recfx_i*np.exp(1j*phi/2)
            signal_recfy_i=signal_recfy_i*np.exp(-1j*phi/2)
            tau = TAU[c] #sqrt((PMD*1e-12)**2*(step_s*10**(-3)))*np.random.normal()
            signal_recfx_i=signal_recfx_i*(np.exp(1j*w*tau/2))
            signal_recfy_i=signal_recfy_i*(np.exp(-1j*w*tau/2))
            signal_recx=np.fft.ifft(signal_recfx_i)
            signal_recy=np.fft.ifft(signal_recfy_i)
            signal_recx=signal_recx*att
            signal_recy=signal_recy*att
            c += 1
        
        noise_x = np.sqrt(var/2)*(np.random.randn(l_s)+1j*np.random.randn(l_s))
        noise_y = np.sqrt(var/2)*(np.random.randn(l_s)+1j*np.random.randn(l_s))
        signal_recx=np.sqrt(g[2*j+1])*signal_recx + noise_x
        signal_recy=np.sqrt(g[2*j+1])*signal_recy + noise_y
    return signal_recx,signal_recy

def ssfm1(signal, l_span, gamma, alpha_db, beta2, dt, TAU, THETA, PHI, B, NF_db, h, f0, N_s, t):
    alpha = (np.log(10)/10)*(10**(-3))*(alpha_db)
    G_db = alpha_db*(l_span)
    g = 10**(G_db/10)
    nf = 10**(NF_db/10)
    var = (g[0]*g[1]-1)*h*f0*(nf)*B

    signal_recx, signal_recy = signal[0], signal[1]
    l_s = int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * np.fft.fftshift(np.arange(-l_s/2, l_s/2))
    step_s = 1e3

    att = np.exp((-alpha/2)*step_s)  # loss
    n_span = int(len(l_span)/2)
    # c = 0
    for j in range(n_span):
        dis1 = np.exp(-(1j/2)*beta2[2*j]*(w**2)*step_s)
        dis2 = np.exp(-(1j/2)*beta2[2*j+1]*(w**2)*step_s)
        for _ in range(l_span[j*2]):
            signal_recfx = np.fft.fft(signal_recx)
            signal_recfy = np.fft.fft(signal_recy)

            signal_recfx = signal_recfx*dis1
            signal_recfy = signal_recfy*dis1

            # theta=THETA[c]
            # signal_recfx_i= cos_m(theta)*signal_recfx + sin_m(theta)*signal_recfy
            # signal_recfy_i= -sin_m(theta)*signal_recfx + cos_m(theta)*signal_recfy

            # phi=PHI[c]
            # signal_recfx_i=signal_recfx_i*np.exp(1j*phi/2)
            # signal_recfy_i=signal_recfy_i*np.exp(-1j*phi/2)

            # tau = TAU[c]
            # signal_recfx_i=signal_recfx_i*(np.exp(1j*w*tau/2))
            # signal_recfy_i=signal_recfy_i*(np.exp(-1j*w*tau/2))

            signal_recx = np.fft.ifft(signal_recfx) * att[2*j]
            signal_recy = np.fft.ifft(signal_recfy) * att[2*j]

            signal_recx = signal_recx * \
                np.exp(1j*8/9*gamma[2*j]*(abs(signal_recx)
                       ** 2+abs(signal_recy)**2)*step_s)
            signal_recy = signal_recy * \
                np.exp(1j*8/9*gamma[2*j]*(abs(signal_recy)
                       ** 2+abs(signal_recx)**2)*step_s)
            # c += 1
        signal_recx = np.sqrt(g[2*j])*signal_recx
        signal_recy = np.sqrt(g[2*j])*signal_recy
        for _ in range(l_span[j*2+1]):
            signal_recfx = np.fft.fft(signal_recx)
            signal_recfy = np.fft.fft(signal_recy)

            signal_recfx = signal_recfx*dis2
            signal_recfy = signal_recfy*dis2

            # theta=THETA[c]
            # signal_recfx_i= cos_m(theta)*signal_recfx + sin_m(theta)*signal_recfy
            # signal_recfy_i= -sin_m(theta)*signal_recfx + cos_m(theta)*signal_recfy

            # phi=PHI[c]
            # signal_recfx_i=signal_recfx_i*np.exp(1j*phi/2)
            # signal_recfy_i=signal_recfy_i*np.exp(-1j*phi/2)

            # tau = TAU[c]
            # signal_recfx_i=signal_recfx_i*(np.exp(1j*w*tau/2))
            # signal_recfy_i=signal_recfy_i*(np.exp(-1j*w*tau/2))

            signal_recx = np.fft.ifft(signal_recfx) * att[2*j+1]
            signal_recy = np.fft.ifft(signal_recfy) * att[2*j+1]
            # c += 1

        noise_x = np.sqrt(var/2)*(np.random.randn(l_s)+1j*np.random.randn(l_s))
        noise_y = np.sqrt(var/2)*(np.random.randn(l_s)+1j*np.random.randn(l_s))
        signal_recx = np.sqrt(g[2*j+1])*signal_recx + noise_x
        signal_recy = np.sqrt(g[2*j+1])*signal_recy + noise_y
    return signal_recx, signal_recy


def DBP1(signal, l_span, gamma, alpha_db, beta2, dt, TAU, THETA, PHI, B, NF_db, h, f0, N_s, t):
    alpha = (np.log(10)/10)*(10**(-3))*(alpha_db)
    G_db = alpha_db*(l_span)
    g = 10**(G_db/10)

    signal_recx, signal_recy = signal[0], signal[1]
    l_s = int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * np.fft.fftshift(np.arange(-l_s/2, l_s/2))
    step_s = 1e3

    att = np.exp((-alpha/2)*step_s)  # loss
    n_span = int(len(l_span)/2)
    # c = 0
    for j in range(n_span-1, -1, -1):
        dis1 = np.exp(-(1j/2)*-beta2[2*j]*(w**2)*step_s)
        dis2 = np.exp(-(1j/2)*-beta2[2*j+1]*(w**2)*step_s)

        signal_recx = signal_recx/np.sqrt(g[2*j+1])
        signal_recy = signal_recy/np.sqrt(g[2*j+1])
        for _ in range(l_span[j*2+1]):
            signal_recfx = np.fft.fft(signal_recx/att[2*j+1])
            signal_recfy = np.fft.fft(signal_recy/att[2*j+1])

            signal_recfx = signal_recfx*dis2
            signal_recfy = signal_recfy*dis2

            signal_recx = np.fft.ifft(signal_recfx)
            signal_recy = np.fft.ifft(signal_recfy)

        signal_recx = signal_recx/np.sqrt(g[2*j])
        signal_recy = signal_recy/np.sqrt(g[2*j])
        for _ in range(l_span[j*2]):
            signal_recx = signal_recx * \
                np.exp(1j*-8/9*gamma[2*j]*(abs(signal_recx)
                       ** 2+abs(signal_recy)**2)*step_s)
            signal_recy = signal_recy * \
                np.exp(1j*-8/9*gamma[2*j]*(abs(signal_recy)
                       ** 2+abs(signal_recx)**2)*step_s)

            signal_recfx = np.fft.fft(signal_recx/att[2*j])
            signal_recfy = np.fft.fft(signal_recy/att[2*j])

            signal_recfx = signal_recfx*dis1
            signal_recfy = signal_recfy*dis1

            signal_recx = np.fft.ifft(signal_recfx)
            signal_recy = np.fft.ifft(signal_recfy)

    return signal_recx, signal_recy

def DBP_nonsymm(signal,l_span,gamma,alpha_db,beta2,dt,TAU,THETA,PHI,B,NF_db,h,f0,N_s,t):
    alpha = (np.log(10)/10)*(10**(-3))*(alpha_db)
    G_db = alpha_db*(l_span)
    g = 10**(G_db/10)
    nf = 10**(NF_db/10)
    
    signal_recx,signal_recy = signal[0],signal[1]
    l_s = int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * np.fft.fftshift(np.arange(-l_s/2,l_s/2))
    step_s = 1e3
    
    att = np.exp((-alpha/2)*step_s)#loss
    n_span = int(len(l_span)/2)
    c = int(np.sum(l_span))
    for j in range(n_span):
        indx = -2*j-1
        signal_recx=signal_recx/np.sqrt(g[indx])
        signal_recy=signal_recy/np.sqrt(g[indx])
        for jj in range(l_span[-j*2-1]):
            
            c -= 1
            #signal_recx=signal_recx*np.exp(-1j*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s)
            #signal_recy=signal_recy*np.exp(-1j*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s)
            
            signal_recx=signal_recx/att
            signal_recy=signal_recy/att
            
            signal_recx_i=np.fft.fft(signal_recx)
            signal_recy_i=np.fft.fft(signal_recy)
            
            tau = TAU[c] #sqrt((PMD*1e-12)**2*(step_s*10**(-3)))*np.random.normal()
            signal_recfx_i=signal_recx_i*(np.exp(-1j*w*tau/2))
            signal_recfy_i=signal_recy_i*(np.exp(1j*w*tau/2))
            
            phi=PHI[c]#2*pi*np.random.uniform(0,1)
            signal_recfx_i=signal_recfx_i*np.exp(-1j*phi/2)
            signal_recfy_i=signal_recfy_i*np.exp(1j*phi/2)
            
            theta=THETA[c]#2*pi*np.random.uniform(0,1)
            signal_recfx= cos_m(-theta)*signal_recfx_i + sin_m(-theta)*signal_recfy_i
            signal_recfy= -sin_m(-theta)*signal_recfx_i + cos_m(-theta)*signal_recfy_i
            
            dis = np.exp((1j/2)*beta2[indx]*(w**2)*step_s)
            signal_recfx_i=signal_recfx*dis
            signal_recfy_i=signal_recfy*dis
            
            signal_recx=np.fft.ifft(signal_recfx_i)
            signal_recy=np.fft.ifft(signal_recfy_i)
            ##############################################
        indx = -2*j-2
        signal_recx=signal_recx/np.sqrt(g[indx])
        signal_recy=signal_recy/np.sqrt(g[indx])
        for jj in range(l_span[indx]):
            
            c -= 1
            signal_recx=signal_recx*np.exp(-1j*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s)
            signal_recy=signal_recy*np.exp(-1j*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s)
            
            signal_recx=signal_recx/att
            signal_recy=signal_recy/att
            
            signal_recx_i=np.fft.fft(signal_recx)
            signal_recy_i=np.fft.fft(signal_recy)
            
            tau = TAU[c] #sqrt((PMD*1e-12)**2*(step_s*10**(-3)))*np.random.normal()
            signal_recfx_i=signal_recx_i*(np.exp(-1j*w*tau/2))
            signal_recfy_i=signal_recy_i*(np.exp(1j*w*tau/2))
            
            phi=PHI[c]#2*pi*np.random.uniform(0,1)
            signal_recfx_i=signal_recfx_i*np.exp(-1j*phi/2)
            signal_recfy_i=signal_recfy_i*np.exp(1j*phi/2)
            
            theta=THETA[c]#2*pi*np.random.uniform(0,1)
            signal_recfx= cos_m(-theta)*signal_recfx_i + sin_m(-theta)*signal_recfy_i
            signal_recfy= -sin_m(-theta)*signal_recfx_i + cos_m(-theta)*signal_recfy_i
            
            dis = np.exp((1j/2)*beta2[indx]*(w**2)*step_s)
            signal_recfx_i=signal_recfx*dis
            signal_recfy_i=signal_recfy*dis
            
            signal_recx=np.fft.ifft(signal_recfx_i)
            signal_recy=np.fft.ifft(signal_recfy_i)
            ##############################################
    return signal_recx,signal_recy

def DSP_nonsymm(signal,l_span,gamma,alpha_db,beta2,dt,TAU,THETA,PHI,B,NF_db,h,f0,N_s,t):
    alpha = (np.log(10)/10)*(10**(-3))*(alpha_db)
    G_db = alpha_db*(l_span)
    g = 10**(G_db/10)
    nf = 10**(NF_db/10)
    
    signal_rectx,signal_recty = signal[0],signal[1]
    l_s = int(len(signal_rectx))
    w = 2*np.pi/(l_s*dt) * np.fft.fftshift(np.arange(-l_s/2,l_s/2))
    step_s = 1e3
    
    att = np.exp((-alpha/2)*step_s)#loss
    n_span = int(len(l_span)/2)
    c = int(np.sum(l_span))
    signal_recfx=np.fft.fft(signal_rectx)
    signal_recfy=np.fft.fft(signal_recty)
    for j in range(n_span):
        indx = -2*j-1
        
        for jj in range(l_span[-j*2-1]):
            c -= 1
            
            tau = TAU[c] #sqrt((PMD*1e-12)**2*(step_s*10**(-3)))*np.random.normal()
            signal_recfx = signal_recfx*(np.exp(-1j*w*tau/2))
            signal_recfy = signal_recfy*(np.exp(1j*w*tau/2))
            
            phi=PHI[c]#2*pi*np.random.uniform(0,1)
            signal_recfx = signal_recfx*np.exp(-1j*phi/2)
            signal_recfy = signal_recfy*np.exp(1j*phi/2)
            
            theta=THETA[c]#2*pi*np.random.uniform(0,1)
            signal_recfx,signal_recfy = cos_m(-theta)*signal_recfx + sin_m(-theta)*signal_recfy,-sin_m(-theta)*signal_recfx + cos_m(-theta)*signal_recfy
            
            ##############################################
        indx = -2*j-2
        for jj in range(l_span[indx]):
            c -= 1
            tau = TAU[c] #sqrt((PMD*1e-12)**2*(step_s*10**(-3)))*np.random.normal()
            signal_recfx = signal_recfx*(np.exp(-1j*w*tau/2))
            signal_recfy = signal_recfy*(np.exp(1j*w*tau/2))
            
            phi=PHI[c]#2*pi*np.random.uniform(0,1)
            signal_recfx = signal_recfx*np.exp(-1j*phi/2)
            signal_recfy = signal_recfy*np.exp(1j*phi/2)
            
            theta=THETA[c]#2*pi*np.random.uniform(0,1)
            signal_recfx,signal_recfy = cos_m(-theta)*signal_recfx + sin_m(-theta)*signal_recfy,-sin_m(-theta)*signal_recfx + cos_m(-theta)*signal_recfy
            
    signal_recx=np.fft.ifft(signal_recfx)
    signal_recy=np.fft.ifft(signal_recfy)
    return signal_recx,signal_recy

def t2f(t):
    dt = t[1]-t[0]
    F = 1/dt;
    f = np.linspace(-F/2,F/2,len(t),endpoint=False)
    return f

def DBP_dpol(signal,l_f,n_span,n_seg,gamma,alpha_db,beta2,dt):
    #Noise addition
    alpha=(log(10)/10)*(10**(-3))*(alpha_db)
    l_span=(l_f/n_span)
    signal_recx,signal_recy=signal[0],signal[1]
    l_s=int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    step_s=(l_f/((n_seg)*(n_span)))
    dis=np.exp(-(1j/2)*(beta2)*(w**2)*step_s)#dispersion.
    for j in range(n_span):
        for jj in range(n_seg):
            signal_recfx=fft(signal_recx)
            signal_recfy=fft(signal_recy)
            #add dispersion 
            signal_recfx=signal_recfx*dis
            signal_recfy=signal_recfy*dis

            signal_recx=ifft(signal_recfx)
            signal_recy=ifft(signal_recfy)
        
            signal_recx=signal_recx*np.exp(1j*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s)
            signal_recy=signal_recy*np.exp(1j*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s)
    return signal_recx,signal_recy

def channel2(signal,B,dt,h,f0,PMD,theta,phi):
    signal_recx,signal_recy=signal[0],signal[1]
    l_s=int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    signal_recfx=fft(signal_recx)
    signal_recfy=fft(signal_recy)
    signal_recfx_i= cos_m(theta)*signal_recfx + sin_m(theta)*signal_recfy
    signal_recfy_i= -sin_m(theta)*signal_recfx + cos_m(theta)*signal_recfy
    signal_recfx_i=signal_recfx_i*np.exp(1j*phi/2)
    signal_recfy_i=signal_recfy_i*np.exp(-1j*phi/2)
    #add pmd
    tau =18e-12
    signal_recfx_i=signal_recfx_i*(np.exp(1j*w*tau/2))
    signal_recfy_i=signal_recfy_i*(np.exp(-1j*w*tau/2))
    signal_recxx=ifft(signal_recfx_i)
    signal_recyy=ifft(signal_recfy_i)
    return signal_recxx,signal_recyy

def RRC(signal,B,dt,rof):
    sx,sy=signal[0],signal[1]
    l_s=int(len(sx))
    f=1/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    # H=1+cos(pi/(rof*B)*(abs(f)-(1-rof)*B/2))
    return f

def cmd(signal,l_f,beta2,dt):
    signal_recx,signal_recy=signal[0],signal[1]
    l_s=int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * fftshift(np.arange(-l_s/2,l_s/2))
    dis_com=np.exp(-(1j/2)*(-beta2)*(w**2)*l_f)#dispersion.
    signal_recfx=fft(signal_recx)*dis_com
    signal_recfy=fft(signal_recy)*dis_com
    signal_recx=ifft(signal_recfx)
    signal_recy=ifft(signal_recfy)
    return signal_recx,signal_recy

def CMA_RDEM(signal,Nsps,N_s,N_taps,p):
    s_inx=signal[0]/sqrt(p)
    s_iny=signal[1]/sqrt(p)
    l_s=len(s_inx)
    mu=0.001
    itr=10000 # training symbols.
    th=5000   # 
    # R1=abs(1/3+1j/3)
    # R2=abs(1/3+1j)
    # R3=abs(1+1j)
    R1=0.4470035621347161
    R2=0.9995303606046092
    R3=1.3410107005462
    g1=((R1)+(R2))/2
    g3=((R2)+(R3))/2
###############################################################################
#Convergance of the X_pol
    F_1=np.zeros((N_taps))+0j#inpx__outx
    F_2=np.zeros((N_taps))+0j#inpy__outx
    F_1[int(floor(N_taps/2))]=1+0j
    #CMA
    for l in range(th):
      idx_s=l*(Nsps)
      s_x=s_inx[idx_s:idx_s+N_taps]
      s_y=s_iny[idx_s:idx_s+N_taps]
      st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
      e_x=2*(abs(st_outx)**2-sqrt(2))*st_outx
      F_1=F_1-mu*e_x*np.conjugate(s_x)
      F_2=F_2-mu*e_x*np.conjugate(s_y)
#*************************************************************
#RDE
    
    for l in range(th,itr):  
       idx_s=l*(Nsps)
       s_x=s_inx[idx_s:idx_s+N_taps]
       s_y=s_iny[idx_s:idx_s+N_taps]
       st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
#*************************************************************
       if (abs(st_outx)<g1):
         Rs_x=R1**2       
       elif (abs(st_outx)>g3):
         Rs_x=R3**2
       else:
         Rs_x=R2**2 

       e_x=2*(abs(st_outx)**2-Rs_x)*st_outx
       F_1=F_1-mu*e_x*np.conjugate(s_x)
       F_2=F_2-mu*e_x*np.conjugate(s_y)   
###############################################################################
#Convergance for X and Y pol.
    F_4=np.conjugate(np.flip(F_1))
    F_3=-1*np.conjugate(np.flip(F_2))
#******************************************************************************
    for l in range(th):
      idx_s=l*(Nsps)
      s_x=s_inx[idx_s:idx_s+N_taps]
      s_y=s_iny[idx_s:idx_s+N_taps]
      st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
      st_outy=np.dot(F_3,s_x)+np.dot(F_4,s_y)
      e_x=2*(abs(st_outx)**2-sqrt(2))*st_outx
      e_y=2*(abs(st_outy)**2-sqrt(2))*st_outy
      F_1=F_1-mu*e_x*np.conjugate(s_x)
      F_2=F_2-mu*e_x*np.conjugate(s_y)
      F_3=F_3-mu*e_y*np.conjugate(s_x)
      F_4=F_4-mu*e_y*np.conjugate(s_y)
#*************************************************************
#RDE
    for l in range(th,itr):  
       idx_s=l*(Nsps)
       s_x=s_inx[idx_s:idx_s+N_taps]
       s_y=s_iny[idx_s:idx_s+N_taps]
       st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
       st_outy=np.dot(F_3,s_x)+np.dot(F_4,s_y)
#*************************************************************
       if (abs(st_outx)<g1):
         Rs_x=R1**2       
       elif (abs(st_outx)>g3):
         Rs_x=R3**2
       else:
         Rs_x=R2**2 
       if (abs(st_outy)<g1):
         Rs_y=R1**2       
       elif (abs(st_outy)>g3):
         Rs_y=R3**2
       else:
         Rs_y=R2**2 
       
       e_x=2*(abs(st_outx)**2-Rs_x)*st_outx
       e_y=2*(abs(st_outy)**2-Rs_y)*st_outy
       F_1=F_1-mu*e_x*np.conjugate(s_x)
       F_2=F_2-mu*e_x*np.conjugate(s_y)
       F_3=F_3-mu*e_y*np.conjugate(s_x)
       F_4=F_4-mu*e_y*np.conjugate(s_y)
    
    limit=N_s-N_taps
    s_outx=np.zeros((1,limit))+0j
    s_outy=np.zeros((1,limit))+0j
    for l in range (limit):
         idx_s=l*2
         s_x=s_inx[idx_s:idx_s+N_taps]
         s_y=s_iny[idx_s:idx_s+N_taps]
         s_outx[0,l]=np.dot(F_1,s_x)+np.dot(F_2,s_y)
         s_outy[0,l]=np.dot(F_3,s_x)+np.dot(F_4,s_y)
         
    sx_out=s_outx[0]*sqrt(p)
    sy_out=s_outy[0]*sqrt(p)  
    return sx_out,sy_out
    

def CMA_RDE(signal,Nsps,N_s,N_taps,p):
    s_inx=signal[0]/sqrt(p)
    s_iny=signal[1]/sqrt(p)
    l_s=len(s_inx)   
#*****************************************************************************************
#initilize the weights of the four FIR filters
    F_1=np.zeros((N_taps))+0j#inpx__outx
    F_2=np.zeros((N_taps))+0j#inpy__outx
    F_3=np.zeros((N_taps))+0j#inpx__outy
    F_4=np.zeros((N_taps))+0j #inp_y__outy
    F_1[int(floor(N_taps/2))]=1+0j
    F_4[int(floor(N_taps/2))]=1+0j
    mu=0.001
#***********************************************************
    itr=10000 #symbols
    th=5000
#***********************************************************
#CMA
    for l in range(th):
      idx_s=l*(Nsps)
      s_x=s_inx[idx_s:idx_s+N_taps]
      s_y=s_iny[idx_s:idx_s+N_taps]
      st_outx=np.vdot(F_1,s_x)+np.vdot(F_2,s_y)
      st_outy=np.vdot(F_3,s_x)+np.vdot(F_4,s_y)
      e_x=(1-abs(st_outx)**2)
      e_y=(1-abs(st_outy)**2)
      F_1=F_1+mu*e_x*(s_x)*np.conjugate(st_outx)
      F_2=F_2+mu*e_x*(s_y)*np.conjugate(st_outx)
      F_3=F_3+mu*e_y*(s_x)*np.conjugate(st_outy)
      F_4=F_4+mu*e_y*(s_y)*np.conjugate(st_outy)
#*************************************************************
#RDE
    R1=0.4470035621347161
    R2=0.9995303606046092
    R3=1.3410107005462
    g1=((R1)+(R2))/2
    g3=((R2)+(R3))/2
    
    for l in range(th,itr):  
       idx_s=l*(Nsps)
       s_x=s_inx[idx_s:idx_s+N_taps]
       s_y=s_iny[idx_s:idx_s+N_taps]
       st_outx=np.vdot(F_1,s_x)+np.vdot(F_2,s_y)
       st_outy=np.vdot(F_3,s_x)+np.vdot(F_4,s_y)
#*************************************************************
       if (abs(st_outx)<g1):
         Rs_x=R1**2       
       elif (abs(st_outx)>g3):
         Rs_x=R3**2
       else:
         Rs_x=R2**2 
       if (abs(st_outy)<g1):
         Rs_y=R1**2       
       elif (abs(st_outy)>g3):
         Rs_y=R3**2
       else:
         Rs_y=R2**2 
       
       e_x=(Rs_x-abs(st_outx)**2)
       e_y=(Rs_y-abs(st_outy)**2)
       F_1=F_1+mu*e_x*(s_x)*np.conjugate(st_outx)
       F_2=F_2+mu*e_x*(s_y)*np.conjugate(st_outx)
       F_3=F_3+mu*e_y*(s_x)*np.conjugate(st_outy)
       F_4=F_4+mu*e_y*(s_y)*np.conjugate(st_outy)
    
    limit=N_s-N_taps
    s_outx=np.zeros((1,limit))+0j
    s_outy=np.zeros((1,limit))+0j
    for l in range (limit):
         idx_s=l*2
         s_x=s_inx[idx_s:idx_s+N_taps]
         s_y=s_iny[idx_s:idx_s+N_taps]
         s_outx[0,l]=np.vdot(F_1,s_x)+np.vdot(F_2,s_y)
         s_outy[0,l]=np.vdot(F_3,s_x)+np.vdot(F_4,s_y)
#****************************************************************************************
    sx_out=s_outx[0]*sqrt(p)
    sy_out=s_outy[0]*sqrt(p)  
    return sx_out,sy_out
#****************************************************************************************************
def CMA_RDE2(signal,Nsps,N_s,N_taps,p):
    s_inx=signal[0]/sqrt(p)
    s_iny=signal[1]/sqrt(p)
    l_s=len(s_inx)
#*****************************************************************************************
#initilize the weights of the four FIR filters
    F_1=np.zeros((N_taps))+0j#inpx__outx
    F_2=np.zeros((N_taps))+0j#inpy__outx
    F_3=np.zeros((N_taps))+0j#inpx__outy
    F_4=np.zeros((N_taps))+0j #inp_y__outy
    F_1[int(floor(N_taps/2))+1]=1+0j
    F_4[int(floor(N_taps/2))+1]=1+0j
    mu=0.001
#***********************************************************
    itr=10000 #symbols
    th=5000
#***********************************************************
#CMA
    th=5000
    for l in range(th):
      idx_s=l*(Nsps)
      s_x=s_inx[idx_s:idx_s+N_taps]
      s_y=s_iny[idx_s:idx_s+N_taps]
      st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
      st_outy=np.dot(F_3,s_x)+np.dot(F_4,s_y)
      e_x=2*(abs(st_outx)**2-1)*st_outx
      e_y=2*(abs(st_outy)**2-1)*st_outy
      F_1=F_1-mu*e_x*np.conjugate(s_x)
      F_2=F_2-mu*e_x*np.conjugate(s_y)
      F_3=F_3-mu*e_y*np.conjugate(s_x)
      F_4=F_4-mu*e_y*np.conjugate(s_y)
#*************************************************************
#RDE
    R1=0.4470035621347161
    R2=0.9995303606046092
    R3=1.3410107005462
    g1=((R1)+(R2))/2
    g3=((R2)+(R3))/2
    
    for l in range(th,itr):  
       idx_s=l*(Nsps)
       s_x=s_inx[idx_s:idx_s+N_taps]
       s_y=s_iny[idx_s:idx_s+N_taps]
       st_outx=np.dot(F_1,s_x)+np.dot(F_2,s_y)
       st_outy=np.dot(F_3,s_x)+np.dot(F_4,s_y)
#*************************************************************
       if (abs(st_outx)<g1):
         Rs_x=R1**2       
       elif (abs(st_outx)>g3):
         Rs_x=R3**2
       else:
         Rs_x=R2**2 
       if (abs(st_outy)<g1):
         Rs_y=R1**2       
       elif (abs(st_outy)>g3):
         Rs_y=R3**2
       else:
         Rs_y=R2**2 
       
       e_x=2*(abs(st_outx)**2-Rs_x)*st_outx
       e_y=2*(abs(st_outy)**2-Rs_y)*st_outy
       F_1=F_1-mu*e_x*np.conjugate(s_x)
       F_2=F_2-mu*e_x*np.conjugate(s_y)
       F_3=F_3-mu*e_y*np.conjugate(s_x)
       F_4=F_4-mu*e_y*np.conjugate(s_y)
    
    limit=N_s-N_taps
    s_outx=np.zeros((1,limit))+0j
    s_outy=np.zeros((1,limit))+0j
    for l in range (limit):
         idx_s=l*2
         s_x=s_inx[idx_s:idx_s+N_taps]
         s_y=s_iny[idx_s:idx_s+N_taps]
         s_outx[0,l]=np.dot(F_1,s_x)+np.dot(F_2,s_y)
         s_outy[0,l]=np.dot(F_3,s_x)+np.dot(F_4,s_y)
#****************************************************************************************
    sx_out=s_outx[0]*sqrt(p)
    sy_out=s_outy[0]*sqrt(p)  
    return sx_out,sy_out
#****************************************************************************************************

def pol_dmux(sym_r,sym_t):
    r_x=sym_r[0]
    r_y=sym_r[1]
    t_x=sym_t[0]
    t_y=sym_t[1]
    #****************************************************************************************
    # calculate the crros corelation between the recived symbols and the transmitted symbols. 
    n=len(r_x)-1
    ax=np.arange(-n,n+1,1)
    cor_xx=abs((np.correlate(r_x,t_x, "full")))
    cor_xy=abs((np.correlate(r_x,t_y, "full")))
    cor_yx=abs((np.correlate(r_y,t_x, "full")))
    cor_yy=abs((np.correlate(r_y,t_y, "full")))
    mcor_xx=max((cor_xx))
    i_xx=int(ax[np.argmax(cor_xx)])
    mcor_xy=max(cor_xy)
    i_xy=int(ax[np.argmax(cor_xy)])
    mcor_yx=max(cor_yx)
    i_yx=int(ax[np.argmax(cor_yx)])
    mcor_yy=max(cor_yy)
    i_yy=int(ax[np.argmax(cor_yy)])
    #****************************************************************************************
    if (mcor_xx>mcor_xy):
       sout_x=np.roll(r_x,(-i_xx))   
    if (mcor_xx<=mcor_xy):
       sout_x=np.roll(r_y,(-i_xy)) 
       
    if (mcor_yy>mcor_yx):
       sout_y=np.roll(r_y,(-i_yy))   
    if (mcor_yy<=mcor_yx):
       sout_y=np.roll(r_x,(-i_yx))
       

    return sout_x,sout_y

def cpe(sym_r,sym_t,s_wind,p):
    sr_x=sym_r[0]
    sr_y=sym_r[1]
    st_x=sym_t[0]
    st_y=sym_t[1]
    B=32 # number of angles to be tested.
    b=0
    e=1
    theta_t=(np.arange(start=b, stop=e, step=(1/B))*(pi/2))-(pi/4)
    l_theta=len(theta_t)
    l_sr=len(sr_x)
    s_begin=0
    d_x=np.zeros((l_sr,l_theta))
    d_y=np.zeros((l_sr,l_theta))
    for l in range(s_begin,l_sr):
        srx_test=sr_x[l]*exp(-1j*theta_t)#applay phase shifts to the lth symbol.
        srx_test_d=deQAM16(srx_test,p)#demodulate.
        sry_test=sr_y[l]*exp(-1j*theta_t)#applay phase shifts to the lth symbol.
        sry_test_d=deQAM16(sry_test,p)#demodulate.
        #*******************************************
        #calculate the distances.
        d_x[l]=abs(srx_test-srx_test_d)**2#for x_pol.
        d_y[l]=abs(sry_test-sry_test_d)**2#for y_pol.
    #*******************************************
    #avg distances.
    dx_avg=np.zeros((l_sr,l_theta))
    dy_avg=np.zeros((l_sr,l_theta))
    theta_estx=np.zeros((1,l_sr))
    theta_esty=np.zeros((1,l_sr))
    lim1=int((s_wind/2))
    lim2=int(l_sr-(s_wind/2))
    for l in range(lim1,lim2):
      for k in range(l_theta):
        dx_avg[l,k]=mean(d_x[l-lim1:l+lim1,k])
        dy_avg[l,k]=mean(d_y[l-lim1:l+lim1,k])
    ind_x=np.where(dx_avg[l]==min(dx_avg[l]))
    theta_estx[0,l]=theta_t[int(ind_x[0])]
    ind_y=np.where(dy_avg[l]==min(dy_avg[l]))
    theta_esty[0,l]=theta_t[int(ind_y[0])]
    # #******************************************************************
    # # Cs mitigation 
    phase_trx=diff(theta_estx[0])
    phase_try=diff(theta_esty[0])
    phase_shiftx=np.cumsum((phase_trx<(-pi/4))*(-pi/2)+(phase_trx>(pi/4))*(pi/2))
    phase_shifty=np.cumsum((phase_try<(-pi/4))*(-pi/2)+(phase_try>(pi/4))*(pi/2))
    thet_x=theta_estx[0]-phase_shiftx
    thet_y=theta_esty[0]-phase_shifty
    s_outtx=np.multiply(sr_x,exp(-1j*thet_x))
    s_outty=np.multiply(sr_y,exp(-1j*thet_y))
# #***********************************************************************
    
#***************************************************************************
# #seconed cpe stage
    v_1x =s_outtx[lim1:lim2]
    v_2x =st_x[lim1:lim2]
    ratio_x= np.divide(v_1x,v_2x)
    anglex = cmath.phase(mean(ratio_x))
    ss_x=np.multiply(s_outtx, exp(-1j*anglex))
    v_1y =s_outty[lim1:lim2]
    v_2y =st_y[lim1:lim2]
    ratio_y= np.divide(v_1y,v_2y)
    angley = cmath.phase(mean(ratio_y))
    ss_y=np.multiply(s_outty, exp(-1j*angley))
    #*******************************************************************
    return ss_x,ss_y

def diff(seq):
    l_seq=len(seq)
    seq_d=np.zeros((1,l_seq))
    for l in range(l_seq):
        if (l==0):
            seq_d[0,l]=seq[l]
        else:
            seq_d[0,l]=seq[l]-seq[l-1]
    return seq_d

def d2b(x,k):
    bi=list(bin(x)[2:])
    l=len(bi)
    for i in range(l):
        bi[i]=float(bi[i])
    if l!=k:
        biry=list(zeros(k-l))
        biry.extend(bi)
    else:
        biry=bi
    return biry

def err_count(symb_est,symb_trs):
    return np.sum(np.abs(symb_est-symb_trs))

def pha(v_1,v_2):
    l_v=len(v_1)
    theta=np.zeros((1,l_v))
    div=np.divide(v_1,v_2)
    for l in range(l_v):
        theta[0,l]=cmath.phase(div[l])
    return theta
def noiseadd(B,alpha_db,NF_db,h,f0,N_s,t,l_span):
    alpha=(log(10)/10)*(10**(-3))*(alpha_db)
    G_db=alpha_db*(l_span)
    g=10**(G_db/10)
    nf=10**(NF_db/10)
    var=10*(g-1)*h*f0*(nf/2)*(B)#noise variance.
    noise_x=np.sqrt(var/2)*(np.random.randn(N_s)+1j*np.random.randn(N_s))
    noisex_added=mod(t,noise_x,B)
    noise_y=np.sqrt(var/2)*(np.random.randn(N_s)+1j*np.random.randn(N_s))
    noisey_added=mod(t,noise_y,B)
    
    return noisex_added,noisey_added

def AWGN(x,var,N_span):
    l_s=len(x)
    y=x
    for l in range (10):
      y=y+sqrt(var/2)*(np.random.randn(l_s)+1j*np.random.randn(l_s))
    return y

def data_prep(s_wind,inp_signal,s,N_blocks,N_s,N_d):
    tra_blocks=int(N_blocks/2)
    val_blocks=int(N_blocks/2)
    N_ex_tra=(tra_blocks)*(N_s-(s_wind+N_d-1))#Number of training examples.
    N_ex_val=(val_blocks)*(N_s-(s_wind+N_d-1))#Number of testing examples.
    L_ex=(N_d+s_wind) #length of each example per pol.
    inp_Nnet_x=np.zeros((N_ex_tra,L_ex))+0j
    out_Nnet_x=np.zeros((N_ex_tra,N_d))+0j
    inp_Nnet_y=np.zeros((N_ex_tra,L_ex))+0j
    out_Nnet_y=np.zeros((N_ex_tra,N_d))+0j
    #************************************************
    val_inp_Nnet_x=np.zeros((N_ex_val,L_ex))+0j
    val_out_Nnet_x=np.zeros((N_ex_val,N_d))+0j
    val_inp_Nnet_y=np.zeros((N_ex_val,L_ex))+0j
    val_out_Nnet_y=np.zeros((N_ex_val,N_d))+0j
    #*************************************************
    count=0
    for k in range(N_blocks):
      for l in range(N_s-(s_wind+N_d-1)): 
        if (k<tra_blocks):
          inp_Nnet_x[count]=inp_signal[k,0,l:l+int(s_wind/2)+N_d+int(s_wind/2)]
          inp_Nnet_y[count]=inp_signal[k,1,l:l+int(s_wind/2)+N_d+int(s_wind/2)]
          out_Nnet_x[count]=s[k,0,int(s_wind/2)+l:l+int(s_wind/2)+N_d]
          out_Nnet_y[count]=s[k,1,int(s_wind/2)+l:l+int(s_wind/2)+N_d]
        elif (k>tra_blocks):
          val_Nnet_inp_x[l]=inp_signal[k,0,l:l+int(s_wind/2)+N_d+int(s_wind/2)]
          val_Nnet_out_x[l]=s[k,0,int(s_wind/2)+l:l+int(s_wind/2)+N_d]
          val_Nnet_inp_y[l]=inp_signal[k,1,l:l+int(s_wind/2)+N_d+int(s_wind/2)]
          val_Nnet_out_y[l]=s[k,1,int(s_wind/2)+l:l+int(s_wind/2)+N_d]
        count=count+1
    inp_Nnet=np.concatenate((inp_Nnet_x,inp_Nnet_y),axis=1)#training of the Nnet(Input layer)
    out_Nnet=out_Nnet_x#training of the Nnet(output layer)
    val_Nnet_inp=np.concatenate((val_Nnet_inp_x,val_Nnet_inp_y),axis=1)#val of the Nnet(Input layer)
    val_Nnet_out=val_Nnet_out_x #val of the Nnet(output layer)
  #****************************************************************************************************
  #data reshape
    val_Nnet_inp_resh=np.zeros((N_ex_val,2*2*L_ex))
    val_Nnet_out_resh=np.zeros((N_ex_val,2*N_d))
    inp_Nnet_resh=np.zeros((N_ex_tra,2*2*L_ex))
    out_Nnet_resh=np.zeros((N_ex_tra,2*N_d))
    for l in range (N_ex_tra):
      inp_Nnet_resh[l]=resh_data(inp_Nnet[l],'real')
      out_Nnet_resh[l]=resh_data(out_Nnet[l],'real')
    for l in range(N_ex_val):
      val_Nnet_inp_resh[l]=resh_data(val_Nnet_inp[l],'real')
      val_Nnet_out_resh[l]=resh_data(val_Nnet_out[l],'real')
    return inp_Nnet_resh,out_Nnet_resh,val_Nnet_inp_resh,val_Nnet_out_resh

def resh_data(Data,mapping):
    L_data=len(Data)
    count=0
    if mapping =="real":
      X=np.zeros(2*L_data)
      for l in range(L_data):
         X[count]=real(Data[l])
         count=count+1
         X[count]=imag(Data[l])
         count=count+1
    if mapping=="complex":
      X=np.zeros(int(L_data/2))+0j
      for l in range(int(L_data/2)):
          X[l]=Data[count]+1j*Data[count+1]
          count=count+2
    return X

def matchfil (s,B,t,Ns):
    l1 = -int(floor(Ns/2))
    l2 = int(ceil(Ns/2)) 
    dt = t[1]-t[0]
    R=np.zeros(Ns)+0j
    norm=np.linalg.norm(np.sinc(B*t))
    for l in range(l1,l2):
      bas=(np.sinc(B*t-l))
      R[l+int(floor(Ns/2))] = sum(s*((np.sinc(B*t-l)*B*dt)))
    return R


def DBP_DM(signal,l_span,gamma,alpha_db,beta2,dt,B,NF_db,h,f0,N_s,t,steps):
    signal_recx,signal_recy = signal[0],signal[1]
    l_s = int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * np.fft.fftshift(np.arange(-l_s/2,l_s/2))
    n_span = int(len(l_span)/2)
    if steps<1:
        n_span = int(len(l_span)/2 * steps)
        l_span2 = np.zeros(n_span*2)
        for i in range(n_span):
            l_span2[2*i] = np.sum(l_span[np.array([2*j for j in range(i*int(1/steps),(i+1)*int(1/steps))])])
            l_span2[2*i+1] = np.sum(l_span[np.array([2*j+1 for j in range(i*int(1/steps),(i+1)*int(1/steps))])])
        l_span = l_span2
        steps = 1
    for j in range(n_span):
        step_s1 = l_span[-2*j-2]/steps * 1e3
        step_s2 = l_span[-2*j-1]/steps * 1e3
        cf = np.exp((np.log(10)/10)*alpha_db*np.linspace(1/(2*steps),1-1/(2*steps),steps,endpoint=True)*l_span[-2*j-2])
        cf = np.ones(steps)
        cf = cf/np.mean(cf)
        signal_recfx=np.fft.fft(signal_recx)
        signal_recfy=np.fft.fft(signal_recy)
        dis = np.exp((1j/2)*beta2[-2*j-1]*(w**2)*l_span[-2*j-1] * 1e3)
        signal_recfx_i=signal_recfx*dis
        signal_recfy_i=signal_recfy*dis
        signal_recx=np.fft.ifft(signal_recfx_i)
        signal_recy=np.fft.ifft(signal_recfy_i)
        halfdis = np.exp((1j/2)*beta2[-2*j-2]*(w**2)*step_s1/2)
        
        W = dft(l_s, scale='sqrtn')
        Winv = 1/(l_s*W)
        D = np.diagflat(np.exp(1j/2*beta2[-2*j-2]*(w**2)*step_s1/2))
        halfLin_op = np.dot(Winv,np.dot(D,W))
        
        D = np.diagflat(np.exp(1j/2*beta2[-2*j-2]*(w**2)*step_s1))
        fullLin_op = np.dot(Winv,np.dot(D,W))
        
        # for jj in range(steps):
        #     signal_recfx=np.fft.fft(signal_recx)
        #     signal_recfy=np.fft.fft(signal_recy)
        #     signal_recfx_i=signal_recfx*halfdis
        #     signal_recfy_i=signal_recfy*halfdis
        #     signal_recx=np.fft.ifft(signal_recfx_i)
        #     signal_recy=np.fft.ifft(signal_recfy_i)
        #     signal_recx,signal_recy=signal_recx*np.exp(-1j*cf[jj]*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s1),signal_recy*np.exp(-1j*cf[jj]*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s1)
        #     signal_recfx=np.fft.fft(signal_recx)
        #     signal_recfy=np.fft.fft(signal_recy)
        #     signal_recfx_i=signal_recfx*halfdis
        #     signal_recfy_i=signal_recfy*halfdis
        #     signal_recx=np.fft.ifft(signal_recfx_i)
        #     signal_recy=np.fft.ifft(signal_recfy_i)
        
        
        # for jj in range(steps):
        #     signal_recx = np.dot(signal_recx,halfLin_op)
        #     signal_recy = np.dot(signal_recy,halfLin_op)
        #     signal_recx,signal_recy = signal_recx*np.exp(-1j*cf[jj]*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s1),signal_recy*np.exp(-1j*cf[jj]*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s1)
        #     signal_recx = np.dot(signal_recx,halfLin_op)
        #     signal_recy = np.dot(signal_recy,halfLin_op)
        
        kernel = halfLin_op[256-1]
        for jj in range(steps):
            concX = np.concatenate([signal_recx[int(l_s/2):],signal_recx,signal_recx[:int(l_s/2)-1],])
            concY = np.concatenate([signal_recy[int(l_s/2):],signal_recy,signal_recy[:int(l_s/2)-1],])
            signal_recx = np.convolve(concX,kernel,mode='valid')
            signal_recy = np.convolve(concY,kernel,mode='valid')
            signal_recx,signal_recy = signal_recx*np.exp(-1j*cf[jj]*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s1),signal_recy*np.exp(-1j*cf[jj]*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s1)
            concX = np.concatenate([signal_recx[int(l_s/2):],signal_recx,signal_recx[:int(l_s/2)-1],])
            concY = np.concatenate([signal_recy[int(l_s/2):],signal_recy,signal_recy[:int(l_s/2)-1],])
            signal_recx = np.convolve(concX,kernel,mode='valid')
            signal_recy = np.convolve(concY,kernel,mode='valid')
    return signal_recx,signal_recy,halfLin_op,fullLin_op

def DBP_DM2(signal,l_span,gamma,alpha_db,beta2,dt,B,NF_db,h,f0,N_s,t,steps):
    signal_recx,signal_recy = signal[0],signal[1]
    l_s = int(len(signal_recx))
    w = 2*np.pi/(l_s*dt) * np.fft.fftshift(np.arange(-l_s/2,l_s/2))
    n_span = int(len(l_span)/2)
    if steps<1:
        n_span = int(len(l_span)/2 * steps)
        l_span2 = np.zeros(n_span*2)
        for i in range(n_span):
            l_span2[2*i] = np.sum(l_span[np.array([2*j for j in range(i*int(1/steps),(i+1)*int(1/steps))])])
            l_span2[2*i+1] = np.sum(l_span[np.array([2*j+1 for j in range(i*int(1/steps),(i+1)*int(1/steps))])])
        l_span = l_span2
        steps = 1
    for j in range(n_span):
        step_s1 = l_span[-2*j-2]/steps * 1e3
        step_s2 = l_span[-2*j-1]/steps * 1e3
        cf = np.exp((np.log(10)/10)*alpha_db*np.linspace(1/(2*steps),1-1/(2*steps),steps,endpoint=True)*l_span[-2*j-2])
        cf = np.ones(steps)
        cf = cf/np.mean(cf)
        signal_recfx=np.fft.fft(signal_recx)
        signal_recfy=np.fft.fft(signal_recy)
        dis = np.exp((1j/2)*beta2[-2*j-1]*(w**2)*l_span[-2*j-1] * 1e3)
        signal_recfx_i=signal_recfx*dis
        signal_recfy_i=signal_recfy*dis
        signal_recx=np.fft.ifft(signal_recfx_i)
        signal_recy=np.fft.ifft(signal_recfy_i)
        halfdis = np.exp((1j/2)*beta2[-2*j-2]*(w**2)*step_s1/2)
        for jj in range(steps):
            signal_recfx=np.fft.fft(signal_recx)
            signal_recfy=np.fft.fft(signal_recy)
            signal_recfx_i=signal_recfx*halfdis
            signal_recfy_i=signal_recfy*halfdis
            signal_recx=np.fft.ifft(signal_recfx_i)
            signal_recy=np.fft.ifft(signal_recfy_i)
            signal_recx=signal_recx*np.exp(-1j*cf[jj]*gamma*((abs(signal_recx))**2+(2/3)*(abs(signal_recy))**2)*step_s1)
            signal_recy=signal_recy*np.exp(-1j*cf[jj]*gamma*((abs(signal_recy))**2+(2/3)*(abs(signal_recx))**2)*step_s1)
            signal_recfx=np.fft.fft(signal_recx)
            signal_recfy=np.fft.fft(signal_recy)
            signal_recfx_i=signal_recfx*halfdis
            signal_recfy_i=signal_recfy*halfdis
            signal_recx=np.fft.ifft(signal_recfx_i)
            signal_recy=np.fft.ifft(signal_recfy_i)
    return signal_recx,signal_recy

def sym_error(A,B,bins):
    A1 = np.digitize(A[0].flatten(),bins) 
    B1 = np.digitize(B[0].flatten(),bins)
    A2 = np.digitize(A[1].flatten(),bins)
    B2 = np.digitize(B[1].flatten(),bins)
    errors = np.sum(np.logical_or(A1 != B1, A2 != B2))
    return errors

def examples(A,trim,length,mode=1):
    n = np.shape(A)[1]
    A = A[:,trim:n-trim]
    n = np.shape(A)[1]
    indx = np.arange(int(length/2),n-int(length/2),2)
    N_examples = len(indx)
    indxmat = np.repeat(indx.reshape(N_examples,1),length,axis = 1) + np.repeat(np.arange(int(-length/2),int(length/2),dtype='int').reshape(1,length),N_examples,axis=0)
    Split = np.expand_dims(A[:,indxmat],3)
    if mode == 1:
        Out = [np.real(Split[0]),
               np.imag(Split[0]),
               np.real(Split[1]),
               np.imag(Split[1])]
    elif mode == 2:
        Out = [np.real(Split[0,:,::2,:]),
               np.imag(Split[0,:,::2,:]),
               np.real(Split[1,:,::2,:]),
               np.imag(Split[1,:,::2,:])]
    return Out,indx+trim

def examples2(A,trim,length):
    n = np.shape(A)[1]
    A = A[:,trim:n-trim]
    n = np.shape(A)[1]
    indx = np.arange(int(length/2),n-int(length/2),1)
    N_examples = len(indx)
    indxmat = np.repeat(indx.reshape(N_examples,1),length,axis = 1) + np.repeat(np.arange(int(-length/2),int(length/2),dtype='int').reshape(1,length),N_examples,axis=0)
    Split = A[:,indxmat]
    Out = np.concatenate([np.real(Split[0]),
            np.imag(Split[0]),
            np.real(Split[1]),
            np.imag(Split[1])],axis=1)
    return Out,indx+trim

def real_100km(shape, dtype=None):
    d = shape[0]
    weights = np.zeros(d)
    weights[int(d/2)-26:int(d/2)+27] = np.array([-0.00540791405364871025085449218750,0.02034036070108413696289062500000,-0.03528429567813873291015625000000,0.02716069296002388000488281250000,0.00879955478012561798095703125000,-0.03526334092020988464355468750000,0.01162899564951658248901367187500,0.03363601118326187133789062500000,-0.02135208435356616973876953125000,-0.04329593852162361145019531250000,0.03636013716459274291992187500000,0.06189768016338348388671875000000,-0.05368083342909812927246093750000,-0.08657134324312210083007812500000,0.04495610669255256652832031250000,0.09394728392362594604492187500000,0.08776285499334335327148437500000,-0.13320420682430267333984375000000,-0.14895382523536682128906250000000,-0.07638701796531677246093750000000,0.16162027418613433837890625000000,-0.00995401293039321899414062500000,0.13168819248676300048828125000000,0.19242160022258758544921875000000,0.21615140140056610107421875000000,-0.30833399295806884765625000000000,0.65761822462081909179687500000000,-0.30762404203414916992187500000000,0.21690790355205535888671875000000,0.19106659293174743652343750000000,0.12915414571762084960937500000000,-0.00171795638743788003921508789062,0.15227834880352020263671875000000,-0.07193777710199356079101562500000,-0.14836741983890533447265625000000,-0.13373097777366638183593750000000,0.08553527295589447021484375000000,0.09640448540449142456054687500000,0.04503166303038597106933593750000,-0.08748990297317504882812500000000,-0.05505323410034179687500000000000,0.06458797305822372436523437500000,0.03618578612804412841796875000000,-0.04635487496852874755859375000000,-0.01920285448431968688964843750000,0.03620111197233200073242187500000,0.00605986546725034713745117187500,-0.03214106708765029907226562500000,0.01093605440109968185424804687500,0.02202752046287059783935546875000,-0.03092652373015880584716796875000,0.01835860684514045715332031250000,-0.00499081658199429512023925781250]).reshape(64,1,1)
    return weights.reshape(d,1,1)

def imag_100km(shape, dtype=None):
    d = shape[0]
    weights = np.zeros(d)
    weights[int(d/2)-26:int(d/2)+27] = np.array([-0.00392722804099321365356445312500,0.01843907684087753295898437500000,-0.03993786498904228210449218750000,0.04683879017829895019531250000000,-0.01690536923706531524658203125000,-0.03462532535195350646972656250000,0.04911621659994125366210937500000,0.00729923555627465248107910156250,-0.06832975149154663085937500000000,0.02655940316617488861083984375000,0.07946358621120452880859375000000,-0.05653160437941551208496093750000,-0.09742600470781326293945312500000,0.07808642089366912841796875000000,0.10734537243843078613281250000000,-0.02226262725889682769775390625000,-0.11984155327081680297851562500000,-0.11817788332700729370117187500000,0.07922763377428054809570312500000,0.11261941492557525634765625000000,0.11296049505472183227539062500000,0.09647353738546371459960937500000,0.00766400014981627464294433593750,-0.04097148776054382324218750000000,-0.06165906041860580444335937500000,-0.09516382962465286254882812500000,-0.09760475158691406250000000000000,-0.08197638392448425292968750000000,-0.07132663577795028686523437500000,-0.04604784399271011352539062500000,0.01963827200233936309814453125000,0.09212937206029891967773437500000,0.10781759023666381835937500000000,0.11639987677335739135742187500000,0.08330686390399932861328125000000,-0.12379945069551467895507812500000,-0.12156739085912704467773437500000,-0.01449301280081272125244140625000,0.10266274958848953247070312500000,0.07541271299123764038085937500000,-0.09250590205192565917968750000000,-0.05783401802182197570800781250000,0.07800841331481933593750000000000,0.02657296881079673767089843750000,-0.06646459549665451049804687500000,0.00739280041307210922241210937500,0.04496945813298225402832031250000,-0.02903931774199008941650390625000,-0.01963740959763526916503906250000,0.04567129164934158325195312500000,-0.03721617162227630615234375000000,0.01651900447905063629150390625000,-0.00346782500855624675750732421875]).reshape(64,1,1)
    return weights.reshape(d,1,1)

def initR(shape, dtype=None):
    d = shape[0]
    weights = np.zeros(shape)
    weights[int(d/2),:,:] = 1
    return weights

def initI(shape, dtype=None):
    return np.zeros(shape)

def initR40km(shape, dtype=None):
    d = shape[0]
    weights = np.array([-0.009307014001977965, 0.009400817094015858, -0.009560369214226235, 0.009790736402823725, -0.010099536000355723, 0.010497547373606942, -0.010999640414612284, 0.011626172078273246, -0.012405098085690376, 0.013375210681094516, -0.014591194995820445, 0.01613168297995781, -0.018112307363750132, 0.02070705255180561, -0.024182788945607617, 0.028951904009235578, -0.03563783877052438, 0.045097207470348025, -0.05815966416293855, 0.07434677812572622, -0.08796059393337348, 0.08028995578258298, -0.015360644342422132, -0.12742580252117033, 0.25040017089546407, -0.1236113012843533, -0.1626896486848517, -0.0073628885255272655, 0.21476435602714972, 0.18059895112018162, 0.19450034381064693, 0.1246988287843987, 0.19450034381064651, 0.18059895112018176, 0.21476435602715047, -0.00736288852552619, -0.1626896486848518, -0.12361130128435294, 0.2504001708954633, -0.12742580252116928, -0.015360644342421292, 0.08028995578258362, -0.0879605939333733, 0.07434677812572757, -0.0581596641629392, 0.04509720747034916, -0.03563783877052313, 0.028951904009236272, -0.0241827889456077, 0.020707052551805986, -0.01811230736375056, 0.01613168297995769, -0.014591194995821101, 0.0133752106810945, -0.012405098085690085, 0.011626172078273902, -0.01099964041461032, 0.01049754737360851, -0.010099536000354478, 0.00979073640282343, -0.009560369214226674, 0.009400817094015045, -0.009307014001974945, 0.00927606090417073,]).reshape(64,1,1)
    return weights
    #
    
def initI40km(shape, dtype=None):
    d = shape[0]
    weights = np.array([0.0021599437107385137, -0.002177579688272925, 0.0022073346793677, -0.00224975181860477, 0.0023055877475032685, -0.002375794874619171, 0.0024614708283309107, -0.00256373615084277, 0.0026834600809079482, -0.0028206665196424067, 0.002973260416342176, -0.003134283856509276, 0.003285917779775288, -0.0033861068791902365, 0.003338123737576043, -0.0029203030968871896, 0.0016239571299040646, 0.001708900469581347, -0.009623372718363075, 0.02722883494044598, -0.062806496967325, 0.12307317503884826, -0.19258030072985088, 0.19856428509424834, -0.03931670401868009, -0.18762271576464823, 0.07820629733916644, 0.21607262875951827, 0.09277387588204344, -0.04126804262833497, -0.13253276664498165, -0.1444227613853347, -0.13253276664498254, -0.0412680426283355, 0.09277387588204263, 0.2160726287595186, 0.07820629733916806, -0.1876227157646489, -0.03931670401867969, 0.19856428509424728, -0.19258030072985122, 0.1230731750388489, -0.06280649696732528, 0.02722883494044609, -0.009623372718363797, 0.0017089004695815743, 0.0016239571299035416, -0.002920303096887113, 0.003338123737576258, -0.0033861068791918303, 0.003285917779775785, -0.0031342838565071667, 0.0029732604163424346, -0.002820666519643495, 0.002683460080907483, -0.0025637361508444386, 0.002461470828329795, -0.0023757948746182755, 0.002305587747504686, -0.0022497518186039935, 0.0022073346793691135, -0.0021775796882743787, 0.0021599437107388837, -0.0021541011697530114, ]).reshape(64,1,1)
    return weights
    #
    
def initR_40km(shape, dtype=None):
    return initR40km(shape, dtype)

def initI_40km(shape, dtype=None):
    d = shape[0]
    weights = -np.array([0.0021599437107385137, -0.002177579688272925, 0.0022073346793677, -0.00224975181860477, 0.0023055877475032685, -0.002375794874619171, 0.0024614708283309107, -0.00256373615084277, 0.0026834600809079482, -0.0028206665196424067, 0.002973260416342176, -0.003134283856509276, 0.003285917779775288, -0.0033861068791902365, 0.003338123737576043, -0.0029203030968871896, 0.0016239571299040646, 0.001708900469581347, -0.009623372718363075, 0.02722883494044598, -0.062806496967325, 0.12307317503884826, -0.19258030072985088, 0.19856428509424834, -0.03931670401868009, -0.18762271576464823, 0.07820629733916644, 0.21607262875951827, 0.09277387588204344, -0.04126804262833497, -0.13253276664498165, -0.1444227613853347, -0.13253276664498254, -0.0412680426283355, 0.09277387588204263, 0.2160726287595186, 0.07820629733916806, -0.1876227157646489, -0.03931670401867969, 0.19856428509424728, -0.19258030072985122, 0.1230731750388489, -0.06280649696732528, 0.02722883494044609, -0.009623372718363797, 0.0017089004695815743, 0.0016239571299035416, -0.002920303096887113, 0.003338123737576258, -0.0033861068791918303, 0.003285917779775785, -0.0031342838565071667, 0.0029732604163424346, -0.002820666519643495, 0.002683460080907483, -0.0025637361508444386, 0.002461470828329795, -0.0023757948746182755, 0.002305587747504686, -0.0022497518186039935, 0.0022073346793691135, -0.0021775796882743787, 0.0021599437107388837, -0.0021541011697530114, ]).reshape(64,1,1)
    return weights
    #
    
def initR20km(shape, dtype=None):
    d = shape[0]
    weights = np.array([0.002731697086262345, -0.0027535280138253283, 0.00279045754770182, -0.0028433314714873307, 0.002913394183431886, -0.0030023556941326684, 0.003112488453449033, -0.003246764252617055, 0.00340904683155649, -0.0036043641859913912, 0.0038392979734686236, -0.00412254955160801, 0.004465779885614787, -0.00488488698296195, 0.00540200594184215, -0.006048748032847582, 0.0068716566182296275, -0.00794182567855964, 0.009372771531279942, -0.011355675871669437, 0.014233527179324256, -0.018667309532551062, 0.026025938423879492, -0.039289441287882323, 0.06476607444341546, -0.11256427546016318, 0.17966286727997388, -0.17424202419764578, -0.10721997079823391, 0.32515112295663207, 0.22428303841152103, 0.24823624630273108, 0.22428303841152128, 0.3251511229566338, -0.1072199707982335, -0.17424202419764528, 0.17966286727997355, -0.11256427546016358, 0.06476607444341648, -0.03928944128788182, 0.02602593842387974, -0.018667309532550018, 0.014233527179324305, -0.01135567587166922, 0.009372771531281271, -0.00794182567855858, 0.006871656618229762, -0.006048748032847463, 0.005402005941843894, -0.004884886982961895, 0.004465779885614253, -0.0041225495516086915, 0.0038392979734671426, -0.003604364185991271, 0.0034090468315576936, -0.0032467642526153263, 0.0031124884534504483, -0.00300235569413038, 0.0029133941834310925, -0.0028433314714888924, 0.002790457547704326, -0.0027535280138248595, 0.0027316970862612667, -0.00272447377355301, ]).reshape(64,1,1)
    return weights
    #
    
def initI20km(shape, dtype=None):
    d = shape[0]
    weights = np.array([-0.0034473010815673165, 0.0034740349756390548, -0.003519227386457904, 0.003583863620983078, -0.003669388652068963, 0.003777779298722117, -0.003911647498155066, 0.00407438445383684, -0.004270360231237968, 0.004505200524423879, -0.00478617327416306, 0.005122734989366431, -0.005527314106509645, 0.006016453605266108, -0.006612509769715743, 0.007346230029245889, -0.008260746629819191, 0.009417878870315019, -0.01090818081086292, 0.012866747178673343, -0.015496017593701856, 0.019087191680395035, -0.023977423882269686, 0.030124013102092305, -0.03495360064614286, 0.02527420271581298, 0.040959385065278145, -0.21903065041113706, 0.3448555896481379, 0.05488033455330303, -0.1021221872364163, -0.25318504270088377, -0.10212218723641796, 0.05488033455330249, 0.3448555896481378, -0.2190306504111359, 0.040959385065278325, 0.02527420271581271, -0.03495360064614328, 0.030124013102092225, -0.023977423882269984, 0.01908719168039582, -0.01549601759370279, 0.012866747178673962, -0.010908180810864237, 0.009417878870315354, -0.008260746629819038, 0.00734623002924598, -0.006612509769716928, 0.006016453605266633, -0.005527314106509051, 0.0051227349893661405, -0.004786173274162411, 0.004505200524423778, -0.004270360231240095, 0.0040743844538361305, -0.003911647498154759, 0.0037777792987222928, -0.0036693886520661823, 0.0035838636209819708, -0.0035192273864582575, 0.0034740349756407878, -0.003447301081566115, 0.0034384524983530895, ]).reshape(64,1,1)
    return weights
    #
    
def initR_20km(shape, dtype=None):
    return initR20km(shape, dtype)

def initI_20km(shape, dtype=None):
    d = shape[0]
    weights = -np.array([-0.0034473010815673165, 0.0034740349756390548, -0.003519227386457904, 0.003583863620983078, -0.003669388652068963, 0.003777779298722117, -0.003911647498155066, 0.00407438445383684, -0.004270360231237968, 0.004505200524423879, -0.00478617327416306, 0.005122734989366431, -0.005527314106509645, 0.006016453605266108, -0.006612509769715743, 0.007346230029245889, -0.008260746629819191, 0.009417878870315019, -0.01090818081086292, 0.012866747178673343, -0.015496017593701856, 0.019087191680395035, -0.023977423882269686, 0.030124013102092305, -0.03495360064614286, 0.02527420271581298, 0.040959385065278145, -0.21903065041113706, 0.3448555896481379, 0.05488033455330303, -0.1021221872364163, -0.25318504270088377, -0.10212218723641796, 0.05488033455330249, 0.3448555896481378, -0.2190306504111359, 0.040959385065278325, 0.02527420271581271, -0.03495360064614328, 0.030124013102092225, -0.023977423882269984, 0.01908719168039582, -0.01549601759370279, 0.012866747178673962, -0.010908180810864237, 0.009417878870315354, -0.008260746629819038, 0.00734623002924598, -0.006612509769716928, 0.006016453605266633, -0.005527314106509051, 0.0051227349893661405, -0.004786173274162411, 0.004505200524423778, -0.004270360231240095, 0.0040743844538361305, -0.003911647498154759, 0.0037777792987222928, -0.0036693886520661823, 0.0035838636209819708, -0.0035192273864582575, 0.0034740349756407878, -0.003447301081566115, 0.0034384524983530895, ]).reshape(64,1,1)
    return weights
    #
    
def initR60km(shape, dtype=None):
    d = shape[0]
    weights = np.array([0.015142193983508667, -0.015364222052374699, 0.015745302620495692, -0.01630297922352904, 0.01706385356182996, -0.018065685352617853, 0.019359913771276816, -0.021013696836426496, 0.0231086112579017, -0.025728120385910457, 0.028913847674311335, -0.03254435908173784, 0.036041241812696884, -0.03774521647218584, 0.033827142314067, -0.017052418257961097, -0.022678478521224125, 0.09106267583026972, -0.17000725977335016, 0.19672649176941326, -0.09378308698020346, -0.09691492348498731, 0.12951393170518347, 0.10213779512383396, -0.15138019085815968, -0.13234217246253074, -0.01307264268519101, 0.1443432880032672, 0.1589736528245988, 0.17187713829569903, 0.12002707189888545, 0.13533186014846788, 0.12002707189888501, 0.1718771382956978, 0.15897365282459944, 0.1443432880032688, -0.013072642685189859, -0.13234217246253066, -0.15138019085816, 0.10213779512383347, 0.12951393170518413, -0.09691492348498507, -0.09378308698020461, 0.19672649176941448, -0.17000725977334863, 0.09106267583026957, -0.022678478521222303, -0.017052418257960188, 0.033827142314065646, -0.037745216472187426, 0.03604124181269762, -0.032544359081737556, 0.0289138476743124, -0.02572812038591058, 0.023108611257902867, -0.02101369683642599, 0.019359913771276976, -0.018065685352616302, 0.0170638535618317, -0.016302979223531074, 0.015745302620497312, -0.015364222052373738, 0.015142193983507158, -0.015069260186174702, ]).reshape(64,1,1)
    return weights
    #

def initI60km(shape, dtype=None):
    d = shape[0]
    weights = -np.array([-0.0073815717586582065, 0.007523071038046098, -0.007769664407082797, 0.00813931476825458, -0.008661313124650762, 0.009381769290057101, -0.01037311975827138, 0.011750574564744796, -0.013700567211772976, 0.016529354881303176, -0.020742659533914444, 0.027163176632454506, -0.03706028879653308, 0.05214977168896147, -0.07402039203559203, 0.10201463839119752, -0.12834696892048775, 0.13150700396628343, -0.07775622077690648, -0.047216789369968426, 0.17246743998289868, -0.13865161993582925, -0.07793651009259439, 0.14145509677075604, 0.11629379735794365, -0.09762687754565347, -0.16562192401585246, -0.12048514258190779, -0.0036840742410747945, 0.06243501275843809, 0.11659083233348953, 0.11593426631547883, 0.1165908323334906, 0.062435012758437866, -0.003684074241074073, -0.12048514258190693, -0.16562192401585252, -0.09762687754565452, 0.11629379735794232, 0.14145509677075743, -0.07793651009259486, -0.13865161993582864, 0.17246743998289782, -0.04721678936996879, -0.07775622077690654, 0.13150700396628393, -0.12834696892048805, 0.10201463839119909, -0.07402039203559108, 0.05214977168895977, -0.03706028879653443, 0.02716317663245321, -0.02074265953391398, 0.016529354881302104, -0.013700567211772048, 0.011750574564743902, -0.010373119758269947, 0.009381769290057827, -0.008661313124649779, 0.008139314768254654, -0.007769664407083947, 0.0075230710380475895, -0.007381571758656837, 0.007335433048362557,]).reshape(64,1,1)
    return weights
    #   

def initR_60km(shape, dtype=None):
    d = shape[0]
    weights = np.array([0.015142193983508667, -0.015364222052374699, 0.015745302620495692, -0.01630297922352904, 0.01706385356182996, -0.018065685352617853, 0.019359913771276816, -0.021013696836426496, 0.0231086112579017, -0.025728120385910457, 0.028913847674311335, -0.03254435908173784, 0.036041241812696884, -0.03774521647218584, 0.033827142314067, -0.017052418257961097, -0.022678478521224125, 0.09106267583026972, -0.17000725977335016, 0.19672649176941326, -0.09378308698020346, -0.09691492348498731, 0.12951393170518347, 0.10213779512383396, -0.15138019085815968, -0.13234217246253074, -0.01307264268519101, 0.1443432880032672, 0.1589736528245988, 0.17187713829569903, 0.12002707189888545, 0.13533186014846788, 0.12002707189888501, 0.1718771382956978, 0.15897365282459944, 0.1443432880032688, -0.013072642685189859, -0.13234217246253066, -0.15138019085816, 0.10213779512383347, 0.12951393170518413, -0.09691492348498507, -0.09378308698020461, 0.19672649176941448, -0.17000725977334863, 0.09106267583026957, -0.022678478521222303, -0.017052418257960188, 0.033827142314065646, -0.037745216472187426, 0.03604124181269762, -0.032544359081737556, 0.0289138476743124, -0.02572812038591058, 0.023108611257902867, -0.02101369683642599, 0.019359913771276976, -0.018065685352616302, 0.0170638535618317, -0.016302979223531074, 0.015745302620497312, -0.015364222052373738, 0.015142193983507158, -0.015069260186174702, ]).reshape(64,1,1)
    return weights
    #
    
def initI_60km(shape, dtype=None):
    d = shape[0]
    weights = -np.array([-0.0073815717586582065, 0.007523071038046098, -0.007769664407082797, 0.00813931476825458, -0.008661313124650762, 0.009381769290057101, -0.01037311975827138, 0.011750574564744796, -0.013700567211772976, 0.016529354881303176, -0.020742659533914444, 0.027163176632454506, -0.03706028879653308, 0.05214977168896147, -0.07402039203559203, 0.10201463839119752, -0.12834696892048775, 0.13150700396628343, -0.07775622077690648, -0.047216789369968426, 0.17246743998289868, -0.13865161993582925, -0.07793651009259439, 0.14145509677075604, 0.11629379735794365, -0.09762687754565347, -0.16562192401585246, -0.12048514258190779, -0.0036840742410747945, 0.06243501275843809, 0.11659083233348953, 0.11593426631547883, 0.1165908323334906, 0.062435012758437866, -0.003684074241074073, -0.12048514258190693, -0.16562192401585252, -0.09762687754565452, 0.11629379735794232, 0.14145509677075743, -0.07793651009259486, -0.13865161993582864, 0.17246743998289782, -0.04721678936996879, -0.07775622077690654, 0.13150700396628393, -0.12834696892048805, 0.10201463839119909, -0.07402039203559108, 0.05214977168895977, -0.03706028879653443, 0.02716317663245321, -0.02074265953391398, 0.016529354881302104, -0.013700567211772048, 0.011750574564743902, -0.010373119758269947, 0.009381769290057827, -0.008661313124649779, 0.008139314768254654, -0.007769664407083947, 0.0075230710380475895, -0.007381571758656837, 0.007335433048362557,]).reshape(64,1,1)
    return weights
    #

def filter_taps(shape,distance=20e3,mode='r',dtype=None):
    VecLen = shape[0]
    W = dft(VecLen, scale='sqrtn')
    Winv = 1/(VecLen*W)
    dt = 1/(32e9*2)
    f = t2f( np.arange(0,VecLen*dt,dt))
    f = fftshift(f)
    beta2 = -2.166761921076912e-26
    H = 1j*beta2/2*(2*np.pi*f)**2
    D = np.diagflat(np.exp(distance*H))
    A = np.dot(Winv,np.dot(D,W))
    indx = int((VecLen-1)/2)
    if mode == 'r':
        return np.real(A[indx]).reshape(shape)
    return np.imag(A[indx]).reshape(shape)

def avg(a,factor=4):
    N = int(len(a)/factor)
    Out = np.zeros(N)
    for i in range(N):
        Out[i] = np.mean(a[i*factor:i*factor+factor])
    return Out

def kernel_init(width,step_size,dt,beta2,shape,mode='full'):
    W = dft(width, scale='sqrtn')
    Winv = 1/(width*W)
    w = 2*np.pi/(width*dt) * np.fft.fftshift(np.arange(-width/2,width/2))
    D = np.diagflat(np.exp(1j/2*beta2*(w**2)*step_size/2))
    halfLin_op = np.dot(Winv,np.dot(D,W))
    D = np.diagflat(np.exp(1j/2*beta2*(w**2)*step_size))
    fullLin_op = np.dot(Winv,np.dot(D,W))
    if mode=='full':
        return fullLin_op[int(width/2)].reshape(shape)
    else:
        return halfLin_op[int(width/2)].reshape(shape)
    
    
def cpe_rotation(q0_e,s,sr):
    q0_DBP = 0 + q0_e # recived signal.
    for l in range(2):
        for _ in range(8):
            a = np.concatenate([q0_DBP[l,0,::sr],q0_DBP[l,1,::sr],],axis=0)
            b = np.concatenate([s[l,0],s[l,1],],axis=0)
            Angle = np.mod(np.angle(a)-np.angle(b),2*np.pi)
            Angle2 = np.array([i for i in Angle if i<np.pi]+ [i-2*np.pi for i in Angle if i>np.pi])
            avgAngle = np.sum(Angle2 * abs(a))/np.sum(abs(a))
            q0_DBP[l,0],q0_DBP[l,1] = q0_DBP[l,0] * np.exp(-1j * avgAngle), q0_DBP[l,1] * np.exp(-1j * avgAngle)
    return q0_DBP

def BER_fun(exc,est):
    const = np.array([-3,-1,1,3])/np.sqrt(10)
    thr = np.diff(const)[0]/2
    bins = const[:-1] + thr
    s_dec = np.digitize(np.real(est.flatten()),bins)*2*thr + 1j * np.digitize(np.imag(est.flatten()),bins)*2*thr - 3*thr - 3j*thr
    s_exc = exc.flatten()
    eps = 1e-8
    symbol_errors = np.sum(abs(s_dec - s_exc)>eps)
    b_dec = s2b(np.round(s_dec.flatten()*np.sqrt(10)))
    b_exc = s2b(np.round(s_exc.flatten()*np.sqrt(10)))
    bits_errors = np.sum(b_dec != b_exc)
    BER = bits_errors/len(b_exc)
    print('\n\n\nBER achieved by X=')
    print(BER)
    Qfac_dB_NN = Qfac(BER)
    print('corresponding Q-fac=')
    print(Qfac_dB_NN)
    return Qfac_dB_NN

def BER_funQAM64(exc,est):
    const = np.array([-7,-5,-3,-1,1,3,5,7])/np.sqrt(42)
    thr = np.diff(const)[0]/2
    bins = const[:-1] + thr
    s_dec = np.digitize(np.real(est.flatten()),bins)*2*thr + 1j * np.digitize(np.imag(est.flatten()),bins)*2*thr - 7*thr - 7j*thr
    s_exc = exc.flatten()
    eps = 1e-8
    symbol_errors = np.sum(abs(s_dec - s_exc)>eps)
    b_dec = s2bQAM64(np.round(s_dec.flatten()*np.sqrt(42)))
    b_exc = s2bQAM64(np.round(s_exc.flatten()*np.sqrt(42)))
    bits_errors = np.sum(b_dec != b_exc)
    BER = bits_errors/len(b_exc)
    print('\n\n\nBER achieved by X=')
    print(BER)
    Qfac_dB_NN = Qfac(BER)
    print('corresponding Q-fac=')
    print(Qfac_dB_NN)
    return Qfac_dB_NN

def examples_gen(A,length,step=1):
    n = np.shape(A)[1]
    indx = np.arange(int(length/2),n-int(length/2)+1,step)
    N_examples = len(indx)
    indxmat = np.repeat(indx.reshape(N_examples,1),length,axis = 1) + np.repeat(np.arange(int(-length/2),int(length/2),dtype='int').reshape(1,length),N_examples,axis=0)
    Split = np.expand_dims(A[:,indxmat],3)
    Out = [np.real(Split[0]),
           np.imag(Split[0])]
    return Out,indx

def examples_genXY(A,length,step=1):
    n = np.shape(A)[1]
    indx = np.arange(int(length/2),n-int(length/2)+1,step)
    N_examples = len(indx)
    indxmat = np.repeat(indx.reshape(N_examples,1),length,axis = 1) + np.repeat(np.arange(int(-length/2),int(length/2),dtype='int').reshape(1,length),N_examples,axis=0)
    Split = np.expand_dims(A[:,indxmat],3)
    Out = [np.real(Split[0]),
           np.imag(Split[0]),
           np.real(Split[1]),
           np.imag(Split[1]),
           ]
    return Out,indx

class Phase_function(tf.keras.layers.Layer):
    def __init__(self,initial):
        super(Phase_function, self).__init__()
        self.scale = initial
    def call(self, ReX,ImX,ReY,ImY):
        termX = ImX**2+ReX**2
        termY = ImY**2+ReY**2       
        Ang = self.scale*(termX + termY)    
        OutXr = tf.cos(Ang) * ReX - ImX * tf.sin(Ang)
        OutXi = tf.cos(Ang) * ImX + ReX * tf.sin(Ang)
        OutYr = tf.cos(Ang) * ReY - ImY * tf.sin(Ang)
        OutYi = tf.cos(Ang) * ImY + ReY * tf.sin(Ang)
        return OutXr,OutXi,OutYr,OutYi