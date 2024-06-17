import matplotlib.pyplot as plt
import numpy as np
from dsys import *
from commpy.filters import rrcosfilter
import time

# Simulation parameters
B = 32e9  # Bandwidth of each WDM channel
T = 1 / B  # Symbol duration
N_s = 2**16  # Number of transmitted symbols per polarization
N = 2**19  # Number of samples in the window
M = 16  # Constellation size (QAM16)
pol = 2  # Number of polarizations
N_block = 2  # Number of transmission blocks

# Time vector setup
t = np.linspace(-(N_s) / 2 * T, (N_s) / 2 * T, N, endpoint=False)
f = t2f(t)  # Frequency vector corresponding to time vector
dt = t[1] - t[0]  # Time step
w = 2 * np.pi / (N * dt) * np.fft.fftshift(np.arange(-N / 2, N / 2))

# Bit and symbol setup for modulation
N_bits = int(N_s * np.log2(M))  # Total number of bits per polarization
gen_bits = np.zeros((N_block, pol, N_bits))  # Array to store generated bits for all blocks and polarizations
s = np.zeros((N_block, pol, N_s))+0j # generated symbols

# Preparing signals for transmission
q0 = np.zeros((N_block, pol, N)) + 0j  # Array to hold modulated signals

# Generate random bits and modulate for each block and polarization
for l in range(N_block):
    for k in range(pol):
        gen_bits[l, k] = source(N_bits)  # Generate random bits

# Additional bit arrays for multiple WDM channels
gen_bits0, gen_bits1, gen_bits2, gen_bits3 = [np.zeros((N_block, pol, N_bits)) for _ in range(4)]
for l in range(N_block):
    for k in range(pol):
        gen_bits0[l, k] = source(N_bits)
        gen_bits1[l, k] = source(N_bits)
        gen_bits2[l, k] = source(N_bits)
        gen_bits3[l, k] = source(N_bits)

# Physical constants and parameters for fiber transmission
c = 3e8  # Speed of light in vacuum
h = 6.626075e-34  # Planck's constant
lambda0 = 1.55e-6  # Central wavelength of the light
f0 = c / lambda0  # Center frequency
#******************************************************************************************************
#fiber parametars

l_span = np.array([72, 15, 72, 15, 72, 16, 72, 15, 72, 15, 72, 16, 72, 15, 72, 15, 72, 16, 72, 15, 72, 15, 72, 16, 72, 15, 72, 15,
                  72, 15, 72, 16, 72, 15, 72, 15, 72, 16, 72, 15, 72, 15, 72, 16, 72, 15, 72, 15, 72, 16, 72, 15, 72, 15, 72, 15]).astype(int)
D = np.array([17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80,
             17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, ])
D = D*10**(-6)  # disperion in s/m^2
beta2 = (-lambda0**2)/(2*np.pi*c)*D  # dispersion in s^2/m

gamma = np.array([1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8,
                 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, 1.4, 2.8, ])*10**(-3)
alpha_db = np.array([0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0,
                    0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, ])
pmd = 0.05
NF_db = 5.5
step_s = 1e3
z = int(np.sum(l_span))
THETA = 2*np.pi*np.random.uniform(0, 1, z) * 0
PHI = 2*np.pi*np.random.uniform(0, 1, z) * 0
TAU = np.sqrt((pmd*1e-12)**2*(step_s*10**(-3)))*np.random.normal(size=z) * 0
#**********

s0 = np.zeros((N_block, pol, N_s))+0j  # generated symbols
s1 = np.zeros((N_block, pol, N_s))+0j  # generated symbols
s2 = np.zeros((N_block, pol, N_s))+0j  # generated symbols
s3 = np.zeros((N_block, pol, N_s))+0j  # generated symbols

fc0 = -75e9
fc1 = -37.5e9
fc2 = 37.5e9
fc3 = 75e9

Time = time.time()
for p_dbm in [-2,-1,0,1,2,3,4,5,6,7,8]:
    normalized = 0
    #p_dbm= -3
    p_int = 10**(p_dbm/10)*10**(-3)
    p = p_int / 2 / 5  # BE CAREFUL HERE
    for l in range(N_block):
        for k in range(pol):
            s[l, k] = QAM16(gen_bits[l, k], p)
            s0[l, k] = QAM16(gen_bits0[l, k], p)
            s1[l, k] = QAM16(gen_bits1[l, k], p)
            s2[l, k] = QAM16(gen_bits2[l, k], p)
            s3[l, k] = QAM16(gen_bits3[l, k], p)
            q0[l, k] = mod(t, s[l, k], B) + mod(t, s0[l, k], B) * np.exp(1j*2*np.pi*fc0*t) + mod(t, s1[l, k], B) * np.exp(
                1j*2*np.pi*fc1*t) + mod(t, s2[l, k], B) * np.exp(1j*2*np.pi*fc2*t) + mod(t, s3[l, k], B) * np.exp(1j*2*np.pi*fc3*t)
        print(l)

    finess= 256
    Q0= avg( np.abs(np.fft.fft(q0[0,0]) / len(f)), finess )
    plt.figure()
    plt.plot(f[::finess],np.abs(np.fft.fftshift(Q0)))
    plt.title('Frequency spectrum of Transmitted signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Spectral Energy')
    plt.grid()

    qz = np.zeros((N_block, pol, N))+0j     # recived signal.
    for l in range(N_block):
        qz[l, 0], qz[l, 1] = ssfm1(
            q0[l], l_span, gamma, alpha_db, beta2, dt, TAU, THETA, PHI, B, NF_db, h, f0, N_s, t)
        print(l)
    
    plt.figure()
    plt.plot(f, np.fft.fftshift(np.fft.fft(qz[0, 0])))
    plt.title('WDM Spectrum at Rx in Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    Filter_f = np.fft.ifft(rrcosfilter(
        N, alpha=0.06, Ts=32/34.75, Fs=8)[1])*N_s * 34.75/32
    Filter_f = np.fft.fftshift(np.logical_and(f < 17e9, f > -17e9))

    # plt.figure()
    # plt.plot(f,np.abs(fftshift(Filter_f)))
    # plt.title('RRC Filter (roll-off factor= 0.06)')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('Spectral Energy')
    # plt.grid()

    qz_f = np.zeros((N_block, pol, N))+0j
    for l in range(N_block):
        qz_f[l, 0] = np.fft.ifft(np.fft.fft(qz[l, 0]) * np.abs(Filter_f))
        qz_f[l, 1] = np.fft.ifft(np.fft.fft(qz[l, 1]) * np.abs(Filter_f))

    plt.figure()
    plt.plot(f, np.fft.fftshift(np.fft.fft(qz_f[0, 0])))
    plt.title('Eq Signal Spectrum in Frequency Domain')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


    q0_e = np.zeros((N_block, pol, N))+0j  # recived signal.
    for l in range(N_block):
        q0_e[l, 0], q0_e[l, 1] = DBP1(
            qz_f[l], l_span, gamma, alpha_db, beta2, dt, TAU, THETA, PHI, B, NF_db, h, f0, N_s, t)
        print(l)

    Str1= 'Signal_q0_WDM_2000km_37_5_GHz_RD00_' + str(int(p_dbm)) + 'dB_DM.npy'
    Str2= 'Signal_qz_WDM_2000km_37_5_GHz_RD00_' + str(int(p_dbm)) + 'dB_DM.npy'
    Str3= 'Signal_q0e_WDM_2000km_37_5_GHz_RD00_' + str(int(p_dbm)) + 'dB_DM.npy'
    Str4= 'symbols_WDM_2000km_37_5_GHz_RD00_'+str(int(p_dbm))+'dB.npy'

    np.save(Str1,q0)
    np.save(Str2,qz)
    np.save(Str3,q0_e)
    np.save(Str4,s)

    plt.figure()
    plt.plot(np.real(q0_e[0,0,::8]),np.imag(q0_e[0,0,::8]),'.',markersize= 0.3)
    # plt.figure()
    # plt.plot(np.real(qz[0,0,::8]),np.imag(qz[0,0,::8]),'.',markersize= 0.3)

    print('Data saved for launch power= ', p_dbm, ' dB')
    print('Time elapsed: ', time.time()-Time)
    Time = time.time()
