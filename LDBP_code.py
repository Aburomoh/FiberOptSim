import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import tensorflow.keras as K
from dsys import * # dsys is a custom module for DSP related functions
from scipy.special import erfcinv
from commpy.filters import rrcosfilter

# Initialize lists to store quality factors for different equalization strategies
DBP = []        # Full precision DBP
Linear = []     # Linear Equalization
#NN = []
NNXY = []       # Dual-pol Model-based Neural Network
#DBPwDSP = []
DBP1s = []      # DBP with 1 step/span
#DBP2s = []
#DBP3s = []

# Iterate over a range of launch powers from -2 dBm to 8 dBm
for p_dbm in range(-2, 9):
    # Construct file paths for different signals and symbols saved as NumPy arrays    
    # You need to run WDM_signal_generation.py to generate these files.
    Str1 = 'Signal_q0_WDM_2000km_37_5_GHz_RD00_' + \
        str(int(p_dbm)) + 'dB_DM.npy'
    Str2 = 'Signal_qz_WDM_2000km_37_5_GHz_RD00_' + \
        str(int(p_dbm)) + 'dB_DM.npy'
    Str3 = 'Signal_q0e_WDM_2000km_37_5_GHz_RD00_' + \
        str(int(p_dbm)) + 'dB_DM.npy'
    Str4 = 'symbols_WDM_2000km_37_5_GHz_RD00_'+str(int(p_dbm))+'dB.npy'

    # Define constants for the simulation
    
    B = 32e9  # symbol bandwidth
    T = 1/B        # symbol duration
    N_s = 2**16  # number of transmitted symbols for a singel pol.
    N = (2**19)  # number of samples in the window.
    M = 16  # constelation size
    pol = 2  # polrization
    N_block = 2  # number of transmission blocks
    t = np.linspace(-(N_s)/2*T, (N_s)/2*T, N, endpoint=False)
    dt = t[1]-t[0]
    f = t2f(t)
    w = 2*np.pi/(N*dt) * np.fft.fftshift(np.arange(-N/2, N/2))
    #*********************************************************************************************************
    # number of transmitted bits in each polirization.
    N_bits = int(N_s*np.log2(M))
    gen_bits = np.zeros((N_block, pol, N_bits))  # generated bits

    q0 = np.zeros((N_block, pol, N)) + 0j
    s = np.load(Str4)
    normalized = 0
    #p_dbm = -3
    p_int = 10**(p_dbm/10)*10**(-3)
    p = p_int/2 / 5  # CAREFUL HERE

    for l in range(N_block):
        for k in range(pol):
            q0[l, k] = mod(t, s[l, k], B)
    #############################################################################################################
    qz = np.load(Str2)
    q0_e = np.load(Str3)
    
    # Create a filter using a raised cosine response
    Filter_f = np.fft.ifft(rrcosfilter(
        N, alpha=0.06, Ts=32/34.75, Fs=8)[1])*N_s * 34.75/32
    Filter_f = np.fft.fftshift(np.logical_and(f < 17e9, f > -17e9))
    
    qz_f = np.zeros((N_block, pol, N))+0j # Array to hold filtered received signals
    for l in range(N_block):
        # Apply the filter to the received signal in frequency domain and convert back to time domain
        qz_f[l, 0] = np.fft.ifft(np.fft.fft(qz[l, 0]) * np.abs(Filter_f))
        qz_f[l, 1] = np.fft.ifft(np.fft.fft(qz[l, 1]) * np.abs(Filter_f))
    
    # Load fiber parameters for DM system
    gamma = 1.4e-3
    q_linear = qz_f + 0j
    l_span = np.array([72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13,
                      72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13, 72, 13]).astype(int)
    D = np.array([17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80,
                 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, 17, -80, ])
    
    # Different setting for residual dispersion (not discussed in this code)
    if 'RD00' in Str1:
        res_CD = 32
        r = 1.00    # Ratio of residual dispersion to total generated dispersion inside the link.
    elif 'RD15' in Str1:
        res_CD = 5152
        r = 0.85    # Ratio of residual dispersion to total generated dispersion inside the link.
    elif 'RD30' in Str1:
        res_CD = 10352
        r = 0.70    # Ratio of residual dispersion to total generated dispersion inside the link.
    elif 'RD100' in Str1:
        res_CD = 34272
        r = 0.0     # Ratio of residual dispersion to total generated dispersion inside the link.
        
    beta2 = (-1.55e-6**2)/(2*np.pi*3e8) * 17e-6
    z_CD = res_CD/17 * 1e3
    dis = np.exp(1j/2*beta2*(w**2)*z_CD)
    for l in range(N_block):
        signal_recfx = np.fft.fft(qz_f[l,0])
        signal_recfy = np.fft.fft(qz_f[l,1])
        signal_recfx *= dis
        signal_recfy *= dis
        q_linear[l,0] = np.fft.ifft(signal_recfx)
        q_linear[l,1] = np.fft.ifft(signal_recfy)
    
    q0_cpe = 0 + q_linear
    Phase_shift = 0
    for l in range(N_block):
        for _ in range(10):
            a = np.concatenate(
                [q0_cpe[l, 0, ::8], q0_cpe[l, 1, ::8], ], axis=0)
            b = np.concatenate([q0[l, 0, ::8], q0[l, 1, ::8], ], axis=0)
            Angle = np.mod(np.angle(a)-np.angle(b), 2*np.pi)
            Angle2 = np.array([i for i in Angle if i < np.pi] +
                              [i-2*np.pi for i in Angle if i > np.pi])
            avgAngle = np.sum(Angle2 * abs(a))/np.sum(abs(a))
            q0_cpe[l, 0], q0_cpe[l, 1] = q0_cpe[l, 0] * \
                np.exp(-1j * avgAngle), q0_cpe[l, 1] * np.exp(-1j * avgAngle)
        print(avgAngle)
        q0_cpe[l, 0], q0_cpe[l, 1] = q0_cpe[l, 0] * \
            np.exp(1j * Phase_shift), q0_cpe[l, 1] * np.exp(1j * Phase_shift)
    q0_DBP = 0 + q0_e  # recived signal.
    for l in range(N_block):
        for _ in range(10):
            a = np.concatenate(
                [q0_DBP[l, 0, ::8], q0_DBP[l, 1, ::8], ], axis=0)
            b = np.concatenate([q0[l, 0, ::8], q0[l, 1, ::8], ], axis=0)
            Angle = np.mod(np.angle(a)-np.angle(b), 2*np.pi)
            Angle2 = np.array([i for i in Angle if i < np.pi] +
                              [i-2*np.pi for i in Angle if i > np.pi])
            avgAngle = np.sum(Angle2 * abs(a))/np.sum(abs(a))
            q0_DBP[l, 0], q0_DBP[l, 1] = q0_DBP[l, 0] * \
                np.exp(-1j * avgAngle), q0_DBP[l, 1] * np.exp(-1j * avgAngle)
        print(avgAngle)

    # plt.figure()
    # plt.plot(np.real(q0_e[0,0,::8])/np.sqrt(p),np.imag(q0_e[0,0,::8])/np.sqrt(p),'.',markersize=0.7)
    # plt.figure()
    # plt.plot(np.real(q0_DBP[0,0,::8])/np.sqrt(p),np.imag(q0_DBP[0,0,::8])/np.sqrt(p),'.',markersize=0.7)
    # plt.figure()
    # plt.plot(np.real(q_linear[0,0,::8])/np.sqrt(p),np.imag(q_linear[0,0,::8])/np.sqrt(p),'.',markersize=0.7)
    # plt.figure()
    # plt.plot(np.real(q0_cpe[0,0,::8])/np.sqrt(p),np.imag(q0_cpe[0,0,::8])/np.sqrt(p),'.',markersize=0.7)

    #***************************************************************************

    # plt.figure()
    # plt.plot(np.real(kernel_init(512,72e3,4*dt,beta2,512,mode='full')))
    # plt.plot(np.imag(kernel_init(512,72e3,4*dt,beta2,512,mode='full')))

    '''
    This code is for simulating a model-based neural network with a computation graph similar to DBP. The idea is that
    each layer performs a similar computational operation as a DBP step, however completely in time-domain, alternating
    between linear convolutional filter and nonlinear activation function.
    '''
    K.backend.set_floatx('float32')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=0,
                                                mode='auto', baseline=None, restore_best_weights=False)

    Input_width = Output_width = 2**9
    N_layers = 28   # Number of Neural Network Layers.

    if N_layers == 28:
        lengths = [72e3*(0.5-r), 72e3*(1-r), 72e3*(0.5)]  # 2 layers
    elif N_layers == 14:
        lengths = [72e3*(1-3*r/2), 72e3*(2-2*r), 72e3*(1-r/2)]  # 4 layers
    elif N_layers == 7:
        lengths = [72e3*(2-5*r/2), 72e3*(4-4*r), 72e3*(2-3*r/2)]  # 7 layers
    # elif N_layers == 4:
    #     lengths = [-292.2e3, 21.6e3, 11.4e3]  # 14 layers
    # elif N_layers == 2:
    #     lengths = [-334.2e3, 10.8e3, 42.6e3]  # 28 layers

    filter_width = [0, 0, 0]
    filter_width[0] = 4*int(1+np.abs(2*np.pi*beta2 * lengths[0]/(T/2)**2))
    filter_width[1] = 8*int(1+np.abs(2*np.pi*beta2 * lengths[1]/(T/2)**2))
    filter_width[2] = 4*int(1+np.abs(2*np.pi*beta2 * lengths[2]/(T/2)**2))

    InputXr = tf.keras.layers.Input(shape=(Input_width, 1))
    InputXi = tf.keras.layers.Input(shape=(Input_width, 1))
    InputYr = tf.keras.layers.Input(shape=(Input_width, 1))
    InputYi = tf.keras.layers.Input(shape=(Input_width, 1))

    Act_fun1 = Phase_function(-gamma*72e3*0.2777*p * 28/N_layers)

    Normlayer = tf.keras.layers.LayerNormalization(axis=-2)

    OutXr, OutXi, OutYr, OutYi = InputXr, InputXi, InputYr, InputYi
    LayerR = tf.keras.layers.Conv1D(1, kernel_size=(filter_width[0]),
                                    use_bias=False,
                                    activation=None,
                                    input_shape=(None, Input_width, 1),
                                    trainable=True)
    LayerI = tf.keras.layers.Conv1D(1, kernel_size=(filter_width[0]),
                                    use_bias=False,
                                    activation=None,
                                    input_shape=(None, Input_width, 1),
                                    trainable=True)

    OutXrConc = tf.keras.layers.Concatenate(
        axis=-2)([OutXr[:, -int(filter_width[0]/2):, :], OutXr, OutXr[:, :int(filter_width[0]/2-1), :]])
    OutXiConc = tf.keras.layers.Concatenate(
        axis=-2)([OutXi[:, -int(filter_width[0]/2):, :], OutXi, OutXi[:, :int(filter_width[0]/2-1), :]])
    OutYrConc = tf.keras.layers.Concatenate(
        axis=-2)([OutYr[:, -int(filter_width[0]/2):, :], OutYr, OutYr[:, :int(filter_width[0]/2-1), :]])
    OutYiConc = tf.keras.layers.Concatenate(
        axis=-2)([OutYi[:, -int(filter_width[0]/2):, :], OutYi, OutYi[:, :int(filter_width[0]/2-1), :]])
    OutXr, OutXi, OutYr, OutYi = LayerR(OutXrConc)-LayerI(OutXiConc), LayerR(OutXiConc)+LayerI(
        OutXrConc), LayerR(OutYrConc)-LayerI(OutYiConc), LayerR(OutYiConc)+LayerI(OutYrConc)
    OutXr, OutXi, OutYr, OutYi = Act_fun1(OutXr, OutXi, OutYr, OutYi)
    # OutXr,OutXi,OutYr,OutYi = Normlayer(OutXr)/np.sqrt(2),Normlayer(OutXi)/np.sqrt(2),Normlayer(OutYr)/np.sqrt(2),Normlayer(OutYi)/np.sqrt(2)
    
    for _ in range(N_layers-1):
        LayerR = tf.keras.layers.Conv1D(1, kernel_size=(filter_width[1]),
                                        use_bias=False,
                                        activation=None,
                                        input_shape=(None, Input_width, 1),
                                        trainable=False)
        LayerI = tf.keras.layers.Conv1D(1, kernel_size=(filter_width[1]),
                                        use_bias=False,
                                        activation=None,
                                        input_shape=(None, Input_width, 1),
                                        trainable=False)
        OutXrConc = tf.keras.layers.Concatenate(
            axis=-2)([OutXr[:, -int(filter_width[1]/2):, :], OutXr, OutXr[:, :int(filter_width[1]/2-1), :]])
        OutXiConc = tf.keras.layers.Concatenate(
            axis=-2)([OutXi[:, -int(filter_width[1]/2):, :], OutXi, OutXi[:, :int(filter_width[1]/2-1), :]])
        OutYrConc = tf.keras.layers.Concatenate(
            axis=-2)([OutYr[:, -int(filter_width[1]/2):, :], OutYr, OutYr[:, :int(filter_width[1]/2-1), :]])
        OutYiConc = tf.keras.layers.Concatenate(
            axis=-2)([OutYi[:, -int(filter_width[1]/2):, :], OutYi, OutYi[:, :int(filter_width[1]/2-1), :]])
        OutXr, OutXi, OutYr, OutYi = LayerR(OutXrConc)-LayerI(OutXiConc), LayerR(OutXiConc)+LayerI(
            OutXrConc), LayerR(OutYrConc)-LayerI(OutYiConc), LayerR(OutYiConc)+LayerI(OutYrConc)
        OutXr, OutXi, OutYr, OutYi = Act_fun1(OutXr, OutXi, OutYr, OutYi)
        # OutXr,OutXi,OutYr,OutYi = Normlayer(OutXr)/np.sqrt(2),Normlayer(OutXi)/np.sqrt(2),Normlayer(OutYr)/np.sqrt(2),Normlayer(OutYi)/np.sqrt(2)

    LayerR = tf.keras.layers.Conv1D(1, kernel_size=(filter_width[2]),
                                    use_bias=False,
                                    activation=None,
                                    input_shape=(None, Input_width, 1),
                                    trainable=True)
    LayerI = tf.keras.layers.Conv1D(1, kernel_size=(filter_width[2]),
                                    use_bias=False,
                                    activation=None,
                                    input_shape=(None, Input_width, 1),
                                    trainable=True)

    OutXrConc = tf.keras.layers.Concatenate(
        axis=-2)([OutXr[:, -int(filter_width[2]/2):, :], OutXr, OutXr[:, :int(filter_width[2]/2-1), :]])
    OutXiConc = tf.keras.layers.Concatenate(
        axis=-2)([OutXi[:, -int(filter_width[2]/2):, :], OutXi, OutXi[:, :int(filter_width[2]/2-1), :]])
    OutYrConc = tf.keras.layers.Concatenate(
        axis=-2)([OutYr[:, -int(filter_width[2]/2):, :], OutYr, OutYr[:, :int(filter_width[2]/2-1), :]])
    OutYiConc = tf.keras.layers.Concatenate(
        axis=-2)([OutYi[:, -int(filter_width[2]/2):, :], OutYi, OutYi[:, :int(filter_width[2]/2-1), :]])
    OutXr, OutXi, OutYr, OutYi = LayerR(OutXrConc)-LayerI(OutXiConc), LayerR(OutXiConc)+LayerI(
        OutXrConc), LayerR(OutYrConc)-LayerI(OutYiConc), LayerR(OutYiConc)+LayerI(OutYrConc)
    # OutXr,OutXi,OutYr,OutYi = Normlayer(OutXr)/np.sqrt(2),Normlayer(OutXi)/np.sqrt(2),Normlayer(OutYr)/np.sqrt(2),Normlayer(OutYi)/np.sqrt(2)

    NN_model = tf.keras.models.Model(inputs=[InputXr, InputXi, InputYr, InputYi, ], outputs=[
                                     OutXr[:, 64:-64], OutXi[:, 64:-64], OutYr[:, 64:-64], OutYi[:, 64:-64], ])
    NN_model.compile(optimizer=tf.keras.optimizers.Adam(),  # optimizer=ada_grad
                     loss='mean_squared_error')
    
    indx = []
    for i in range(len(NN_model.layers)):
        if 'conv' in NN_model.layers[i].name:
            indx.append(i)
            
    NN_model.layers[indx[0]].set_weights([np.real(kernel_init(
        filter_width[0], lengths[0], 4*dt, beta2, (filter_width[0], 1, 1), mode='full'))])
    NN_model.layers[indx[1]].set_weights([np.imag(kernel_init(
        filter_width[0], lengths[0], 4*dt, beta2, (filter_width[0], 1, 1), mode='full'))])

    NN_model.layers[indx[-2]].set_weights([np.real(kernel_init(
        filter_width[2], lengths[2], 4*dt, beta2, (filter_width[2], 1, 1), mode='full'))])
    NN_model.layers[indx[-1]].set_weights([np.imag(kernel_init(
        filter_width[2], lengths[2], 4*dt, beta2, (filter_width[2], 1, 1), mode='full'))])

    modeR = False
    for i in indx[2:-2]:
        modeR = not modeR
        if modeR:
            NN_model.layers[i].set_weights([np.real(kernel_init(
                filter_width[1], lengths[1], 4*dt, beta2, (filter_width[1], 1, 1), mode='full'))])
        else:
            NN_model.layers[i].set_weights([np.imag(kernel_init(
                filter_width[1], lengths[1], 4*dt, beta2, (filter_width[1], 1, 1), mode='full'))])

    Input_testXY = qz_f[1, :, ::4]/np.sqrt(p)
    Output_test = np.zeros((2, N_s*2)) + 0j
    Output_test = q0[1, :, ::4]/np.sqrt(p)
    NN_input_testXY, indx_input = examples_genXY(
        Input_testXY, length=Input_width, step=Output_width)
    NN_output_test, indx_output = examples_genXY(
        Output_test, length=Output_width, step=Output_width)
    N_test_examples = np.shape(NN_input_testXY)[1]
    NN_output_test_reshaped = np.concatenate(
        [NN_output_test[0], NN_output_test[1]], axis=2)
    Pred = NN_model.predict(
        [NN_input_testXY[0], NN_input_testXY[1], NN_input_testXY[2], NN_input_testXY[3], ])
    est = Pred[0][:, ::2, 0], Pred[1][:, ::2, 0]
    exc = NN_output_test_reshaped[:, 64:-64:2,
                                  0], NN_output_test_reshaped[:, 64:-64:2, 1]

    q0_input = 0 + qz_f
    Phase_shift = 0
    a = (est[0]+1j*est[1]).flatten()
    b = (exc[0]+1j*exc[1]).flatten()
    for l in range(N_block):
        for _ in range(10):
            Angle = np.mod(np.angle(a)-np.angle(b), 2*np.pi)
            Angle2 = np.array([i for i in Angle if i < np.pi] +
                              [i-2*np.pi for i in Angle if i >= np.pi])
            avgAngle = np.sum(Angle2 * abs(a))/np.sum(abs(a))
            a = a * np.exp(-1j * avgAngle)
            Phase_shift += avgAngle
            print(Phase_shift)
        q0_input[l, 0], q0_input[l, 1] = qz_f[l, 0] * \
            np.exp(-1j * Phase_shift), qz_f[l, 1] * np.exp(-1j * Phase_shift)
            
    Input = q0_input[0, :, ::4]/np.sqrt(p)
    Output = np.zeros((2, N_s*2)) + 0j
    Output = q0[0, :, ::4]/np.sqrt(p)
    
    # plt.figure()
    # plt.plot(np.real(q0_input[0,0,::8]),np.imag(q0_input[0,0,::8]),'.',markersize=0.7)
    # plt.figure()
    # plt.plot(np.real(q0_cpe[0,0,::8]),np.imag(q0_cpe[0,0,::8]),'.',markersize=0.7)
    
    NN_inputXY, indx_input = examples_genXY(Input, length=Input_width, step=63)
    NN_output, indx_output = examples_genXY(
        Output, length=Output_width, step=63)

    N_examples = np.shape(NN_output[0])[0]

    NN_input_reshapedXY = np.concatenate(
        [NN_inputXY[0], NN_inputXY[1], NN_inputXY[2], NN_inputXY[3], ], axis=2)
    NN_output_reshaped = np.concatenate(
        [NN_output[0], NN_output[1], NN_output[2], NN_output[3], ], axis=2)

    Training_history = NN_model.fit([NN_input_reshapedXY[:, :, 0:1], NN_input_reshapedXY[:, :, 1:2], NN_input_reshapedXY[:, :, 2:3], NN_input_reshapedXY[:, :, 3:4], ], [NN_output_reshaped[:, 64:-64, 0:1], NN_output_reshaped[:, 64:-64, 1:2], NN_output_reshaped[:, 64:-64, 2:3], NN_output_reshaped[:, 64:-64, 3:4], ], validation_split=0.20, shuffle=True, callbacks=[callback],
                                    epochs=30)

    # for i in range(0, len(indx), 2):
    #     plt.figure()
    #     plt.plot(NN_model.layers[indx[i]].get_weights()[0].flatten())
    #     plt.plot(NN_model.layers[indx[i+1]].get_weights()[0].flatten())

    try:
        plt.figure(figsize=(12, 8))
        plt.plot(Training_history.history['loss'])
        plt.plot(Training_history.history['val_loss'])
        plt.title('model accuracy, z = '+str(int(z/1000))+' km')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.gca().set_ylim(bottom=0)
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
    except:
        pass

    Input_testXY = q0_input[1, :, ::4]/np.sqrt(p)

    Output_test = np.zeros((2, N_s*2)) + 0j
    Output_test = q0[1, :, ::4]/np.sqrt(p)

    NN_input_testXY, indx_input = examples_genXY(
        Input_testXY, length=Input_width, step=int(Output_width*3/4))
    NN_output_test, indx_output = examples_genXY(
        Output_test, length=Output_width, step=int(Output_width*3/4))

    N_test_examples = np.shape(NN_input_testXY)[1]

    NN_output_test_reshaped = np.concatenate(
        [NN_output_test[0], NN_output_test[1]], axis=2)

    Pred = NN_model.predict(
        [NN_input_testXY[0], NN_input_testXY[1], NN_input_testXY[2], NN_input_testXY[3], ])

    est = Pred[0][:, ::2, 0], Pred[1][:, ::2, 0]
    exc = NN_output_test_reshaped[:, 64:-64:2,
                                  0], NN_output_test_reshaped[:, 64:-64:2, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(est[0].flatten(), est[1].flatten(), '.', markersize=0.7)
    plt.plot(exc[0].flatten(), exc[1].flatten(), 'o', markersize=2)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlim([-1.4, 1.4])
    plt.ylim([-1.4, 1.4])
    plt.grid()

    plt.figure(figsize=(6, 6))
    plt.plot(np.real(q0_cpe[0, 0, ::8])/np.sqrt(p),
             np.imag(q0_cpe[0, 0, ::8])/np.sqrt(p), '.', markersize=0.7)
    plt.plot(exc[0].flatten(), exc[1].flatten(), 'o', markersize=2)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlim([-1.4, 1.4])
    plt.ylim([-1.4, 1.4])
    plt.grid()

    plt.figure(figsize=(6, 6))
    plt.plot(np.real(q0_DBP[0, 0, ::8])/np.sqrt(p),
             np.imag(q0_DBP[0, 0, ::8])/np.sqrt(p), '.', markersize=0.7)
    plt.plot(exc[0].flatten(), exc[1].flatten(), 'o', markersize=2)
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlim([-1.4, 1.4])
    plt.ylim([-1.4, 1.4])
    plt.grid()

    #*************************************************************
    const = np.array([-3, -1, 1, 3])/np.sqrt(10)
    thr = np.diff(const)[0]/2
    bins = const[:-1] + thr
    s_exc = np.zeros((2, np.prod(np.shape(est)[1:]))) + 0j
    s_dec = np.zeros((2, np.prod(np.shape(est)[1:]))) + 0j

    s_dec[0] = np.digitize(est[0].flatten(), bins)*2*thr + 1j * \
        np.digitize(est[1].flatten(), bins)*2*thr - 3*thr - 3j*thr
    s_exc[0] = (NN_output_test_reshaped[:, 64:-64:2, 0] + 1j *
                NN_output_test_reshaped[:, 64:-64:2, 1]).flatten()
    #s_dec[1] = np.digitize(np.real(s_rec[1]),bins)*2*thr + 1j * np.digitize(np.imag(s_rec[1]),bins)*2*thr - 3*thr - 3j*thr
    eps = 1e-8
    symbol_errors = np.sum(
        abs(s_dec[0] - s_exc[0]) > eps) + np.sum(abs(s_dec[1] - s_exc[1]) > eps)

    b_dec = s2b(np.round(s_dec[0].flatten()*np.sqrt(10)))
    b_exc = s2b(np.round(s_exc[0].flatten()*np.sqrt(10)))
    bits_errors = np.sum(b_dec != b_exc)

    BER = bits_errors/len(b_exc)
    print('\n\n\nBER achieved by NN=')
    print(BER)
    Qfac_dB_NN = Qfac(BER)
    print('corresponding Q-fac=')
    print(Qfac_dB_NN)
    NNXY.append(Qfac_dB_NN)

    #*************************************************************
    #DBP
    s_exc_DBP = np.zeros((2, N_s)) + 0j
    s_dec_DBP = np.zeros((2, N_s)) + 0j

    s_dec_DBP[0] = np.digitize(np.real(q0_DBP[1, 0, ::8]/np.sqrt(p)).flatten(), bins)*2*thr + 1j * \
        np.digitize(np.imag(q0_DBP[1, 0, ::8]/np.sqrt(p)
                            ).flatten(), bins)*2*thr - 3*thr - 3j*thr
    s_exc_DBP[0] = (s[1, 0]/np.sqrt(p)).flatten()
    #s_dec[1] = np.digitize(np.real(s_rec[1]),bins)*2*thr + 1j * np.digitize(np.imag(s_rec[1]),bins)*2*thr - 3*thr - 3j*thr
    eps = 1e-8
    symbol_errors = np.sum(abs(
        s_dec_DBP[0] - s_exc_DBP[0]) > eps) + np.sum(abs(s_dec_DBP[1] - s_exc_DBP[1]) > eps)

    b_dec_DBP = s2b(np.round(s_dec_DBP[0].flatten()*np.sqrt(10)))
    b_exc_DBP = s2b(np.round(s_exc_DBP[0].flatten()*np.sqrt(10)))
    bits_errors_DBP = np.sum(b_dec_DBP != b_exc_DBP)

    print(bits_errors_DBP)

    BER_DBP = bits_errors_DBP/len(b_exc_DBP)
    print('\n\n\nBER achieved by DBP=')
    print(BER_DBP)
    Qfac_dB_NN = Qfac(BER_DBP)
    print('corresponding Q-fac=')
    print(Qfac_dB_NN)
    DBP.append(Qfac_dB_NN)

    #*************************************************************
    #cpe
    s_exc_Linear = np.zeros((2, N_s)) + 0j
    s_dec_Linear = np.zeros((2, N_s)) + 0j

    s_dec_Linear[0] = np.digitize(np.real(q0_cpe[1, 0, ::8]/np.sqrt(p)).flatten(), bins)*2*thr + \
        1j * np.digitize(np.imag(q0_cpe[1, 0, ::8]/np.sqrt(p)
                                 ).flatten(), bins)*2*thr - 3*thr - 3j*thr
    s_exc_Linear[0] = (s[1, 0]/np.sqrt(p)).flatten()
    #s_dec[1] = np.digitize(np.real(s_rec[1]),bins)*2*thr + 1j * np.digitize(np.imag(s_rec[1]),bins)*2*thr - 3*thr - 3j*thr
    eps = 1e-8
    symbol_errors = np.sum(abs(s_dec_Linear[0] - s_exc_Linear[0]) > eps) + np.sum(
        abs(s_dec_Linear[1] - s_exc_Linear[1]) > eps)

    b_dec_Linear = s2b(np.round(s_dec_Linear[0].flatten()*np.sqrt(10)))
    b_exc_Linear = s2b(np.round(s_exc_Linear[0].flatten()*np.sqrt(10)))
    bits_errors_Linear = np.sum(b_dec_Linear != b_exc_Linear)

    print(bits_errors_Linear)

    BER_Linear = bits_errors_Linear/len(b_exc_Linear)
    print('\n\n\nBER achieved by Linear Equalizer=')
    print(BER_Linear)
    Qfac_dB_NN = Qfac(BER_Linear)
    print('corresponding Q-fac=')
    print(Qfac_dB_NN)
    Linear.append(Qfac_dB_NN)
