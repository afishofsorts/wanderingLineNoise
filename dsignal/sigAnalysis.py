from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

##############################
# PLOTS FOR SIMULATED SIGNAL #
##############################

# performs Fast Fourier Transform for coordinate data set
def FFT(x, y, Ts):
    # INPUTS:
    # x:         1D array of x values
    # y:         1D array of y values
    # Ts:        Sampling rate
    # OUTPUTS:
    # xf:        1D array of FFT frequency values
    # yg:        1D array of weighted FFT amplitudes

    N = len(x)
    yf = fft(y)
    xf = fftfreq(N, Ts)[:N//2] # slices positive half of fftfreq return
    yg = 2.0/N * np.abs(yf[0:N//2]) # slices first half to match xf and converts to weighted amplitude
    return xf, yg

# takes weighted FFT and returns coordinates for y values above threshold
def FFTPeaks(xf, yg, thresh = 0.0001):
    # INPUTS:
    # xf:        1D array of FFT frequency values
    # yg:        1D array of weighted FFT amplitudes
    # OUTPUTS:
    # peaks:     2D array of coordinates for y values above threshold

    peaks = [] # peaks list initialization
    for i in range(len(yg)-2):
        if thresh < yg[i]:
            peaks = np.append(peaks, xf[i]) # adds any FFT amplitudes above thresh
    return peaks

# plots Fast Fourier Transform for coordinate data set
def plotFFT(x, y, Ts, dir='none'):
    # INPUTS:
    # x:         1D array of x values
    # y:         1D array of y values
    # Ts:        Sampling rate
    # dir:       Directory to save plot PNG to
    # OUTPUTS:
    # xf:        1D array of FFT frequency values
    # yg:        1D array of weighted FFT amplitudes
    # Image:     PNG of FFT plot saved to dir

    xf, yg = FFT(x, y, Ts)
    plt.plot(xf, yg, '-')
    plt.xlabel('frequency (Hz)'); plt.ylabel('FFT')
    plt.grid()
    if dir=='none':
        plt.show()
    else:
        plt.savefig(dir)
        plt.close
    return xf, yg

# plots spectrogram of input data using short FFT
def plotSpectrogram(data, Ts, fmax, hop: int, fmin=0, dir='none', vmax=0):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # data:      1D array
    # Ts:        Sampling rate
    # fmax:      Maximum frequency in data
    # dir:       Directory to save plot PNG to
    # OUTPUTS:
    # Image:     PNG of FFT plot saved to dir

    M = hop*5; g_std = int(hop*8/10); mfft = hop*10

    N = len(data); 

    w = gaussian(M, std=g_std, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(w, hop=hop, fs=1/Ts, mfft=mfft, scale_to='psd')
    Sx = SFT.stft(data)  # perform the STFT

    fig1, ax1 = plt.subplots(figsize=(10., 6.))
    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    ax1.set_title(rf"STFT ({round(SFT.m_num*SFT.T, 2):g}$\,s$ Gaussian window, " +
                rf"$\sigma_t={round(g_std*SFT.T, 2)}\,$s)")
    ax1.set(xlabel=f"Time $t$ in seconds ({round(SFT.p_num(N), 2)} slices, " +
                rf"$\Delta t = {round(SFT.delta_t, 2):g}\,$s)",
            ylabel=f"Freq. $f$ in Hz ({round(SFT.f_pts, 2)} bins, " +
                rf"$\Delta f = {round(SFT.delta_f, 2):g}\,$Hz)",
            xlim=(t_lo, t_hi), ylim=(fmin, fmax))

    if vmax == 0:
        im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                        extent=SFT.extent(N), cmap='viridis')
    else:
        im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                        extent=SFT.extent(N), cmap='viridis', vmax=vmax)
    fig1.colorbar(im1, label="PSD")
    fig1.tight_layout()
    if dir=='none':
        plt.show()
    else:
        plt.savefig(dir)
        plt.close()

# plots frequency, resulting sin wave, iid sin wave, and spectrogram of iid against time
def plotAllTime(t, cleanSig, distSig, freqs, freqKnots, Ts, fmax, dir='none'):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # cleanSig:  1D array of clean signal
    # distSig:   1D array of cleanSig with added iid
    # freqs:     1D frequency array
    # freqKnots: 2-tuple of time and frequency values used for BSpline generation  
    # Ts:        Sampling rate
    # fmax:      Maximum frequency in data
    # dir:       Directory to save plot PNG to
    # OUTPUTS:
    # Image:     PNG of FFT plot saved to dir

    plt.figure(figsize=(15, 10)) # plot setup and structure
    plot1 = plt.subplot2grid((4, 1), (0, 0)) 
    plot2 = plt.subplot2grid((4, 1), (1, 0))
    plot3 = plt.subplot2grid((4, 1), (2, 0)) 
    plot4 = plt.subplot2grid((4, 1), (3, 0)) 

    flim = int(1.5*fmax) # increased ceiling for fmax due to iid

    # Spectrogram plot
    N = len(t); g_std = 8  # standard deviation for Gaussian window in samples
    w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(w, hop=10, fs=1/Ts, mfft=200, scale_to='psd')
    Sx = SFT.stft(distSig)  # perform the STFT

    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    plot4.set_title(rf"Short FFTs of i.i.d. Signal in Spectrogram")
    plot4.set(xlabel=f"t (s)", ylabel=f"Frequency (Hz)", xlim=(t_lo, t_hi), ylim=(0, flim))
    im1 = plot4.imshow(abs(Sx), origin='lower', aspect='auto', extent=SFT.extent(N), cmap='viridis') # heat mapping

    # Frequency plot
    plot1.plot(t, freqs, color='hotpink')
    plot1.plot(freqKnots[0], freqKnots[1], 'o', color = 'pink')
    plot1.set(ylabel='Frequency (Hz)', xlim=(t_lo, t_hi), ylim=(0, flim), title='Cubic B-Spline')

    # Clean signal plot
    plot2.plot(t, cleanSig, '-', color='g') 
    plot2.set(ylabel='Strain', xlim=(t_lo, t_hi), title='Clean Signal') 

    # iid signal plot
    plot3.plot(t, distSig, 'o', color='g') 
    plot3.set(ylabel='Strain', xlim=(t_lo, t_hi), title='Dist. Signal') 
    
    plt.tight_layout()
    if dir=='none':
        plt.show()
    else:
        plt.savefig(dir)
        plt.close()

# Plots Power Spectral Density of input data with logarithmic and linear plotting
def plotPSD(data, Ts, title, dir='none', linear=True):
    # INPUTS:
    # data:      1D array
    # Ts:        Sampling rate
    # title:     Plot title
    # dir:       Directory to save plot PNG to
    # OUTPUTS:
    # Image:     PNG of FFT plot saved to dir

    power, freqs = plt.psd(data, 512, 1 / Ts, scale_by_freq=False) # performs psd
    if linear:
        plt.close()
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.plot(freqs, power); plt.ylabel('Intensity (counts)')
    
    plt.tight_layout()
    if dir=='none':
        plt.show()
    else:
        plt.savefig(dir)
        plt.close()

# Plots input frequency signal over spectrogram of some data using short FFT 
def plotSpectComp(t, SpecData, freqs, freqKnots, Ts, fmax, hop: int, fmin=0, dir='none', vmax=0, title=''):
    # INPUTS:
    # t:         1D time array with Ts spacing
    # SpectData: 1D array of data for spectroscopy
    # freqs:     1D frequency array
    # Ts:        Sampling rate
    # fmax:      Maximum frequency in data
    # title:     Header for plot
    # dir:       Directory to save plot PNG to
    # OUTPUTS:
    # Image:     PNG of FFT plot saved to dir

    fig1, ax1 = plt.subplots(figsize=(10., 6.)) # plot setup and structure

    M = hop*5; g_std = int(hop*8/10); mfft = hop*20

    N = len(SpecData)

    w = gaussian(M, std=g_std, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(w, hop=hop, fs=1/Ts, mfft=mfft, scale_to='psd')
    Sx = SFT.stft(SpecData)  # perform the STFT

    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    ax1.set_title(title)
    ax1.set(xlabel=f"Time (s)", ylabel=f"Frequency (Hz)", xlim=(t_lo, t_hi), ylim=(fmin, fmax))

    if vmax == 0:
        im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto', extent=SFT.extent(N), cmap='viridis') # heat mapping
    else:
        im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto', extent=SFT.extent(N), cmap='viridis', vmax=vmax)
    fig1.colorbar(im1, label="PSD")

    ax1.plot(t, freqs, '--', color='r') # adds input frequency
    ax1.plot(freqKnots[0], freqKnots[1], 'o', color = 'r')
    plt.legend(['Frequency Spline'], loc='upper left')
    fig1.tight_layout()
    if dir=='none':
        plt.show()
    else:
        plt.savefig(dir)
        plt.close()