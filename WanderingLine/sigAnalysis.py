from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian

def plotFFT(x, y, Ts, filename='FFTPlot'):
    N = len(x)
    yf = fft(y)
    xf = fftfreq(N, Ts)[:N//2]
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]), '-')
    plt.xlabel('frequency (Hz)'); plt.ylabel('FFT')
    plt.grid()
    plt.savefig(r'C:\Users\casey\Desktop\REU24\WanderingLine\SignalPlots\\' + filename + '.png')
    plt.show()

def plotSpectrogram(times, data, Ts, fmax, filename='SpectrogramPlot'):
    N = len(data); flim = int(1.5*fmax)

    g_std = 8  # standard deviation for Gaussian window in samples
    w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(w, hop=10, fs=1/Ts, mfft=200, scale_to='psd')
    Sx = SFT.stft(data)  # perform the STFT

    fig1, ax1 = plt.subplots(figsize=(10., 6.))
    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    ax1.set_title(rf"STFT ({round(SFT.m_num*SFT.T, 2):g}$\,s$ Gaussian window, " +
                rf"$\sigma_t={round(g_std*SFT.T, 2)}\,$s)")
    ax1.set(xlabel=f"Time $t$ in seconds ({round(SFT.p_num(N), 2)} slices, " +
                rf"$\Delta t = {round(SFT.delta_t, 2):g}\,$s)",
            ylabel=f"Freq. $f$ in Hz ({round(SFT.f_pts, 2)} bins, " +
                rf"$\Delta f = {round(SFT.delta_f, 2):g}\,$Hz)",
            xlim=(t_lo, t_hi), ylim=(0, flim))

    im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                    extent=SFT.extent(N), cmap='viridis')
    fig1.colorbar(im1, label="PSD")
    fig1.tight_layout()
    plt.savefig(r'C:\Users\casey\Desktop\REU24\WanderingLine\SignalPlots\\' + filename + '.png')
    plt.show()

def plotAllTime(signal, Ts, fmax, filename='TimePlots'):
    plt.figure(figsize=(15, 10))
    plot1 = plt.subplot2grid((4, 1), (0, 0)) 
    plot2 = plt.subplot2grid((4, 1), (1, 0))
    plot3 = plt.subplot2grid((4, 1), (2, 0)) 
    plot4 = plt.subplot2grid((4, 1), (3, 0)) 

    N = len(signal[0]); flim = int(1.5*fmax)

    g_std = 8  # standard deviation for Gaussian window in samples
    w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(w, hop=10, fs=1/Ts, mfft=200, scale_to='psd')
    Sx = SFT.stft(signal[1])  # perform the STFT

    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    plot4.set_title(rf"Short FFTs of iid Signal in Spectrogram")
    plot4.set(xlabel=f"t (s)",
            ylabel=f"freq. (Hz)",
            xlim=(t_lo, t_hi), ylim=(0, flim))

    im1 = plot4.imshow(abs(Sx), origin='lower', aspect='auto',
                    extent=SFT.extent(N), cmap='viridis')

    plot1.plot(signal[0], signal[4], color='hotpink') 
    plot1.plot(signal[3][:, 0], signal[3][:, 1], 'o', color = 'pink')
    plot1.set(ylabel='freq (Hz)', xlim=(t_lo, t_hi), ylim=(0, flim), title='Frequency(Hz)')

    plot2.plot(signal[0], signal[2], '-', color='g') 
    # axis values are for some reason floats, need to make adaptable
    plot2.set(ylabel='q(t)', xlim=(t_lo, t_hi), title='Clean Signal') 

    plot3.plot(signal[0], signal[1], 'o', color='g') 
    plot3.set(ylabel='q(t)', xlim=(t_lo, t_hi), title='Dist. Signal') 
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\casey\Desktop\REU24\WanderingLine\SignalPlots\\' + filename + '.png')
    plt.show() 

def plotPSD(data, Ts, filename='PSDLinPlot'):
    plt.figure()
    plot1 = plt.subplot2grid((2, 1), (0, 0)) 
    plot2 = plt.subplot2grid((2, 1), (1, 0))

    power, freqs = plot1.psd(data, 512, 1 / Ts, scale_by_freq=False)
    plot1.set_title('Logarithmic Plot'); plot1.set_xlabel('')
    plot2.plot(freqs, power); plot2.set_ylabel('Power')
    plot2.set_title('Linear Plot'); plot2.set_xlabel('Freq. (Hz)')
    
    plt.tight_layout()
    plt.savefig(r'C:\Users\casey\Desktop\REU24\WanderingLine\SignalPlots\\' + filename + '.png')
    plt.show()

def plotSpectComp(times, Sdata, spline, Ts, fmax, filename='SpectCompPlot'):
    N = len(Sdata); flim = int(1.5*fmax)

    g_std = 8  # standard deviation for Gaussian window in samples
    w = gaussian(50, std=g_std, sym=True)  # symmetric Gaussian window
    SFT = ShortTimeFFT(w, hop=10, fs=1/Ts, mfft=200, scale_to='psd')
    Sx = SFT.stft(Sdata)  # perform the STFT

    fig1, ax1 = plt.subplots(figsize=(10., 6.))
    t_lo, t_hi = SFT.extent(N)[:2]  # time range of plot
    ax1.set_title('Spectrogram of iid Signal and Original Frequency Spline')
    ax1.set(xlabel=f"Time (s)",
            ylabel=f"Freq. (Hz)",
            xlim=(t_lo, t_hi), ylim=(0, flim))

    im1 = ax1.imshow(abs(Sx), origin='lower', aspect='auto',
                    extent=SFT.extent(N), cmap='viridis')
    fig1.colorbar(im1, label="PSD")
    ax1.plot(times, spline, '--', color='r')
    plt.legend(['Frequency Spline'], loc='upper left')
    fig1.tight_layout()
    plt.savefig(r'C:\Users\casey\Desktop\REU24\WanderingLine\SignalPlots\\' + filename + '.png')
    plt.show()