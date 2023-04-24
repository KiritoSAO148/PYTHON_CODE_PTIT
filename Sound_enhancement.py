from math import *
import io, os, sys, time
import array as arr
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import soundfile as sf
import librosa as lb
import scipy as sp
import sympy
import cmath
import timeit
from datetime import datetime

def plot_window(w, title):
    plt.stem(w)
    plt.title(title)
    plt.xlabel('n')
    plt.ylabel('w(n)')
    plt.show()

def plot_rec_window_1(N):
    w = np.linspace(-np.pi, np.pi, 1000)
    A = (np.sin(w * N / 2)) / (N * np.sin(w / 2))
    AdB = 20 * np.log10(np.abs(A))
    plt.plot(w, AdB)
    plt.title('Amplitude Response in dB of Rectangular Window')
    plt.xlabel('Tần số góc (rad)')
    plt.ylabel('Đáp ứng biên độ (dB)')
    plt.grid(True)
    plt.show()

def plot_rec_window_2(N):
    w = np.linspace(-np.pi, np.pi, 10000)
    Wr = np.sin(w * N / 2) / np.sin(w / 2)
    plt.plot(w, Wr)
    plt.xlabel('ω')
    plt.ylabel('Wr(ω)')
    plt.title('Amplitude Response of Rectangular Window')
    plt.grid(True)
    plt.show()

def plot_bartlett_window_1(N):
    w = np.linspace(-np.pi, np.pi, 1000)
    A = np.sin((w * (N - 1) / 2) / 2) / (N * np.sin(w / 2))
    AdB = 20 * np.log10(np.abs(A))
    plt.plot(w, AdB)
    plt.title('Amplitude Response in dB of Bartlett Window')
    plt.xlabel('Tần số góc (rad)')
    plt.ylabel('Đáp ứng biên độ (dB)')
    plt.grid(True)
    plt.show()

def plot_barlett_window_2(N):
    w = np.linspace(-np.pi, np.pi, 1000)
    Wt = np.sin((w * (N - 1) / 2) / 2) / (np.sin(w / 2))
    plt.plot(w, Wt)
    plt.xlabel('ω')
    plt.ylabel('Wt(ω)')
    plt.title('Amplitude Response of Barlett Window')
    plt.grid(True)
    plt.show()

def plot_impulse_response1(h, title):
    plt.stem(h)
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.title(title)
    plt.grid()
    plt.show()

def plot_impulse_response2(hd, title):
    plt.stem(hd)
    plt.xlabel('n')
    plt.ylabel('hd[n]')
    plt.title(title)
    plt.grid()
    plt.show()

def plot_freq_response(hd, title, N):
    H = np.fft.fft(hd)
    H = np.fft.fftshift(H)
    f = np.linspace(-1, 1, N)
    plt.plot(f, np.abs(H))
    plt.xlabel('f')
    plt.ylabel('|H(f)|')
    plt.grid(True)
    plt.title(title)
    plt.show()

def plot_signal(signal, hd, indices, title, fs):
    y = scipy.signal.convolve(signal, hd, mode='same')
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Kết quả lọc tín hiệu')
    ax1.plot(indices, signal, label='Trước lọc')
    ax1.plot(indices, y, label='Sau lọc')
    ax1.set_xlabel('Thời gian (giây)')
    ax1.set_ylabel('Biên độ')
    ax1.legend()
    N = len(signal)
    X = scipy.fftpack.fft(signal)
    Y = scipy.fftpack.fft(y)
    freqs = scipy.fftpack.fftfreq(N, 1 / fs)
    ax2.plot(freqs[:N // 2], np.abs(X[:N // 2]) / N, label='Trước lọc')
    ax2.plot(freqs[:N // 2], np.abs(Y[:N // 2]) / N, label='Sau lọc')
    ax2.set_xlabel('Tần số (Hz)')
    ax2.set_ylabel('Biên độ')
    ax2.legend()
    plt.show()
    signal_fft = np.abs(np.fft.fft(signal))
    filtered_signal = np.convolve(signal, hd, mode='same')
    filtered_signal_fft = np.abs(np.fft.fft(filtered_signal))
    freq_axis = np.linspace(0, N, int(N / 2) + 1)
    plt.plot(freq_axis, signal_fft[:len(freq_axis)], label='Tín hiệu gốc')
    plt.plot(freq_axis, filtered_signal_fft[:len(freq_axis)], label='Tín hiệu đã lọc')
    plt.xlabel('Tần số (tính bằng đơn vị tần số lấy mẫu)')
    plt.ylabel('Biên độ phổ')
    plt.title(title)
    plt.legend()
    plt.show()

def rectangular_window(N):
    w = np.ones(N)
    plot_window(w, 'Retangular Window')
    plot_rec_window_1(N)
    plot_rec_window_2(N)
    return w

def bartlett_window(N):
    a = []
    for i in range(N):
        if i >= 0 and i <= (N - 1) / 2: a.append((2 * i) / (N - 1))
        elif i >= (N - 1) / 2 and i <= N - 1: a.append(2 - ((2 * i) / (N - 1)))
    w = np.array(a)
    # print(w)
    # w = scipy.signal.windows.bartlett(N)
    plot_window(w, 'Barlett Window')
    plot_bartlett_window_1(N)
    plot_barlett_window_2(N)
    return w

def hanning_window(N):
    a = []
    for i in range(N): a.append(0.5 - 0.5 * cos((2 * pi * i) / (N - 1)))
    w = np.array(a)
    # w = scipy.signal.windows.hann(N)
    plot_window(w, 'Hanning Window')
    return w

def hamming_window(N):
    a = []
    for i in range(N): a.append(0.54 - 0.46 * cos((2 * pi * i) / (N - 1)))
    w = np.array(a)
    # w = scipy.signal.windows.hamming(N)
    plot_window(w, 'Hamming Window')
    return w

def blackman_window(N):
    a = []
    for i in range(N): a.append(0.42 - 0.5 * cos((2 * pi * i) / (N - 1)) + 0.08 * cos((4 * pi * i) / (N - 1)))
    w = np.array(a)
    # w = scipy.signal.windows.blackman(N)
    plot_window(w, 'Blackman Window')
    return w

def kaiser_window(N, beta):
    w = scipy.signal.windows.kaiser(N, beta)
    plot_window(w, 'Kaiser Window')
    return w

def lowpass_filter(f_c, N, fs):
    f_c_norm = f_c / fs
    w_c_norm = 2 * pi * f_c_norm
    M = (N - 1) // 2
    h = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: h[n] = w_c_norm / pi
        else: h[n] = sin(w_c_norm * (n - M)) / (pi * (n - M))
    plot_impulse_response1(h, 'Đáp ứng xung của bộ lọc thông thấp')
    return h

def highpass_filter(f_c, N, fs):
    f_c_norm = f_c / fs
    w_c_norm = 2 * pi * f_c_norm
    M = (N - 1) // 2
    h = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: h[n] = 1 - w_c_norm / pi
        else: h[n] = -sin(w_c_norm * (n - M)) / (pi * (n - M))
    plot_impulse_response2(h, 'Đáp ứng xung của bộ lọc thông cao')
    return h

def bandpass_filter(f_c1, f_c2, N, fs):
    # N: số mẫu trong một chu kỳ xung
    f_c1_norm = f_c1 / fs
    f_c2_norm = f_c2 / fs
    w_c1_norm = 2 * pi * f_c1_norm
    w_c2_norm = 2 * pi * f_c2_norm
    M = (N - 1) // 2
    h = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: h[n] = w_c2_norm / pi - w_c1_norm / pi
        else: h[n] = sin(w_c2_norm * (n - M)) / (pi * (n - M)) - sin(w_c1_norm * (n - M)) / (pi * (n - M))
    plot_impulse_response2(h, 'Đáp ứng xung của bộ lọc thông dải')
    return h

def bandstop_filter(f_c1, f_c2, N, fs):
    f_c1_norm = f_c1 / fs
    f_c2_norm = f_c2 / fs
    w_c1_norm = 2 * pi * f_c1_norm
    w_c2_norm = 2 * pi * f_c2_norm
    M = (N - 1) // 2
    h = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: h[n] = 1 - w_c2_norm / pi + w_c1_norm / pi
        else: h[n] = -sin(w_c2_norm * (n - M)) / (pi * (n - M)) + sin(w_c1_norm * (n - M)) / (pi * (n - M))
    plot_impulse_response2(h, 'Đáp ứng xung của bộ lọc chắn dải')
    return h

def lpf(N, f_c, fs, type_of_window):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = bartlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = lowpass_filter(f_c, N, fs)
    hd = window * filter  # Hàm đáp ứng xung của cửa sổ thiết kế được
    # # Tính đáp ứng tần số của bộ lọc thông thấp
    plot_impulse_response2(hd, 'Đáp ứng xung của bộ lọc thông thấp sau khi qua cửa sổ')
    plot_freq_response(hd, 'Đồ hình đáp ứng tần số của bộ lọc thông thấp', N)
    return hd

def hpf(N, f_c, fs, type_of_window):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = bartlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = highpass_filter(f_c, N, fs)
    hd = window * filter  # Hàm đáp ứng xung của cửa sổ thiết kế được
    # # Tính đáp ứng tần số của bộ lọc thông cao
    plot_impulse_response2(hd, 'Đáp ứng xung của bộ lọc thông cao sau khi qua cửa sổ')
    plot_freq_response(hd, 'Đồ hình đáp ứng tần số của bộ lọc thông cao', N)
    return hd

def bpf(N, f_c1, f_c2, fs, type_of_window):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = bartlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = bandpass_filter(f_c1, f_c2, N, fs)
    hd = window * filter  # Hàm đáp ứng xung của cửa sổ thiết kế được
    # Tính đáp ứng tần số của bộ lọc thông dải
    plot_impulse_response2(hd, 'Đáp ứng xung của bộ lọc thông dải sau khi qua cửa sổ')
    plot_freq_response(hd, 'Đồ hình đáp ứng tần số của bộ lọc thông dải', N)
    return hd

def bsf(N, f_c1, f_c2, fs, type_of_window):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = bartlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = bandstop_filter(f_c1, f_c2, N, fs)
    hd = window * filter  # Hàm đáp ứng xung của cửa sổ thiết kế được
    # Tính đáp ứng tần số của bộ lọc thông cao
    plot_impulse_response2(hd, 'Đáp ứng xung của bộ lọc chắn dải sau khi qua cửa sổ')
    plot_freq_response(hd, 'Đồ hình đáp ứng tần số của bộ lọc chắn dải', N)
    return hd

def eq(num_lpf, lpf_cutoff_freqs, num_bpf, bpf_cutoff_freqs, num_hpf, hpf_cutoff_freqs, filter_order, window, fs):
    filter_coeffs = np.zeros((num_lpf + num_bpf + num_hpf, filter_order + 1))
    for i in range(num_lpf):
        filter_coeffs[i, :] = scipy.signal.firwin(filter_order + 1, lpf_cutoff_freqs[i], window=window, pass_zero='lowpass', fs=fs)
    for i in range(num_bpf):
        filter_coeffs[num_lpf + i, :] = scipy.signal.firwin(filter_order + 1, bpf_cutoff_freqs[i], window=window,
                                                            pass_zero=False, fs=fs)
    for i in range(num_hpf):
        filter_coeffs[num_lpf + num_bpf + i, :] = scipy.signal.firwin(filter_order + 1, hpf_cutoff_freqs[i], window=window,
                                                                      pass_zero='highpass', fs=fs)
    filter_coeffs = filter_coeffs / np.sum(filter_coeffs, axis=1)[:, np.newaxis]
    eq_filter_coeffs = np.concatenate(filter_coeffs, axis=0)
    return eq_filter_coeffs

def eqf(num_lpf, lpf_cutoff_freqs, num_bpf, bpf_cutoff_freqs, num_hpf, hpf_cutoff_freqs, filter_order, window, fs):
    eq_filter_coeffs = eq(num_lpf, lpf_cutoff_freqs, num_bpf, bpf_cutoff_freqs, num_hpf, hpf_cutoff_freqs, filter_order, window, fs)
    w, h = scipy.signal.freqz(eq_filter_coeffs)
    fig, ax = plt.subplots()
    ax.plot(w, np.abs(h))
    ax.set_xlabel('Frequency (rad/sample)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Magnitude response of EQ filter')
    ax.grid(True)
    plt.show()

def signal_eq(signal, num_lpf, lpf_cutoff_freqs, num_bpf, bpf_cutoff_freqs, num_hpf, hpf_cutoff_freqs, filter_order, window, fs):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    eq_filter_coeffs = eq(num_lpf, lpf_cutoff_freqs, num_bpf, bpf_cutoff_freqs, num_hpf, hpf_cutoff_freqs, filter_order,
                          window, fs)
    y = scipy.signal.convolve(signal, eq_filter_coeffs, mode = 'same')
    plot_signal(signal, eq_filter_coeffs, indices, 'Đồ hình biên độ phổ của tín hiệu trước và sau khi qua bộ lọc thông thấp', fs)
    return y

def signal_lpf(signal, N, f_c, fs, type_of_window):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    hd = lpf(N, f_c, fs, type_of_window)
    y = scipy.signal.convolve(signal, hd, mode = 'same')
    plot_signal(signal, hd, indices, 'Đồ hình biên độ phổ của tín hiệu trước và sau khi qua bộ lọc thông thấp', fs)
    return y

def signal_hpf(signal, N, f_c, fs, type_of_window):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    hd = hpf(N, f_c, fs, type_of_window)
    y = scipy.signal.convolve(signal, hd, mode = 'same')
    plot_signal(signal, hd, indices, 'Đồ hình biên độ phổ của tín hiệu trước và sau khi qua bộ lọc thông cao', fs)
    return y

def signal_bpf(signal, N, f_c1, f_c2, fs, type_of_window):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    hd = bpf(N, f_c1, f_c2, fs, type_of_window)
    y = scipy.signal.convolve(signal, hd, mode = 'same')
    plot_signal(signal, hd, indices, 'Đồ hình biên độ phổ của tín hiệu trước và sau khi qua bộ lọc thông dải', fs)
    return y

def signal_bsf(signal, N, f_c1, f_c2, fs, type_of_window):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    hd = bsf(N, f_c1, f_c2, fs, type_of_window)
    y = scipy.signal.convolve(signal, hd, mode = 'same')
    plot_signal(signal, hd, indices, 'Đồ hình biên độ phổ của tín hiệu trước và sau khi qua bộ lọc chắn dải', fs)
    return y

if __name__ == '__main__':
    # N = 51; beta = 5.282
    # kaiser_window(N, beta)
    # lpf(N, 4000, 22050, 'Rectangular Window')
    # signal = np.random.randint(low=-10000, high=10000, size=101)
    signal, samperate = sf.read('Accident_-police-car.wav', frames = 44100 * 5)
    print(signal)
    res = [x[0] for x in signal]
    signal = res
    N = 51
    f_c = 4000
    f_c1 = 3000
    f_c2 = 4000
    fs = samperate
    num_filters = 5
    type_of_window = 'Hamming Window'
    y = signal_bsf(signal, N, f_c1, f_c2, fs, type_of_window)
    sf.write('bsf_01.wav', y, samperate)
    # num_lpf = 1
    # num_bpf = 4
    # num_hpf = 1
    # lpf_cutoff_freqs = [1000]
    # bpf_cutoff_freqs = [(2000, 2600), (3200, 3500), (4000, 4700), (5000, 5900)]
    # hpf_cutoff_freqs = [12000]
    # filter_order = 5
    # eqf(num_lpf, lpf_cutoff_freqs, num_bpf, bpf_cutoff_freqs, num_hpf, hpf_cutoff_freqs, filter_order, 'blackman',
    #     44100)
    # y = signal_eq(signal, num_lpf, lpf_cutoff_freqs, num_bpf, bpf_cutoff_freqs, num_hpf, hpf_cutoff_freqs, filter_order,
    #           'blackman', 44100)
    # window = rectangular_window(51)
    # window = hanning_window(N)
    # hamming_window(N)
    # window = blackman_window(N)