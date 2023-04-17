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

def plot_impulse_response(h, title):
    plt.stem(h)
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.title(title)
    plt.grid()
    plt.show()

def plot_freq_response(hd, title, N):
    # Tính đáp ứng tần số của bộ lọc thông thấp
    H = np.fft.fft(hd)
    H = np.fft.fftshift(H)
    # H_db = 20 * np.log10(np.abs(np.fft.fftshift(H)))
    w = np.linspace(-pi, pi, N)

    # Vẽ đồ thị đáp ứng tần số
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(title)
    ax1.plot(w, np.abs(H))
    # plt.plot(f, H_db)
    ax1.set_xlabel('pi')
    ax1.set_ylabel('|H(w)|')
    ax1.grid()
    # ax1.legend()

    # Áp dụng phép biến đổi Fourier để tính toán biên độ phổ
    freq_response = np.abs(np.fft.fft(hd))
    freq_response = np.fft.fftshift(freq_response)

    # Vẽ đồ thị biên độ phổ
    freq_axis = np.linspace(-1, 1, N)  # trục tần số
    ax2.plot(freq_axis, 20 * np.log10(freq_response))  # chuyển đổi sang đơn vị dB
    ax2.set_xlabel('Tần số')
    ax2.set_ylabel('Biên độ (dB)')
    ax2.grid()
    # ax2.legend()

    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig.suptitle('Kết quả lọc tín hiệu')
    # ax1.plot(indices, signal, label='Trước lọc')
    # ax1.plot(indices, y, label='Sau lọc')
    # ax1.set_xlabel('Thời gian (giây)')
    # ax1.set_ylabel('Biên độ')
    # ax1.legend()
    # N = len(signal)
    # X = scipy.fftpack.fft(signal)
    # Y = scipy.fftpack.fft(y)
    # freqs = scipy.fftpack.fftfreq(N, 1 / fs)
    #
    # ax2.plot(freqs[:N // 2], np.abs(X[:N // 2]) / N, label='Trước lọc')
    # ax2.plot(freqs[:N // 2], np.abs(Y[:N // 2]) / N, label='Sau lọc')
    # ax2.set_xlabel('Tần số (Hz)')
    # ax2.set_ylabel('Biên độ')
    # ax2.legend()

    plt.show()

def rectangular_window(N):
    w = np.ones(N)
    plot_window(w, 'Retangular Window')
    return w

def barlett_window(N):
    a = []
    for i in range(N):
        if i >= 0 and i <= (N - 1) / 2: a.append((2 * i) / (N - 1))
        elif i >= (N - 1) / 2 and i <= N - 1: a.append(2 - ((2 * i) / (N - 1)))
    w = np.array(a)
    # print(w)
    # w = scipy.signal.windows.bartlett(N)
    plot_window(w, 'Barlett Window')
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
    plot_impulse_response(h, 'Đáp ứng xung của bộ lọc thông thấp')
    return h

def highpass_filter(f_c, N, fs):
    f_c_norm = f_c / fs
    w_c_norm = 2 * pi * f_c_norm
    M = (N - 1) // 2
    h = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: h[n] = 1 - w_c_norm / pi
        else: h[n] = -sin(w_c_norm * (n - M)) / (pi * (n - M))
    plot_impulse_response(h, 'Đáp ứng xung của bộ lọc thông cao')
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
    plot_impulse_response(h, 'Đáp ứng xung của bộ lọc thông dải')
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
    plot_impulse_response(h, 'Đáp ứng xung của bộ lọc chắn dải')
    return h

def equalizer(f_c, N, fs):
    f_c_norm = f_c / fs
    w_c_norm = 2 * pi * f_c_norm
    M = (N - 1) // 2
    h = np.zeros(N)
    for i in range(N):
        h[i] = 2 * f_c_norm * np.sinc(2 * f_c_norm * (i - (M)))
    plot_impulse_response(h, 'Đáp ứng xung của bộ lọc equalizer')
    return h

def lpf(N, f_c, fs, type_of_window):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = barlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = lowpass_filter(f_c, N, fs)
    hd = window * filter  # Hàm đáp ứng xung của cửa sổ thiết kế được
    # Tính đáp ứng tần số của bộ lọc thông thấp
    H = np.fft.fft(hd)
    H = np.fft.fftshift(H)
    # H_db = 20 * np.log10(np.abs(np.fft.fftshift(H)))
    f = np.linspace(-1, 1, N)

    # Vẽ đồ thị đáp ứng tần số
    plt.plot(f, np.abs(H))
    # plt.plot(f, H_db)
    plt.xlabel('f')
    plt.ylabel('|H(f)|')
    plt.title('Đáp ứng tần số của bộ lọc thông thấp')
    plt.grid()
    plt.show()

    # Áp dụng phép biến đổi Fourier để tính toán biên độ phổ
    freq_response = np.abs(np.fft.fft(hd))
    freq_response = np.fft.fftshift(freq_response)

    # Vẽ đồ thị biên độ phổ
    freq_axis = np.linspace(-1, 1, N)  # trục tần số
    plt.plot(freq_axis, 20 * np.log10(freq_response))  # chuyển đổi sang đơn vị dB
    plt.xlabel('Tần số (tính bằng đơn vị tần số cắt)')
    plt.ylabel('Biên độ (dB)')
    plt.title('Đồ hình biên độ phổ của bộ lọc thông thấp')
    plt.show()
    return hd

def hpf(N, f_c, fs, type_of_window):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = barlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = highpass_filter(f_c, N, fs)
    hd = window * filter  # Hàm đáp ứng xung của cửa sổ thiết kế được
    # # Tính đáp ứng tần số của bộ lọc thông cao
    # H = np.fft.fft(hd)
    # H = np.fft.fftshift(H)
    # f = np.linspace(-1, 1, N)
    #
    # # Vẽ đồ thị đáp ứng tần số
    # plt.plot(f, np.abs(H))
    # plt.xlabel('f')
    # plt.ylabel('|H(f)|')
    # plt.title('Đáp ứng tần số của bộ lọc thông cao')
    # plt.grid()
    # plt.show()
    #
    # # Áp dụng phép biến đổi Fourier để tính toán biên độ phổ
    # freq_response = np.abs(np.fft.fft(hd))
    # freq_response = np.fft.fftshift(freq_response)
    #
    # # Vẽ đồ thị biên độ phổ
    # freq_axis = np.linspace(0, 1, N)  # trục tần số
    # plt.plot(freq_axis, 20 * np.log10(freq_response))  # chuyển đổi sang đơn vị dB
    # plt.xlabel('Tần số (tính bằng đơn vị tần số cắt)')
    # plt.ylabel('Biên độ (dB)')
    # plt.title('Đồ hình biên độ phổ của bộ lọc thông thấp')
    # plt.show()
    plot_freq_response(hd, 'Đồ hình đáp ứng tần số của bộ lọc thông cao', N)
    return hd

def bpf(N, f_c1, f_c2, fs, type_of_window):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = barlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = bandpass_filter(f_c1, f_c2, N, fs)
    hd = window * filter  # Hàm đáp ứng xung của cửa sổ thiết kế được
    # Tính đáp ứng tần số của bộ lọc thông dải
    H = np.fft.fft(hd)
    H = np.fft.fftshift(H)
    f = np.linspace(-1, 1, N)

    # Vẽ đồ thị đáp ứng tần số
    plt.plot(f, np.abs(H))
    plt.xlabel('f')
    plt.ylabel('|H(f)|')
    plt.title('Đáp ứng tần số của bộ lọc thông dải')
    plt.grid()
    plt.show()
    return hd

def bsf(N, f_c1, f_c2, fs, type_of_window):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = barlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = bandstop_filter(f_c1, f_c2, N, fs)
    hd = window * filter  # Hàm đáp ứng xung của cửa sổ thiết kế được
    # Tính đáp ứng tần số của bộ lọc chắn dải
    H = np.fft.fft(hd)
    H = np.fft.fftshift(H)
    f = np.linspace(-1, 1, N)

    # Vẽ đồ thị đáp ứng tần số
    plt.plot(f, np.abs(H))
    plt.xlabel('f')
    plt.ylabel('|H(f)|')
    plt.title('Đáp ứng tần số của bộ lọc chắn dải')
    plt.grid()
    plt.show()
    return hd

def eq(N, f_c, fs, type_of_window, num_filters):
    window = np.zeros(N)
    if type_of_window == 'Rectangular Window': window = rectangular_window(N)
    elif type_of_window == 'Barlett Window': window = barlett_window(N)
    elif type_of_window == 'Hanning Window': window = hanning_window(N)
    elif type_of_window == 'Hamming Window': window = hamming_window(N)
    elif type_of_window == 'Blackman Window': window = blackman_window(N)
    else:
        beta = int(input('Beta: '))
        window = kaiser_window(N, beta)
    filter = equalizer(f_c, N, fs)
    hd = window * filter # Hàm đáp ứng xung của cửa sổ thiết kế được
    h_eq = hd
    for i in range(num_filters - 1):
        h_eq = np.convolve(h_eq, hd)
    # Tính đáp ứng tần số của bộ lọc chắn dải
    H = np.fft.fft(h_eq)
    # H = np.fft.fftshift(H)

    # Vẽ đồ thị đáp ứng tần số
    plt.plot(np.abs(H))
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Đáp ứng tần số của bộ lọc equalizer')
    plt.grid()
    plt.show()
    return hd

def signal_lpf(signal, N, f_c, fs, type_of_window):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    # plt.plot(indices, signal)
    # plt.title('Đồ thị thời gian của tín hiệu trước khi lọc')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()
    hd = lpf(N, f_c, fs, type_of_window)
    y = scipy.signal.convolve(signal, hd, mode = 'same')
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

    return y

def signal_hpf(signal, N, f_c, fs, type_of_window):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    # plt.plot(indices, signal)
    # plt.title('Đồ thị thời gian của tín hiệu trước khi lọc')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()
    hd = hpf(N, f_c, fs, type_of_window)
    y = scipy.signal.convolve(signal, hd, mode = 'same')
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

    # ax2.plot(freqs, np.abs(X) / N, label='Trước lọc')
    # ax2.plot(freqs, np.abs(Y) / N, label='Sau lọc')
    ax2.plot(freqs[:N // 2], np.abs(X[:N // 2]) / N, label='Trước lọc')
    ax2.plot(freqs[:N // 2], np.abs(Y[:N // 2]) / N, label='Sau lọc')
    ax2.set_xlabel('Tần số (Hz)')
    ax2.set_ylabel('Biên độ')
    ax2.legend()

    plt.show()

    # Tạo một tín hiệu ngẫu nhiên và tính toán biên độ phổ của tín hiệu
    signal_fft = np.abs(np.fft.fft(signal))

    # Áp dụng bộ lọc thông thấp vào tín hiệu và tính toán biên độ phổ của tín hiệu đã lọc
    filtered_signal = np.convolve(signal, hd, mode='same')
    filtered_signal_fft = np.abs(np.fft.fft(filtered_signal))

    # Vẽ đồ hình biên độ phổ của tín hiệu trước và sau khi qua bộ lọc thông thấp
    freq_axis = np.linspace(0, 0.5, int(N / 2) + 1)  # trục tần số (tính bằng đơn vị tần số lấy mẫu)
    plt.plot(freq_axis, signal_fft[:len(freq_axis)], label='Tín hiệu gốc')
    plt.plot(freq_axis, filtered_signal_fft[:len(freq_axis)], label='Tín hiệu đã lọc')
    plt.xlabel('Tần số (tính bằng đơn vị tần số lấy mẫu)')
    plt.ylabel('Biên độ phổ')
    plt.title('Đồ hình biên độ phổ của tín hiệu trước và sau khi qua bộ lọc thông thấp')
    plt.legend()
    plt.show()
    return y

def signal_bpf(signal, N, f_c1, f_c2, fs, type_of_window):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    # plt.plot(indices, signal)
    # plt.title('Đồ thị thời gian của tín hiệu trước khi lọc')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()
    hd = bpf(N, f_c1, f_c2, fs, type_of_window)
    y = scipy.signal.convolve(signal, hd, mode = 'same')
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

    # ax2.plot(freqs, np.abs(X) / N, label='Trước lọc')
    # ax2.plot(freqs, np.abs(Y) / N, label='Sau lọc')
    ax2.plot(freqs[:N // 2], np.abs(X[:N // 2]) / N, label='Trước lọc')
    ax2.plot(freqs[:N // 2], np.abs(Y[:N // 2]) / N, label='Sau lọc')
    ax2.set_xlabel('Tần số (Hz)')
    ax2.set_ylabel('Biên độ')
    ax2.legend()

    plt.show()
    return y

def signal_bsf(signal, N, f_c1, f_c2, fs, type_of_window):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    # plt.plot(indices, signal)
    # plt.title('Đồ thị thời gian của tín hiệu trước khi lọc')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()
    hd = bsf(N, f_c1, f_c2, fs, type_of_window)
    y = scipy.signal.convolve(signal, hd, mode = 'same')
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

    # ax2.plot(freqs, np.abs(X) / N, label='Trước lọc')
    # ax2.plot(freqs, np.abs(Y) / N, label='Sau lọc')
    ax2.plot(freqs[:N // 2], np.abs(X[:N // 2]) / N, label='Trước lọc')
    ax2.plot(freqs[:N // 2], np.abs(Y[:N // 2]) / N, label='Sau lọc')
    ax2.set_xlabel('Tần số (Hz)')
    ax2.set_ylabel('Biên độ')
    ax2.legend()

    plt.show()
    return y

def signal_eq(signal, N, f_c, fs, type_of_window, num_filters):
    indices = np.linspace(0, len(signal) - 1, len(signal))
    # plt.plot(indices, signal)
    # plt.title('Đồ thị thời gian của tín hiệu trước khi lọc')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.show()
    hd = eq(N, f_c, fs, type_of_window, num_filters)
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

    # ax2.plot(freqs, np.abs(X) / N, label='Trước lọc')
    # ax2.plot(freqs, np.abs(Y) / N, label='Sau lọc')
    ax2.plot(freqs[:N // 2], np.abs(X[:N // 2]) / N, label='Trước lọc')
    ax2.plot(freqs[:N // 2], np.abs(Y[:N // 2]) / N, label='Sau lọc')
    ax2.set_xlabel('Tần số (Hz)')
    ax2.set_ylabel('Biên độ')
    ax2.legend()

    plt.show()
    return y

if __name__ == '__main__':
    # window = rectangular_window(61)
    # barlett_window(41)
    # N = 61
    # window = hanning_window(41)
    # hamming_window(31)
    # window = blackman_window(N)
    # type_filter = highpass_filter(4000, N, 22050)
    # lpf(N, 4000, 22050, 'Rectangular Window')
    # signal = np.random.randint(low=-10000, high=10000, size=101)
    signal, samperate = sf.read('BabyElephantWalk60.wav', frames = 44100 * 5)
    N = 151
    f_c = 1500
    # f_c1 = 4000
    # f_c2 = 6500
    fs = samperate
    num_filters = 15
    type_of_window = 'Hamming Window'
    # signal_eq(signal, N, f_c, fs, type_of_window, num_filters)
    y = signal_hpf(signal, N, f_c, fs, type_of_window)
    # eq(8000, 101, 44100, 'Hamming Window', 5)
    sf.write('filter.wav', y, samperate)