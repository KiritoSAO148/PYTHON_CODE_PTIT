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

def rectangular_window(N):
    w = np.ones(N)
    plt.stem(w)
    plt.title('Rectangular Window')
    plt.xlabel('n')
    plt.ylabel('w(n)')
    plt.show()
    return w

def barlett_window(N):
    a = []
    for i in range(N):
        if i >= 0 and i <= (N - 1) / 2: a.append((2 * i) / (N - 1))
        elif i >= (N - 1) / 2 and i <= N - 1: a.append(2 - ((2 * i) / (N - 1)))
    w = np.array(a)
    # print(w)
    # w = scipy.signal.windows.bartlett(N)
    plt.stem(w)
    plt.title('Barlett Window')
    plt.xlabel('n')
    plt.ylabel('w(n)')
    plt.show()
    return w

def hanning_window(N):
    a = []
    for i in range(N): a.append(0.5 - 0.5 * cos((2 * pi * i) / (N - 1)))
    w = np.array(a)
    # w = scipy.signal.windows.hann(N)
    plt.stem(w)
    plt.title('Hanning Window')
    plt.xlabel('n')
    plt.ylabel('w(n)')
    plt.show()
    return w

def hamming_window(N):
    a = []
    for i in range(N): a.append(0.54 - 0.46 * cos((2 * pi * i) / (N - 1)))
    w = np.array(a)
    # w = scipy.signal.windows.hamming(N)
    plt.stem(w)
    plt.title('Hamming Window')
    plt.xlabel('n')
    plt.ylabel('w(n)')
    plt.show()
    return w

def blackman_window(N):
    a = []
    for i in range(N): a.append(0.42 - 0.5 * cos((2 * pi * i) / (N - 1)) + 0.08 * cos((4 * pi * i) / (N - 1)))
    w = np.array(a)
    # w = scipy.signal.windows.blackman(N)
    plt.stem(w)
    plt.title('Blackman Window')
    plt.xlabel('n')
    plt.ylabel('w(n)')
    plt.show()
    return w

def kaiser(N, beta):
    w = scipy.signal.windows.kaiser(N, beta)
    plt.stem(w)
    plt.title('Kaiser Window')
    plt.xlabel('n')
    plt.ylabel('w(n)')
    plt.show()
    return w

def lowpass_filter(f_c, N, fs):
    f_c_norm = f_c / fs
    w_c_norm = 2 * pi * f_c_norm
    M = (N - 1) // 2
    hd = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: hd[n] = w_c_norm / pi
        else: hd[n] = sin(w_c_norm * (n - M)) / (pi * (n - M))
    plt.stem(hd)
    plt.xlabel('n')
    plt.ylabel('hd[n]')
    plt.title('Đáp ứng xung của bộ lọc thông thấp')
    plt.grid()
    plt.show()
    return hd

def highpass_filter(f_c, N, fs):
    f_c_norm = f_c / fs
    w_c_norm = 2 * pi * f_c_norm
    M = (N - 1) // 2
    hd = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: hd[n] = 1 - w_c_norm / pi
        else: hd[n] = -sin(w_c_norm * (n - M)) / (pi * (n - M))
    plt.stem(hd)
    plt.xlabel('n')
    plt.ylabel('hd[n]')
    plt.title('Đáp ứng xung của bộ lọc thông cao')
    plt.grid()
    plt.show()
    return hd

def bandpass_filter(f_c1, f_c2, N, fs):
    # N: số mẫu trong một chu kỳ xung
    f_c1_norm = f_c1 / fs
    f_c2_norm = f_c2 / fs
    w_c1_norm = 2 * pi * f_c1_norm
    w_c2_norm = 2 * pi * f_c2_norm
    M = (N - 1) // 2
    hd = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: hd[n] = w_c2_norm / pi - w_c1_norm / pi
        else: hd[n] = sin(w_c2_norm * (n - M)) / (pi * (n - M)) - sin(w_c1_norm * (n - M)) / (pi * (n - M))
    plt.stem(hd)
    plt.xlabel('n')
    plt.ylabel('hd[n]')
    plt.title('Đáp ứng xung của bộ lọc chắn dải')
    plt.grid()
    plt.show()
    return hd

def bandstop_filter(f_c1, f_c2, N, fs):
    f_c1_norm = f_c1 / fs
    f_c2_norm = f_c2 / fs
    w_c1_norm = 2 * pi * f_c1_norm
    w_c2_norm = 2 * pi * f_c2_norm
    M = (N - 1) // 2
    hd = np.zeros(N)
    for n in range(N):
        if n == (N - 1) // 2: hd[n] = 1 - w_c2_norm / pi + w_c1_norm / pi
        else: hd[n] = -sin(w_c2_norm * (n - M)) / (pi * (n - M)) + sin(w_c1_norm * (n - M)) / (pi * (n - M))
    plt.stem(hd)
    plt.xlabel('n')
    plt.ylabel('hd[n]')
    plt.title('Đáp ứng xung của bộ lọc chắn dải')
    plt.grid()
    plt.show()
    return hd

def filter(window, type_filter):
    h = window * type_filter
    plt.stem(h)
    plt.xlabel('n')
    plt.ylabel('h[n]')
    plt.title('Đáp ứng xung của bộ lọc')
    plt.grid()
    plt.show()
    return h

if __name__ == '__main__':
    # window = rectangular_window(101)
    # barlett_window(41)
    N = 61
    # window = hanning_window(41)
    # hamming_window(31)
    window = blackman_window(N)
    type_filter = highpass_filter(4000, N, 22050)
    h = filter(window, type_filter)
    # Tính đáp ứng tần số của bộ lọc thông thấp
    H = np.fft.fft(h)
    H = np.fft.fftshift(H)
    f = np.linspace(-0.5, 0.5, N)

    # Vẽ đồ thị đáp ứng tần số
    plt.plot(f, np.abs(H))
    plt.xlabel('f')
    plt.ylabel('|H(f)|')
    plt.title('Đáp ứng tần số của bộ lọc thông thấp')
    plt.grid()
    plt.show()