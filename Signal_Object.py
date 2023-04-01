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
import Main

class Signal:
    def __init__(self, amplitude, startPos, fs, info, length):
        self.__amplitude = amplitude
        self.__startPos = startPos
        self.__fs = fs
        self.__info = info
        self.__length = length

    def get_startPos(self):
        return self.__startPos

    def set_startPos(self, startPos):
        self.__startPos = startPos

    def get_amplitude(self):
        return self.__amplitude

    def set_amplitude(self, amplitude):
        self.__amplitude = amplitude

    def get_length(self):
        return self.__length

    def set_length(self, length):
        self.__length = length

    def get_Info(self):
        return self.__info

    def set_info(self, info):
        self.__info = info

    def get_fs(self):
        return self.__fs

    def set_fs(self, fs):
        self.__fs = fs

    def __str__(self):
        return f'{self.__amplitude} {self.__startPos} {self.__fs} {self.__info} {self.__length}'

    def getRandSignal(self, length, amplitude, startPos = 0):
        self.__length = length
        self.__amplitude = amplitude
        self.__startPos = startPos

    def plotSignal(self):
        indices = np.linspace(0, self.__length - 1, self.__length)
        indices += self.__startPos
        # print(indices)
        # print(self.__amplitude)
        # print(self.__startPos)
        plt.stem(indices, self.__amplitude)
        plt.title('Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def plotResSignal(self, other, signal):
        # self.chuanHoa()
        # other.chuanHoa()
        # self.chuanHoaArr(signal)
        # self.chuanHoaPos(signal)
        # other.chuanHoaArr(signal)
        # other.chuanHoaPos(signal)
        indices1 = np.linspace(0, self.__length - 1, self.__length)
        indices1 += self.__startPos
        indices2 = np.linspace(0, other.__length - 1, other.__length)
        indices2 += other.__startPos
        indices = np.linspace(0, signal.__length - 1, signal.__length)
        indices += signal.__startPos
        # print(indices1)
        # print(indices2)
        # print(indices)
        # print(self.__startPos)
        # print(other.__startPos)
        # print(signal.__startPos)
        # print(self.__amplitude)
        # print(other.__amplitude)
        # print(signal.__amplitude)
        plt.plot(indices, self.__amplitude, label = 'Signal 1')
        plt.plot(indices, other.__amplitude, label = 'Signal 2')
        plt.plot(indices, signal.__amplitude, label = 'Signal Result')
        plt.title('Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()

    def chuanHoaArr(self, other):
        am1, am2, pos1, pos2 = self.__amplitude, other.__amplitude, self.__startPos, other.__startPos
        if len(am1) == len(am2): return
        n = max(len(am1), len(am2))
        if len(am1) < len(am2):
            if pos1 > pos2:
                for _ in range(n - len(am1)): am1 = np.insert(am1, 0, 0)
                self.__startPos = pos2
            elif pos1 < pos2:
                for _ in range(n - len(am1)): am1 = np.insert(am1, len(am1), 0)
        elif len(am1) > len(am2):
            if pos1 < pos2:
                for _ in range(n - len(am2)): am2 = np.insert(am2, 0, 0)
                other.__startPos = pos1
            elif pos1 > pos2:
                for _ in range(n - len(am2)): am2 = np.insert(am2, len(am2), 0)
        self.__amplitude, other.__amplitude = am1, am2

    def chuanHoaPos(self, other):
        if self.__startPos > other.__startPos:
            left_over = self.__startPos - other.__startPos
            for _ in range(left_over):
                self.__amplitude = np.insert(self.__amplitude, 0, 0)
                other.__amplitude = np.insert(other.__amplitude, len(other.__amplitude), 0)
        elif self.__startPos < other.__startPos:
            left_over = other.__startPos - self.__startPos
            for _ in range(left_over):
                other.__amplitude = np.insert(other.__amplitude, 0, 0)
                self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)

    def chuanHoa(self):
        if self.__startPos > 0:
            for _ in range(self.__startPos): self.__amplitude = np.insert(self.__amplitude, 0, 0)
            self.__startPos = 0
        elif self.__startPos < -len(self.__amplitude):
            for _ in range(abs(self.__startPos) - len(self.__amplitude) + 1):
                self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
            # self.__startPos = -len(self.__amplitude)
        self.__length = len(self.__amplitude)
        # print(self.__amplitude)
        # print(self.__startPos)

    def __add__(self, other):
        self.chuanHoaArr(other)
        self.chuanHoaPos(other)
        pos = min(self.__startPos, other.__startPos)
        ampli = self.__amplitude + other.__amplitude
        S = Signal(ampli, pos, 44100, 'Sample Signal Object 3', len(ampli))
        S.chuanHoa()
        return S

    def __sub__(self, other):
        self.chuanHoaArr(other)
        self.chuanHoaPos(other)
        pos = min(self.__startPos, other.__startPos)
        ampli = self.__amplitude - other.__amplitude
        S = Signal(ampli, pos, 44100, 'Sample Signal Object 3', len(ampli))
        S.chuanHoa()
        return S

    def __mul__(self, other):
        self.chuanHoaArr(other)
        self.chuanHoaPos(other)
        pos = min(self.__startPos, other.__startPos)
        ampli = self.__amplitude * other.__amplitude
        S = Signal(ampli, pos, 44100, 'Sample Signal Object 3', len(ampli))
        S.chuanHoa()
        return S

    def mul_const(self, const):
        self.chuanHoa()
        for i in range(self.__length): self.__amplitude[i] *= const

    def inverse(self):
        self.chuanHoa()
        res = np.array([0])
        if self.__startPos < 0:
            a = self.__amplitude[:abs(self.__startPos) + 1]
            # print(a)
            b = self.__amplitude[abs(self.__startPos) + 1:]
            # print(b)
            res = np.concatenate((b[::-1], a[::-1]))
            self.__startPos = -len(b)
        else:
            a = self.__amplitude[self.__startPos:]
            # print(a)
            res = a[::-1]
            self.__startPos = -len(a) + 1
        self.__amplitude = np.array(res)
        # print(res)

    def delay(self, k):
        self.__startPos += abs(k)
        print(self.__startPos)
        ampli = self.__amplitude
        if self.__startPos > 0:
            for _ in range(self.__startPos): self.__amplitude = np.insert(self.__amplitude, 0, 0)
        elif self.__startPos < 0:
            for _ in range(abs(self.__startPos) - len(self.__amplitude)):
                self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
        # self.__amplitude = ampli
        self.getRandSignal(len(self.__amplitude), self.__amplitude, min(0, self.__startPos))
        print(len(self.__amplitude))
        print(self.__amplitude)
        # self.plotSignal()

    def early(self, k):
        self.__startPos -= abs(k)
        print(self.__startPos)
        ampli = self.__amplitude
        if abs(self.__startPos) > len(self.__amplitude):
            for _ in range(abs(self.__startPos) - len(self.__amplitude)):
                self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
        # self.__amplitude = ampli
        elif self.__startPos > 0:
            for _ in range(self.__startPos): self.__amplitude = np.insert(self.__amplitude, 0, 0)
        self.getRandSignal(len(self.__amplitude), self.__amplitude, min(0, self.__startPos))
        print(len(self.__amplitude))
        print(self.__amplitude)
        # self.plotSignal()

    def energy(self):
        return np.sum(self.__amplitude ** 2)

    def power(self):
        N = sympy.Symbol('N')
        En = self.energy()
        expr = (1 // (2 * N + 1)) * En
        return sympy.limit(expr, N, sympy.oo)

    def convolve1(self, other):
        res = [0] * (self.__length + other.__length - 1)
        for i in range(len(res)):
            for j in range(other.__length):
                if i - j < 0 or i - j >= self.__length: continue
                res[i] += self.__amplitude[i - j] * other.__amplitude[j]
        ans = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
        pos = 0
        if self.__startPos * other.__startPos > 0: pos = self.__startPos + other.__startPos
        else: pos = min(self.__startPos, other.__startPos)
        ans.getRandSignal(len(res), np.array(res), pos)
        return ans

    def convolve2(self, other):
        arr = numpy.convolve(self.__amplitude, other.__amplitude)
        ans = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
        pos = 0
        if self.__startPos * other.__startPos > 0:
            pos = self.__startPos + other.__startPos
        else:
            pos = min(self.__startPos, other.__startPos)
        ans.getRandSignal(len(arr), arr, pos)
        return ans

    def dft1(self):
        x, N = self.__amplitude, self.__length
        res = np.zeros(N, dtype = complex)
        for k in range(N):
            for n in range(N):
                res[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
        return res

    def dft2(self):
        return np.fft.fft(self.__amplitude)

    def idft1(self, x):
        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(2j * np.pi * k * n / N)
        x = np.dot(M, x) / N
        return x

    def idft2(self):
        X = self.dft1()
        return np.fft.ifft(X)

    def showdft1(self):
        X = self.dft1(); x = self.__amplitude; N = len(x)
        print(x)
        print(X)
        freq = np.arange(N) * (self.__fs / N)
        print(freq)
        plt.stem(freq, np.abs(X))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()

    def showdft2(self):
        X = self.dft2(); x = self.__amplitude; N = len(x)
        print(x)
        print(X)
        freq = np.arange(N) * (self.__fs / N)
        print(freq)
        plt.stem(freq, np.abs(X))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()

    def fft(self, x):
        n = len(x)
        if n == 1:
            return x
        even = self.fft(x[0::2])
        odd = self.fft(x[1::2])
        T = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
        return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

    def ifft(self, x):
        n = len(x)
        if n == 1:
            return x
        even = self.fft(x[0::2])
        odd = self.fft(x[1::2])
        T = [cmath.exp(2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
        return [even[k] + T[k] for k in range(n // 2)] + [even[k] - T[k] for k in range(n // 2)]

    def show_ifft(self, x):
        X = np.array(self.ifft(x))
        N = len(X)
        X *= 1 / N
        return X.real

    def dct(self):
        return scipy.fftpack.dct(self.__amplitude, type = 2)

    def idct(self):
        x = self.dct()
        return scipy.fftpack.idct(x, type = 2) / (2 * len(x))

    def cross_correlation2(self, other):
        other.inverse()
        return self.convolve1(other)

    def cross_correlation1(self, other):
        return np.correlate(self.__amplitude, other.__amplitude, mode = 'full')

    def auto_correlation(self):
        autocorr = np.correlate(self.__amplitude, self.__amplitude, mode = 'full')
        return autocorr[self.__length - 1:]

def operation():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    signal2 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 2', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    b = np.array([3, -1, 1, 0, 6])
    pos1 = -3; pos2 = 2
    signal1.getRandSignal(len(a), a, pos1)
    signal2.getRandSignal(len(b), b, pos2)
    print(signal1.get_amplitude())
    print(signal2.get_amplitude())
    # signal1.plotSignal()
    # signal2.plotSignal()
    signal = signal1 + signal2
    # signal.plotSignal()
    signal1.plotResSignal(signal2, signal)

def mul_const_operation(const):
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = 3
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    signal1.mul_const(const)
    signal1.plotSignal()

def inverse():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = 3
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    signal1.inverse()
    signal1.plotSignal()

def delay(k):
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    signal1.delay(k)
    signal1.plotSignal()

def early(k):
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    signal1.early(k)
    signal1.plotSignal()

def energy_power():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.chuanHoa()
    signal1.plotSignal()
    print(signal1.energy())
    print(signal1.power())

def convolve():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    signal2 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 2', 1)
    # a = np.array([-1, 2, -5, 0, -4, 4, -4])
    # b = np.array([3, -1, 1, 0, 6, 5])
    a = np.random.randint(low = -10000, high = 10000, size = 50001, dtype = int)
    b = np.random.randint(low=-1000, high=1000, size=1001, dtype=int)
    pos1 = 3; pos2 = -1
    signal1.getRandSignal(len(a), a, pos1)
    signal2.getRandSignal(len(b), b, pos2)
    signal1.chuanHoa()
    signal2.chuanHoa()
    start1 = timeit.default_timer()
    signal12 = signal1.convolve1(signal2)
    # signal12.plotSignal()
    print(signal12.get_amplitude())
    print(signal12.get_startPos())
    # signal12.plotSignal()
    stop1 = timeit.default_timer()
    start2 = timeit.default_timer()
    signal21 = signal1.convolve2(signal2)
    print(signal21.get_amplitude())
    print(signal21.get_startPos())
    # signal12.plotSignal()
    stop2 = timeit.default_timer()
    print(stop1 - start1)
    print(stop2 - start2)

def dft():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    # a = np.array([-1, 2, -5, 0, -4, 4, -4])
    a = np.random.randint(low=-10000, high=10000, size=5001, dtype=int)
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    start1 = timeit.default_timer()
    signal1.showdft1()
    stop1 = timeit.default_timer()
    start2 = timeit.default_timer()
    signal1.showdft2()
    stop2 = timeit.default_timer()
    print(stop1 - start1)
    print(stop2 - start2)

def idft():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    # a = np.random.randint(low=-10000, high=10000, size=5001, dtype=int)
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    # signal1.plotSignal()
    start1 = timeit.default_timer()
    x1 = signal1.dft1()
    X1 = signal1.idft1(x1)
    print(X1)
    stop1 = timeit.default_timer()
    start2 = timeit.default_timer()
    x2 = signal1.dft2()
    X2 = signal1.idft2()
    print(X2)
    stop2 = timeit.default_timer()
    print(stop1 - start1)
    print(stop2 - start2)

def fft():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    # a = np.array([-1, 2, -5, 0, -4, 4, -4, 7])
    a = np.random.randint(low=-10000, high=10000, size=1048576, dtype=int)
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    # signal1.plotSignal()
    start1 = timeit.default_timer()
    print(signal1.fft(a))
    stop1 = timeit.default_timer()
    start2 = timeit.default_timer()
    print(sp.fftpack.fft(signal1.get_amplitude()))
    stop2 = timeit.default_timer()
    print(stop1 - start1)
    print(stop2 - start2)
    # print(2**20)

def ifft():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    # a = np.array([-1, 2, -5, 0, -4, 4, -4, 7])
    a = np.random.randint(low=-10000, high=10000, size=1048576, dtype=int)
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    # signal1.plotSignal()
    start1 = timeit.default_timer()
    # print(signal1.fft(a))
    x = signal1.ifft(a)
    print(signal1.show_ifft(x))
    stop1 = timeit.default_timer()
    start2 = timeit.default_timer()
    print(sp.fftpack.ifft(x).real)
    stop2 = timeit.default_timer()
    print(stop1 - start1)
    print(stop2 - start2)

def dct():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    # a = np.array([-1, 2, -5, 0, -4, 4, -4])
    a = np.random.randint(low=-10000, high=10000, size=1048576, dtype=int)
    print(a)
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    # signal1.plotSignal()
    print(signal1.dct())
    print(signal1.idct())

def cross_correlation():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    signal2 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 2', 1)
    # a = np.array([-1, 2, -5, 0, -4, 4, -4])
    # b = np.array([3, -1, 1, 0, 6, 5])
    a = np.random.randint(low=-10000, high=10000, size=50001, dtype=int)
    b = np.random.randint(low=-1000, high=1000, size=1001, dtype=int)
    pos1 = 3; pos2 = -1
    signal1.getRandSignal(len(a), a, pos1)
    signal2.getRandSignal(len(b), b, pos2)
    # signal1.chuanHoa()
    # signal2.chuanHoa()
    start1 = timeit.default_timer()
    x = signal1.cross_correlation1(signal2)
    stop1 = timeit.default_timer()
    start2 = timeit.default_timer()
    y = signal1.cross_correlation2(signal2).get_amplitude()
    stop2 = timeit.default_timer()
    print(x)
    print(y)
    print(stop1 - start1)
    print(stop2 - start2)

def auto_correlation():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    # signal1.chuanHoa()
    # signal1.plotSignal()
    start = timeit.default_timer()
    print(signal1.auto_correlation())
    stop = timeit.default_timer()
    print(stop - start)

def example():
    data1, samplerate1 = sf.read('my-file-3.wav', frames = 44100 * 3, fill_value = 0)
    data2, samplerate2 = sf.read('my-file-4.wav', frames = 44100 * 3, fill_value = 0)
    signal1 = Signal(np.array([0]), 0, 44100 * 3, "Sample Signal Object", 0)
    signal2 = Signal(np.array([0]), 0, 44100 * 3, "Sample Signal Object", 0)
    res1 = np.array([x[0] for x in data1])
    res2 = np.array([x[0] for x in data2])
    # print(res1 + res2)
    # print(len(res1))
    # print(len(res2))
    signal1.getRandSignal(len(res1), res1, 0)
    signal2.getRandSignal(len(res2), res2, 0)
    # signal = signal1 + signal2
    # print(signal.get_amplitude())
    # signal1.plotResSignal(signal2, signal)
    start = timeit.default_timer()
    print(signal1.auto_correlation())
    stop = timeit.default_timer()
    print(stop - start)

if __name__ == '__main__':
    Main.main()