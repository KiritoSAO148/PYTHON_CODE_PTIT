from math import *
import io, os, sys, time
import array as arr
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fftpack
import soundfile as sf
import librosa as lb
import scipy as sp
import sympy
import cmath

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
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
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
            for _ in range(abs(self.__startPos) - len(self.__amplitude) + 1): self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
            # self.__startPos = -len(self.__amplitude)
        self.__length = len(self.__amplitude)
        # print(self.__amplitude)
        # print(self.__startPos)

    def __add__(self, other):
        self.chuanHoaArr(other)
        self.chuanHoaPos(other)
        return self.__amplitude + other.__amplitude

    def __sub__(self, other):
        self.chuanHoaArr(other)
        self.chuanHoaPos(other)
        return self.__amplitude - other.__amplitude

    def __mul__(self, other):
        self.chuanHoaArr(other)
        self.chuanHoaPos(other)
        return self.__amplitude * other.__amplitude

    def mul_const(self, const):
        for i in range(self.__length): self.__amplitude[i] *= const

    def inverse(self):
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
            for _ in range(abs(self.__startPos) - len(self.__amplitude)): self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
        # self.__amplitude = ampli
        self.getRandSignal(len(self.__amplitude), self.__amplitude, min(0, self.__startPos))
        print(len(self.__amplitude))
        print(self.__amplitude)
        self.plotSignal()

    def early(self, k):
        self.__startPos -= abs(k)
        print(self.__startPos)
        ampli = self.__amplitude
        if abs(self.__startPos) > len(self.__amplitude):
            for _ in range(abs(self.__startPos) - len(self.__amplitude)): self.__amplitude = np.insert(self.__amplitude, len(self.__amplitude), 0)
        # self.__amplitude = ampli
        elif self.__startPos > 0:
            for _ in range(self.__startPos): self.__amplitude = np.insert(self.__amplitude, 0, 0)
        self.getRandSignal(len(self.__amplitude), self.__amplitude, min(0, self.__startPos))
        print(len(self.__amplitude))
        print(self.__amplitude)
        self.plotSignal()

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
                if i - j >= 0 and i - j < self.__length: res[i] += self.__amplitude[i - j] * other.__amplitude[j]
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
    b = np.array([3, -1, 1, 0, 6, 5])
    pos1 = 3; pos2 = -1
    signal1.getRandSignal(len(a), a, pos1)
    signal2.getRandSignal(len(b), b, pos2)
    signal1.plotSignal()
    signal2.plotSignal()
    pos = min(pos1, pos2)
    signal = signal1 + signal2
    S = Signal(signal, pos, 44100, 'Sample Signal Object 3', len(signal))
    ampliS = S.get_amplitude()
    if S.get_startPos() > 0:
        for _ in range(S.get_startPos()): ampliS = np.insert(ampliS, 0, 0)
    S2 = Signal(ampliS, pos, 44100, 'Sample', len(ampliS))
    S2.set_startPos(min(pos, 0))
    print(S)
    print(type(S))
    S2.plotSignal()

def mul_const_operation(const):
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = 3
    signal1.getRandSignal(len(a), a, pos)
    signal1.chuanHoa()
    signal1.plotSignal()
    signal1.mul_const(const)
    signal1.plotSignal()

def inverse():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.chuanHoa()
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
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    b = np.array([3, -1, 1, 0, 6, 5])
    pos1 = 3; pos2 = -1
    signal1.getRandSignal(len(a), a, pos1)
    signal2.getRandSignal(len(b), b, pos2)
    signal1.chuanHoa()
    signal2.chuanHoa()
    signal12 = signal1.convolve1(signal2)
    # signal12 = signal1.convolve2(signal2)
    print(signal12.get_amplitude())
    print(signal12.get_startPos())
    signal12.plotSignal()

def dft():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    signal1.showdft1()
    # signal1.showdft2()

def idft():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    x = signal1.dft1()
    X = signal1.idft1(x)
    print(X)
    # x = signal1.dft2()
    # X = signal1.idft2()
    # print(X)

def fft():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4, 7])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    print(signal1.fft(a))

def ifft():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    print(signal1.fft(a))
    x = signal1.ifft(a)
    print(signal1.show_ifft(x))

def dct():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -2
    signal1.getRandSignal(len(a), a, pos)
    signal1.plotSignal()
    print(signal1.dct())
    print(signal1.idct())

def cross_correlation():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    signal2 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 2', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    b = np.array([3, -1, 1, 0, 6, 5])
    pos1 = 3; pos2 = -1
    signal1.getRandSignal(len(a), a, pos1)
    signal2.getRandSignal(len(b), b, pos2)
    # signal1.chuanHoa()
    # signal2.chuanHoa()
    x = signal1.cross_correlation1(signal2)
    y = signal1.cross_correlation2(signal2).get_amplitude()
    print(x)
    print(y)

def auto_correlation():
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    pos = -10
    signal1.getRandSignal(len(a), a, pos)
    # signal1.chuanHoa()
    signal1.plotSignal()
    print(signal1.auto_correlation())

if __name__ == '__main__':
    operation()

    # mul_const_operation(5)

    # inverse()

    # delay(3)

    # early(3)

    # energy_power()

    # convolve()

    # dft()

    # idft()

    # fft()

    # ifft()

    # dct()

    # cross_correlation()

    # auto_correlation()

    # signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    # signal2 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 2', 1)
    # a = np.array([-1, 2, -5, 0, -4, 4, -4])
    # # [-1, 2, -5, 0, -4, 4, -4, -3, 1, -3]
    # # [3, -1, 1, 0, 1, 2, 1, 1]
    # b = np.array([3, -1, 1, 0, 6, 5])
    #
    # pos1 = 3; pos2 = -1
    # pos = min(pos1, pos2)
    # signal1.getRandSignal(len(a), a, pos1)
    # signal2.getRandSignal(len(b), b, pos2)
    # signal1.plotSignal()
    # signal1.chuanHoa()
    # signal1.inverse()
    # signal1.plotSignal()
    # signal2.plotSignal()
    # signal = signal1 + signal2
    # S = Signal(signal, pos, 44100, 'Sample Signal Object 3', len(signal))
    # ampliS = S.get_amplitude()
    # if S.get_startPos() > 0:
    #     for _ in range(S.get_startPos()): ampliS = np.insert(ampliS, 0, 0)
    # S2 = Signal(ampliS, pos, 44100, 'Sample', len(ampliS))
    # S2.set_startPos(min(pos, 0))
    # print(S)
    # print(type(S))
    # S2.plotSignal()

    # signal3 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    # signal3.getRandSignal(len(a), a, pos1)
    # ampliS3 = signal3.get_amplitude()
    # if signal3.get_startPos() > 0:
    #     for _ in range(signal3.get_startPos() - 0): ampliS3 = np.insert(ampliS3, 0, 0)
    # S3 = Signal(ampliS3, pos1, 44100, 'Sample', len(ampliS3))
    # S3.set_startPos(min(pos, 0))
    # S3.plotSignal()
    # print(S3.get_amplitude())
    # print(S3.get_startPos())
    # S3.inverse()
    # print(S3.get_amplitude())
    # print(S3.get_startPos())
    # S3.plotSignal()

    # signal4 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 4', 1)
    # signal4.getRandSignal(len(a), a, pos1)
    # signal4.delay(3)

    # signal5 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 5', 1)
    # signal5.getRandSignal(len(a), a, pos1)
    # signal5.chuanHoa()
    # signal5.plotSignal()
    # print(signal5.get_amplitude())
    # print(signal5.energy())
    # print(signal5.power())


    # print(signal1.auto_correlation())
    #
    # print(signal1.cross_correlation1(signal2))
    # print(signal1.cross_correlation2(signal2).get_amplitude())
    # signal1.chuanHoa()
    # signal2.chuanHoa()
    # # signal1.chuanHoaArr(signal2)
    # print(signal1.get_amplitude())
    # print(signal2.get_amplitude())
    # x = signal1.dct()
    # print(x)
    # print(signal1.idct())
    #
    # signal11 = signal1.convolve1(signal2)
    # signal22 = signal1.convolve2(signal2)
    # print(signal11.get_amplitude())
    # print(signal11.get_startPos())
    # signal11.plotSignal()
    # print(signal11.dft1())
    # x = signal11.dft1()
    # print(x)
    # X = signal11.idft1(x)
    # print(X.real)
    # print(signal11.idft2().real)
    # print(signal11.get_amplitude())

    # print(signal11.fft([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    # print(signal11.dft2())
    # signal11.showdft()

    # print(signal1.fft([1, 2, 3, 0]))
    # x = signal1.ifft([1, 2, 3, 0])
    # print(signal1.show_ifft(x))

    # signal7 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    # signal7.getRandSignal(len(a), a, 3)
    # ampliS = signal7.get_amplitude()
    # if signal7.get_startPos() > 0:
    #     for _ in range(signal7.get_startPos()): ampliS = np.insert(ampliS, 0, 0)
    # S2 = Signal(ampliS, 0, 44100, 'Sample', len(ampliS))
    # S2.set_startPos(0)
    # S2.plotSignal()
    # S2.inverse()
    # S2.plotSignal()