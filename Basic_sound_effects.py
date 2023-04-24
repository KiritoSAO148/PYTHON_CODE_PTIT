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

def plot_signal(x, y):
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.title("Âm thanh gốc")
    plt.xlabel("Time (samples)")

    plt.subplot(2, 1, 2)
    plt.plot(y)
    plt.title("Âm thanh sau khi áp dụng hiệu ứng")
    plt.xlabel("Time (samples)")

    plt.show()

def echo(input_file, output_file, alphas, delays):
    x, samperate = sf.read(input_file)
    y_delayed = np.zeros_like(x)
    for i in range(len(delays)):
        delay_samples = int(delays[i] * samperate)
        y_delayed[delay_samples:] += alphas[i] * x[:-delay_samples]
    y = x + y_delayed
    plot_signal(x, y)
    sf.write(output_file, y, samperate)

def fade_in(input_file, output_file, alpha, duration):
    signal, sr = sf.read(input_file)
    n_samples = duration * sr
    r = np.linspace(0, 1, n_samples)
    print(n_samples)
    print(len(r))
    print(r)
    a = alpha * r
    # print(r)
    # print(alpha)
    print(a)
    y = signal * np.concatenate([a, np.ones(len(signal) - n_samples)])
    plot_signal(signal, y)
    sf.write(output_file, y, sr)

def fade_out(input_file, output_file, alpha, duration):
    signal, sr = sf.read(input_file)
    n_samples = duration * sr
    r = np.linspace(1, 0, n_samples)
    print(n_samples)
    print(len(r))
    # print(r)
    a = alpha * r
    print(len(a))
    # print(r)
    # print(alpha)
    # print(a)
    y = signal * np.concatenate([np.ones(len(signal) - n_samples), a])
    plot_signal(signal, y)
    sf.write(output_file, y, sr)

def amplification(input_file, output_file, gain):
    signal, sr = sf.read(input_file)
    y = signal * gain
    plot_signal(signal, y)
    sf.write(output_file, y, sr)

def chorus(input_file, output_file, delay, depth, feedback, mix):
    x, sr = sf.read(input_file)
    delay_samples = int(delay * sr)
    depth_samples = int(depth * sr)
    buffer = np.zeros_like(x)
    y = np.zeros_like(x)
    for i in range(len(x)):
        delay_index = i - delay_samples - int(np.sin(2 * np.pi * i * depth / sr) * depth_samples)
        delayed = buffer[delay_index] if delay_index >= 0 else 0
        buffer[i] = x[i] + feedback * delayed
        y[i] = (1 - mix) * x[i] + mix * buffer[i]
    plot_signal(x, y)
    sf.write(output_file, y, sr)

def flanger(input_file, output_file, delay_time, modulation_depth, modulation_rate):
    input_signal, sample_rate = sf.read(input_file)
    delay_samples = int(delay_time * sample_rate)
    output_signal = np.zeros_like(input_signal)
    delay_buffer = np.zeros(delay_samples)
    modulation = modulation_depth * np.sin(2 * np.pi * modulation_rate * np.arange(len(input_signal)) / sample_rate)
    for n in range(len(input_signal)):
        delayed_sample = delay_buffer[n % delay_samples]
        output_signal[n] = input_signal[n] + delayed_sample
        delayed_input = input_signal[n] + modulation[n]
        delay_buffer[n % delay_samples] = delayed_input
    plot_signal(input_signal, output_signal)
    sf.write(output_file, output_signal, sample_rate)
    return output_signal

def tremolo(input_file, output_file, rate, depth):
    x, sr = sf.read(input_file)
    t = np.linspace(0, len(x) / sr, len(x), endpoint=False)
    tremolo_wave = (1 + depth * np.sin(2 * np.pi * rate * t)) / 2
    y = x * tremolo_wave
    plot_signal(x, y)
    sf.write(output_file, y, sr)

def reversal(input_file, output_file):
    x, sr = sf.read(input_file)
    y = np.flip(x)
    plot_signal(x, y)
    sf.write(output_file, y, sr)
    return y

def vibrato(input_file, output_file, rate, depth, delay):
    y, sr = lb.load(input_file, sr=None, mono=True)
    delay_samples = int(delay * sr)
    lfo = depth * np.sin(2 * np.pi * rate * np.arange(len(y)) / sr)
    y_vibrato = np.zeros_like(y)
    for n in range(len(y)):
        delayed_sample = y[n - delay_samples] if n >= delay_samples else 0
        y_vibrato[n] = y[n] + delayed_sample * lfo[n]
    sf.write(output_file, y_vibrato, int(sr))
    plt.figure(figsize=(10, 4))
    plt.plot(y, label='Input')
    plt.plot(y_vibrato, label='Output')
    plt.legend()
    plt.title('Vibrato Effect')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show()
    return y_vibrato

if __name__ == '__main__':
    input_file = 'BabyElephantWalk60.wav'
    output_file = 'reversal.wav'
    reversal(input_file, output_file)
    # Dãy giá trị alpha và delay
    alphas = [0.2, 0.4, 0.45, 0.6]
    delays = [0.1, 0.15, 0.2, 0.3]  # đơn vị giây
    # echo(input_file, output_file, alphas, delays)
    # fade_in(input_file, output_file, 0.8, 10)
    # fade_out(input_file, output_file, 0.95, 10)
    # amplification(input_file, output_file, 2)
    # chorus(input_file, output_file, 0.5, 0.03, 0.5, 0.5)
    # delay_time = 0.06
    # modulation_depth = 0.01
    # modulation_rate = 0.25
    # flanger(input_file, output_file, delay_time, modulation_depth, modulation_rate)
    # vibrato(input_file, output_file, 10, 0.8, 0.3)
    depth = 0.5
    rate = 10
    # tremolo(input_file, output_file, rate, depth)