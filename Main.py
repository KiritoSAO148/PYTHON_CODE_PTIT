# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import freqz
#
# def fir_eq_design(fc, order, window, num_taps):
#     # Tính toán thông số của bộ lọc
#     fs = 2*fc
#     wc = 2*np.pi*fc/fs
#     b = np.zeros(num_taps)
#     for n in range(num_taps):
#         if n == (num_taps - 1) // 2:
#             b[n] = 2*fc/fs
#         else:
#             b[n] = np.sin(wc*(n - (num_taps - 1) // 2))/(np.pi*(n - (num_taps - 1) // 2))
#     # Áp dụng hàm cửa sổ
#     if window == 'hamming':
#         w = np.hamming(num_taps)
#     elif window == 'hanning':
#         w = np.hanning(num_taps)
#     elif window == 'blackman':
#         w = np.blackman(num_taps)
#     else:
#         w = np.ones(num_taps)
#     b *= w
#     # Tính toán đáp ứng tần số
#     w, h = freqz(b, worN=1024)
#     w *= fs/(2*np.pi)
#     h_db = 20*np.log10(np.abs(h))
#     # Vẽ đáp ứng tần số
#     plt.plot(w, h_db)
#     plt.title('FIR Equalizer Frequency Response')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude (dB)')
#     plt.grid(True)
#     plt.show()
#     # Tính toán và hiển thị đáp ứng xung
#     impulse_response = np.zeros(num_taps)
#     impulse_response[(num_taps - 1) // 2] = 1
#     h_ir = np.convolve(b, impulse_response)
#     t = np.arange(0, len(h_ir))/fs
#     plt.plot(t, h_ir)
#     plt.title('FIR Equalizer Impulse Response')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.grid(True)
#     plt.show()
#
# fir_eq_design(fc=1000, order=50, window='hamming', num_taps=101)

# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt
#
# # Tham số của bộ lọc
# fs = 1000.0  # Tần số lấy mẫu
# cutoff_freq = 100.0  # Tần số cắt
# filter_order = 50  # Bậc của bộ lọc
# num_filters = 2  # Số bộ lọc trong EQ
# window = 'hamming'  # Loại hàm cửa sổ
#
# # Tính toán hệ số của bộ lọc
# nyquist_freq = 0.5 * fs
# normalized_cutoff_freq = cutoff_freq / nyquist_freq
# taps = signal.firwin(filter_order, normalized_cutoff_freq, window=window)
#
# # Áp dụng bộ lọc cho tín hiệu đầu vào
# input_signal = np.random.randn(10000)
# filtered_signal = signal.lfilter(taps, 1, input_signal)
#
# # Trực quan hóa đáp ứng xung của bộ lọc
# plt.figure()
# for i in range(num_filters):
#     freq_response = signal.freqz(taps)
#     mag_response = np.abs(freq_response[1])
#     phase_response = np.unwrap(np.angle(freq_response[1]))
#     freqs = freq_response[0] / np.pi * nyquist_freq
#     plt.semilogx(freqs, 20 * np.log10(mag_response), label=f'Filter {i+1}')
#     taps *= -1  # Đảo chiều bộ lọc để tạo bộ lọc đối xứng
# plt.title('Frequency response of EQ filters')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude [dB]')
# plt.legend()
# plt.show()

# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt
#
# # Cac tham so cua bo loc EQ
# cutoff_freq = 1000 # tan so cat
# filter_order = 100 # bac cua bo loc
# window_type = "hann" # loai ham cua so
# num_filters = 5 # so bo loc trong EQ
#
# # Tinh toan he so bo loc FIR
# nyquist_freq = 44100 / 2
# cutoff_normalized = cutoff_freq / nyquist_freq
# filter_coeffs = signal.firwin(filter_order, cutoff_normalized)
#
# # Ap dung ham cua so
# window = signal.get_window(window_type, filter_order)
# filter_coeffs *= window
#
# # Tao bo loc EQ
# eq_filter_coeffs = np.tile(filter_coeffs, num_filters)
#
# # Tinh toan dap ung xung cua bo loc EQ
# freq, response = signal.freqz(eq_filter_coeffs)
# magnitude_response = np.abs(response)
# phase_response = np.unwrap(np.angle(response))
#
# # Ve do thi dap ung xung
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(freq, magnitude_response)
# plt.title('Magnitude Response')
# plt.xlabel('Frequency (radians/sample)')
# plt.ylabel('Magnitude (dB)')
#
# plt.subplot(2, 1, 2)
# plt.plot(freq, phase_response)
# plt.title('Phase Response')
# plt.xlabel('Frequency (radians/sample)')
# plt.ylabel('Phase (radians)')
#
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal as signal
#
# # Xác định thông số bộ lọc FIR EQ
# fs = 44100  # tần số lấy mẫu
# fc = 1000  # tần số cắt
# N = 100  # bậc của bộ lọc
# window = 'hanning'  # loại hàm cửa sổ
# num_filters = 3  # số bộ lọc trong EQ không sử dụng hàm có sẵn
#
# # Thiết kế bộ lọc FIR EQ
# h = np.zeros(N)
# for n in range(N):
#     h[n] = 2 * fc / fs * np.sinc(2 * fc / fs * (n - (N-1)/2))
#
# # Áp dụng hàm cửa sổ
# if window == 'hanning':
#     w = np.hanning(N)
# elif window == 'hamming':
#     w = np.hamming(N)
# elif window == 'blackman':
#     w = np.blackman(N)
#
# h = h * w
#
# # Tính toán đáp ứng xung của bộ lọc FIR EQ
# w, H = signal.freqz(h)
#
# # Vẽ đáp ứng xung
# fig, ax = plt.subplots()
# ax.plot(w, 20 * np.log10(abs(H)), 'b')
# ax.set_title('Đáp ứng xung của bộ lọc FIR EQ')
# ax.set_xlabel('Tần số (rad/sample)')
# ax.set_ylabel('Độ lớn (dB)')
# ax.set_xlim([0, np.pi])
# ax.grid(True)
# plt.show()

# import numpy as np
#
# # Tần số cắt
# fc = 1000
# # Tần số lấy mẫu
# fs = 8000
# # Bậc của bộ lọc
# N = 100
# # Loại hàm cửa sổ
# window = np.hanning(N)
# # Số bộ lọc trong EQ
# num_filters = 3
#
# # Thiết kế bộ lọc FIR sử dụng hàm sinc
# n = np.arange(N)
# h = 2*fc/fs * np.sinc(2*fc/fs*(n-(N-1)/2))
#
# # Áp dụng hàm cửa sổ
# h *= window
#
# # Tạo ra các bộ lọc còn lại trong EQ
# h_eq = h
# for i in range(num_filters-1):
#     h_eq = np.convolve(h_eq, h)
#
# # Tính toán đáp ứng xung của bộ lọc EQ
# H_eq = np.fft.fft(h_eq)
#
# # Vẽ đáp ứng xung của bộ lọc EQ
# import matplotlib.pyplot as plt
# plt.plot(np.abs(H_eq))
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Tạo tín hiệu sin
# fs = 1000 # tần số lấy mẫu
# f = 100 # tần số sin
# t = np.arange(0, 1, 1/fs) # thời gian
# x = np.sin(2*np.pi*f*t) # tín hiệu sin
#
# # Hiển thị đồ hình biên độ thời gian
# plt.plot(t, x)
# plt.xlabel('Thời gian (giây)')
# plt.ylabel('Biên độ')
# plt.title('Tín hiệu sin')
# plt.show()
# from scipy.signal import firwin
#
# # Thiết kế bộ lọc thông thấp FIR
# cutoff = 50 # tần số cắt
# numtaps = 101 # độ dài cửa sổ
# b = firwin(numtaps, cutoff, fs=fs, window='hamming')
# from scipy.signal import lfilter
#
# # Áp dụng bộ lọc thông thấp FIR lên tín hiệu sin
# y = lfilter(b, 1, x)
#
# from scipy.fft import fft, fftfreq
#
# # Hiển thị đồ hình biên độ thời gian trước và sau khi lọc
# fig, (ax1, ax2) = plt.subplots(2, 1)
# fig.suptitle('Kết quả lọc tín hiệu sin')
#
# ax1.plot(t, x, label='Trước lọc')
# ax1.plot(t, y, label='Sau lọc')
# ax1.set_xlabel('Thời gian (giây)')
# ax1.set_ylabel('Biên độ')
# ax1.legend()
#
# # Hiển thị đồ hình biên độ phổ trước và sau khi lọc
# N = len(x)
# X = fft(x)
# Y = fft(y)
# freqs = fftfreq(N, 1/fs)
#
# ax2.plot(freqs[:N//2], np.abs(X[:N//2])/N, label='Trước lọc')
# ax2.plot(freqs[:N//2], np.abs(Y[:N//2])/N, label='Sau lọc')
# ax2.set_xlabel('Tần số (Hz)')
# ax2.set_ylabel('Biên độ')
# ax2.legend()
#
# plt.show()

# import numpy as np
#
# # Tần số cắt
# fc = 6800
# # Tần số lấy mẫu
# fs = 44100
# # Bậc của bộ lọc
# N = 1001
# # Loại hàm cửa sổ
# window = np.bartlett(N)
# # Số bộ lọc trong EQ
# num_filters = 10
#
# # Thiết kế bộ lọc FIR sử dụng hàm sinc
# n = np.arange(N)
# h = 2*fc/fs * np.sinc(2*fc/fs*(n-(N-1)/2))
#
# # Áp dụng hàm cửa sổ
# h *= window
#
# # Tạo ra các bộ lọc còn lại trong EQ
# h_eq = h
# for i in range(num_filters-1):
#     h_eq = np.convolve(h_eq, h)
#
# # Tính toán đáp ứng xung của bộ lọc EQ
# H_eq = np.fft.fft(h_eq)
#
# # Vẽ đáp ứng xung của bộ lọc EQ
# import matplotlib.pyplot as plt
# plt.plot(np.abs(H_eq))
# plt.xlabel('Frequency')
# plt.ylabel('Magnitude')
# plt.show()

import librosa
import numpy as np
import soundfile as sf

# def add_echo(input_file, output_file, alpha, delay):
#     # Load input audio file
#     y, sr = sf.read(input_file)
#
#     # Create delayed version of input signal
#     y_delayed = np.zeros_like(y)
#     for i in range(len(y)):
#         delay_samples = int(delay[i] * sr)
#         if i < delay_samples:
#             y_delayed[i] = y[i]
#         else:
#             y_delayed[i] = y[i] + alpha[i] * y[i - delay_samples]
#
#     # Save output audio file
#     sf.write(output_file, y_delayed, sr)
#
# alphas = [0.2, 0.4, 0.6, 0.8]
# delays = [0.1, 0.2, 0.3, 0.4]
# add_echo('BabyElephantWalk60.wav', 'audio_1.wav', alphas, delays)

# import matplotlib.pyplot as plt
# import numpy as np
#
# # tạo dữ liệu mẫu
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# # tạo 1 figure với 2 subplot
# fig, (ax1, ax2) = plt.subplots(2)
#
# # trục đầu tiên
# ax1.plot(x, y1, label='sin(x)')
# ax1.set_title('Đồ thị hàm sin và cos')
# ax1.legend()
#
# # trục thứ hai
# ax2.plot(x, y2, label='cos(x)')
# ax2.set_xlabel('x')
# ax2.set_ylabel('y')
# ax2.legend()
#
# # hiển thị đồ thị
# plt.show()

# from scipy import signal
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the desired frequency response
# freq = [0, 0.2, 0.3, 0.5]  # Frequency points
# gain = [1, 1, 0, 0]  # Desired gains at each frequency point
#
# # Define the filter parameters
# numtaps = 64  # Number of filter coefficients
#
# # Design the filter
# taps = signal.firwin2(numtaps, freq, gain)
#
# # Plot the filter's frequency response
# w, h = signal.freqz(taps)
# f = w / (2 * np.pi)
# plt.plot(f, 20 * np.log10(abs(h)))
# plt.title('Filter Frequency Response')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain (dB)')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def hanning_window(N):
    """
    Returns the Hanning window of length N.
    """
    return 0.5 - 0.5 * np.cos(2 * np.pi / (N - 1) * np.arange(N))

def plot_hanning_frequency_responses(N):
    """
    Plots the frequency-domain responses of the Hanning window.
    """
    # Compute the frequency response
    w = np.linspace(-np.pi, np.pi, 1000)
    A = 0.5 * (np.sin(w * N / 2) / np.sin(w / 2)) + 0.5 / 2 * \
        (np.sin(w * np.pi / 2 - (N - 1) * np.pi / (2 * (N - 1))) / np.sin(w / 2 - np.pi / (2 * (N - 1))))
    AdB = 20 * np.log10(np.abs(A))

    # Compute the phase response
    P = np.unwrap(np.angle(A))

    # Compute the group delay
    eps = 1e-6
    gd = -np.diff(P) / np.diff(w)
    gd = np.concatenate((gd, [gd[-1]]))

    # Plot the amplitude response
    plt.subplot(3, 1, 1)
    plt.plot(w, AdB)
    plt.title('Hanning Window Frequency Responses')
    plt.ylabel('Amplitude (dB)')
    plt.grid(True)

    # Plot the phase response
    plt.subplot(3, 1, 2)
    plt.plot(w, P)
    plt.ylabel('Phase (radians)')
    plt.grid(True)

    # Plot the group delay
    plt.subplot(3, 1, 3)
    plt.plot(w, gd)
    plt.xlabel('Frequency (radians/sample)')
    plt.ylabel('Group Delay (samples)')
    plt.grid(True)

    plt.show()

# Example usage
plot_hanning_frequency_responses(32)


