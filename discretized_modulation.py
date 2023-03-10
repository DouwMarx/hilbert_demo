import plotly.io as pio
pio.renderers.default = "browser"
import numpy as np
import scipy.signal as signal
from plotly.subplots import make_subplots

# fs = 5000 # Sampling frequency
# fc = 100 # Carrier frequency
# fm = 10 # Modulation frequency
#
# t = np.arange(0, 1, 1/fs) # Time vector
#
# carrier_time = np.cos(2*np.pi*fc*t)
#
# # Create a modulating signal
# modulator_time = np.cos(2*np.pi*fm*t)
#
# # Modulate the carrier
# modulated_time = carrier_time * modulator_time # This is just element-wise multiplication
#
# # Get the hilbert transform of the modulated signal
# hilbert = signal.hilbert(modulated_time)
#
# # get the frequency representations
# carrier_freq = np.fft.fft(carrier_time)/len(carrier_time)
# modulator_freq = np.fft.fft(modulator_time)/len(modulator_time)
# modulated_freq = np.fft.fft(modulated_time)/len(modulated_time)
# fft_freq = np.fft.fftfreq(len(carrier_time), 1/fs)
#
# # Create a filter
# center_freq = 100
# bandwidth = 30
# b, a = signal.butter(4, [center_freq-bandwidth/2, center_freq+bandwidth/2], btype="bandpass", fs=fs)
#
# # Filter the modulated signal
# filtered_time = signal.filtfilt(b, a, modulated_time)
#
# # Get the frequency representation of the filtered signal
# filtered_freq = np.fft.fft(filtered_time)/len(filtered_time)

# Plot the time domain signals using plotly
import plotly.graph_objects as go

class HilbertDemo(object):
    def __init__(self,fs=1000,t_duration=1):
        self.fs = fs
        self.t_duration = t_duration
        self.t = np.arange(0, self.t_duration, 1/self.fs)

        self.parameter_grid = {"f_carrier": {"min": 50, "max": 100, "step": 0.1, "id": "f_carrier", "value": 70},
                                 "f_modulator": {"min": 5, "max": 10, "step": 0.1, "id": "f_modulator", "value": 5}}

    def get_carrier(self, f_carrier):
        return np.cos(2*np.pi*f_carrier*self.t)

    def get_modulator(self, f_modulator): # TODO: Make for arbitrary bandwith
        return np.cos(2*np.pi*f_modulator*self.t)

    def get_modulated(self, f_carrier, f_modulator):
        return self.get_carrier(f_carrier) * self.get_modulator(f_modulator)

    def get_time_domain_trace(self, signal, name):
        return go.Scatter(x=self.t, y=signal, name=name)

    def get_frequency_domain_magnitude_trace(self, signal, name):
        spectrum = np.fft.rfft(signal)/len(signal)
        freq = np.fft.rfftfreq(len(signal), 1/self.fs)
        return go.Scatter(x=freq, y=np.abs(spectrum), name=name) # Only show up to half of Nyquist frequency

    def make_time_domain_signal_components_plot_at_state(self, f_carrier, f_modulator):
        fig = go.Figure()
        fig.add_trace(self.get_time_domain_trace(self.get_carrier(f_carrier), "Carrier"))
        fig.add_trace(self.get_time_domain_trace(self.get_modulator(f_modulator), "Modulator"))
        fig.add_trace(self.get_time_domain_trace(self.get_modulated(f_carrier, f_modulator), "Modulated"))

        # Add title
        fig.update_layout(title="Time domain signals")

        # Add the axis labels
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")
        return fig

    def make_frequency_domain_signal_components_plot_at_state(self, f_carrier, f_modulator):
        fig = go.Figure()
        fig.add_trace(self.get_frequency_domain_magnitude_trace(self.get_carrier(f_carrier), "Carrier"))
        fig.add_trace(self.get_frequency_domain_magnitude_trace(self.get_modulator(f_modulator), "Modulator"))
        fig.add_trace(self.get_frequency_domain_magnitude_trace(self.get_modulated(f_carrier, f_modulator), "Modulated"))

        # Add title
        fig.update_layout(title="Frequency domain signals")

        # Add the axis labels
        fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        return fig

# hd = HilbertDemo()
# hd.make_signal_components_plot_at_state(50, 5).show()

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=t, y=carrier_time, name="Carrier"))
# fig.add_trace(go.Scatter(x=t, y=modulator_time, name="Modulator"))
# fig.add_trace(go.Scatter(x=t, y=modulated_time, name="Modulated"))
# fig.show()
#
# # Plot the amplitude and phase of the frequency domain signals as subplots side by side
# freq_fig = make_subplots(rows=2, cols=1, subplot_titles=("Magnitude", "Phase"))
#
# freq_fig.add_trace(go.Scatter(x=fft_freq, y=np.abs(carrier_freq), name="Carrier"), row=1, col=1)
# freq_fig.add_trace(go.Scatter(x=fft_freq, y=np.abs(modulator_freq), name="Modulator"), row=1, col=1)
# freq_fig.add_trace(go.Scatter(x=fft_freq, y=np.abs(modulated_freq), name="Modulated"), row=1, col=1)
#
# freq_fig.add_trace(go.Scatter(x=fft_freq, y=np.angle(carrier_freq), name="Carrier"), row=2, col=1)
# freq_fig.add_trace(go.Scatter(x=fft_freq, y=np.angle(modulator_freq), name="Modulator"), row=2, col=1)
# freq_fig.add_trace(go.Scatter(x=fft_freq, y=np.angle(modulated_freq), name="Modulated"), row=2, col=1)
# freq_fig.show()
#
#
# # Plot the frequency representation of the amplitude of the modulating signal and the filtered signal
# filter_fig = go.Figure()
# filter_fig.add_trace(go.Scatter(x=fft_freq, y=np.abs(modulator_freq), name="Modulator"))
# filter_fig.add_trace(go.Scatter(x=fft_freq, y=np.abs(filtered_freq), name="Filtered"))
# filter_fig.show()
#
#











