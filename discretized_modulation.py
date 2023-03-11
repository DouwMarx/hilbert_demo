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
    def __init__(self, fs=1000, t_duration=1):
        self.fs = fs
        self.t_duration = t_duration
        self.t = np.arange(0, self.t_duration, 1 / self.fs)
        self.freq_range = np.fft.rfftfreq(len(self.t), 1 / self.fs)

        self.parameter_grid = {"f_carrier": {"min": 50, "max": 100, "step": 0.1, "id": "f_carrier", "value": 70},
                               "f_modulator": {"min": 5, "max": 10, "step": 0.1, "id": "f_modulator", "value": 7},
                               "filter_center_freq_error": {"min": -10, "max": 10, "step": 0.1,
                                                            "id": "filter_center_freq_error", "value": 0},
                               "filter_bandwidth": {"min": 0, "max": 20, "step": 0.1, "id": "filter_bandwidth",
                                                    "value": 10}
                               }

        # Set the initial state
        self.update_state(self.parameter_grid["f_carrier"]["value"], self.parameter_grid["f_modulator"]["value"],
                          self.parameter_grid["filter_center_freq_error"]["value"],
                          self.parameter_grid["filter_bandwidth"]["value"])

    def update_state(self, f_carrier, f_modulator, filter_center_freq_error, filter_bandwidth):
        # Set the parameters
        self.f_carrier = f_carrier
        self.f_modulator = f_modulator
        self.filter_center_freq_error = filter_center_freq_error
        self.filter_bandwidth = filter_bandwidth

        # Set the time domain signals
        self.time_domain_carrier = self.get_carrier()
        self.time_domain_modulator = self.get_modulator()
        self.time_domain_modulated = self.get_modulated(self.time_domain_carrier, self.time_domain_modulator)

        # Set the frequency domain signals
        self.frequency_domain_carrier = np.fft.rfft(self.time_domain_carrier) / len(self.time_domain_carrier)
        self.frequency_domain_modulator = np.fft.rfft(self.time_domain_modulator) / len(self.time_domain_modulator)
        self.frequency_domain_modulated = np.fft.rfft(self.time_domain_modulated) / len(self.time_domain_modulated)

        # Define the filter
        self.filter_coefficients = self.make_filter(self.f_carrier + self.filter_center_freq_error,
                                                    self.filter_bandwidth)

        # Get the hilbert transform of the filtered and unfiltered signals
        self.time_domain_modulated_filtered = signal.filtfilt(*self.filter_coefficients, self.time_domain_modulated)
        self.time_domain_modulated_filtered_hilbert = self.get_hibert_transform(self.time_domain_modulated_filtered)

        self.time_domain_modulated_unfiltered_hilbert = self.get_hibert_transform(self.time_domain_modulated)

    def get_carrier(self):
        return np.cos(2 * np.pi * self.f_carrier * self.t)

    def get_modulator(self):  # TODO: Make for arbitrary bandwith
        return np.cos(2 * np.pi * self.f_modulator * self.t)

    def make_filter(self, center_freq, bandwidth):
        b, a = signal.butter(4, [center_freq - bandwidth / 2, center_freq + bandwidth / 2], btype="bandpass",
                             fs=self.fs)
        return b, a

    @staticmethod
    def get_modulated(carrier, modulator):
        return carrier * modulator

    @staticmethod
    def get_hibert_transform(sig):
        return signal.hilbert(sig)

    def make_time_domain_trace(self, signal, name):
        return go.Scatter(x=self.t, y=signal, name=name)

    def make_frequency_domain_magnitude_trace(self, complex_spectrum, name):
        return go.Scatter(x=self.freq_range, y=np.abs(complex_spectrum), name=name)

    def make_time_domain_signal_components_plot(self):
        fig = go.Figure()
        fig.add_trace( self.make_time_domain_trace(self.time_domain_carrier, "Carrier"))
        fig.add_trace(self.make_time_domain_trace(self.time_domain_modulator, "Modulating"))
        fig.add_trace(self.make_time_domain_trace(self.time_domain_modulated, "Modulated"))

        # Add title
        fig.update_layout(title="Time domain signals")

        # Add the axis labels
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")
        return fig

    def make_frequency_domain_signal_components_plot(self):
        fig = go.Figure()
        # Show the respective frequency domain signals
        fig.add_trace(self.make_frequency_domain_magnitude_trace(self.frequency_domain_carrier, "Carrier"))
        fig.add_trace(self.make_frequency_domain_magnitude_trace(self.frequency_domain_modulator, "Modulating"))
        fig.add_trace(self.make_frequency_domain_magnitude_trace(self.frequency_domain_modulated, "Modulated"))

        # Show the frequency response of the filter
        w, h = signal.freqz(*self.filter_coefficients, worN=self.freq_range, fs=self.fs)
        fig.add_trace(self.make_frequency_domain_magnitude_trace(h, "Filter"))

        # Add title
        fig.update_layout(title="Frequency domain signals")

        # Add the axis labels
        fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        return fig

    def make_processed_time_series_plot(self):
        fig = go.Figure()

        # Show original modulated time series and its analytical envelope
        fig.add_trace(self.make_time_domain_trace(self.time_domain_modulated, "Modulated signal"))
        fig.add_trace(self.make_time_domain_trace(np.abs(self.time_domain_modulated_unfiltered_hilbert),
                                                  "Unfiltered analytical \n signal magnitude"))

        # Show filtered modulated time series and its analytical envelope
        fig.add_trace(self.make_time_domain_trace(self.time_domain_modulated_filtered, "Filtered modulated signal"))
        fig.add_trace(self.make_time_domain_trace(np.abs(self.time_domain_modulated_filtered_hilbert),
                                                  "Filtered analytical \n signal magnitude"))

        fig.update_layout(title="Filtered and unfiltered signal envelopes")
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")
        return fig
