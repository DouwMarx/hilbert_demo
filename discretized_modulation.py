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
                               "filter_bandwidth": {"min": 1, "max": 30, "step": 0.1, "id": "filter_bandwidth",
                                                    "value": 10},
                               "amplitude_sensitivity": {"min": 0.1, "max": 1, "step": 0.1,
                                                         "id": "amplitude_sensitivity",
                                                         "value": 0.6},
                               "number_of_modulator_components": {"min": 1, "max": 10, "step": 1,
                                                                  "id": "number_of_modulator_components",
                                                                  "value": 1},
                               "modulator_bandwidth": {"min": 0.1, "max": 20, "step": 0.1, "id": "modulator_bandwidth",
                                                       "value": 10},
                               }

        # Set the initial state
        self.update_state(self.parameter_grid["f_carrier"]["value"], self.parameter_grid["f_modulator"]["value"],
                          self.parameter_grid["filter_center_freq_error"]["value"],
                          self.parameter_grid["filter_bandwidth"]["value"],
                          self.parameter_grid["amplitude_sensitivity"]["value"],
                          self.parameter_grid["number_of_modulator_components"]["value"],
                          self.parameter_grid["modulator_bandwidth"]["value"]
                          )

    def update_state(self, f_carrier,
                     f_modulator,
                     filter_center_freq_error,
                     filter_bandwidth,
                     amplitude_sensitivity,
                     n_modulator_components,
                     modulator_bandwidth
                     ):
        # Set the parameters
        self.f_carrier = f_carrier
        self.f_modulator = f_modulator
        self.filter_center_freq_error = filter_center_freq_error
        self.filter_bandwidth = filter_bandwidth
        self.r = amplitude_sensitivity
        self.n_modulator_components = n_modulator_components
        self.modulator_bandwidth = modulator_bandwidth

        # Get the list of modulator frequencies if there are multiple components
        if self.n_modulator_components > 1:
            # Make random frequencies between -0.5 and 0.5 and then scale them to the modulator bandwidth
            np.random.seed(0)
            random_numbers = np.random.rand(self.n_modulator_components) - 0.5
            self.modulator_frequencies = random_numbers * self.modulator_bandwidth + self.f_modulator

            # Make random amplitudes between 0 and 1
            self.modulator_amplitudes = np.random.rand(self.n_modulator_components)

            # Make random phases between 0 and 2pi
            self.modulator_phases = np.random.rand(self.n_modulator_components) * 2 * np.pi
        else:
            self.modulator_frequencies = [self.f_modulator]
            self.modulator_amplitudes = [1]
            self.modulator_phases = [0]

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
        self.time_domain_modulated_filtered_analytical = self.get_analytical_signal(self.time_domain_modulated_filtered)

        self.time_domain_modulated_unfiltered_analytical = self.get_analytical_signal(self.time_domain_modulated)

        # Get the frequency representations of the filtered and unfiltered signals after hilbert transform
        self.frequency_domain_modulated_filtered_analytical_magnitude = np.fft.rfft(
            np.abs(self.time_domain_modulated_filtered_analytical)) / len(
            self.time_domain_modulated_filtered_analytical)
        self.frequency_domain_modulated_unfiltered_analytical_magnitude = np.fft.rfft(
            np.abs(self.time_domain_modulated_unfiltered_analytical)) / len(
            self.time_domain_modulated_unfiltered_analytical)

    def get_carrier(self):
        return np.cos(2 * np.pi * self.f_carrier * self.t)

    def get_modulator(self):  # TODO: Make for arbitrary bandwith
        # return 1 + self.r * np.cos(2 * np.pi * self.f_modulator * self.t)
        modulator = np.zeros(len(self.t))
        for freq, amp, phase in zip(self.modulator_frequencies, self.modulator_amplitudes, self.modulator_phases):
            modulator += amp * np.cos(2 * np.pi * freq * self.t + phase)

        # Rescale signal to be between -1 and 1
        modulator = (modulator - np.min(modulator)) / (np.max(modulator) - np.min(modulator))
        modulator = modulator * 2 - 1

        return 1 + self.r * modulator

    def make_filter(self, center_freq, bandwidth):
        b, a = signal.butter(4, [center_freq - bandwidth / 2, center_freq + bandwidth / 2], btype="bandpass",
                             fs=self.fs)
        return b, a

    @staticmethod
    def get_modulated(carrier, modulator):
        return carrier * modulator

    @staticmethod
    def get_analytical_signal(sig):
        # Compute the analytical signal using the hilbert transform
        return signal.hilbert(sig)

    def make_time_domain_trace(self, signal, name):
        return go.Scatter(x=self.t, y=signal, name=name)

    def make_frequency_domain_magnitude_trace(self, complex_spectrum, name):
        return go.Scatter(x=self.freq_range, y=np.abs(complex_spectrum), name=name)

    def make_time_domain_signal_components_plot(self):
        fig = go.Figure()
        fig.add_trace(self.make_time_domain_trace(self.time_domain_carrier, "Carrier"))
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
        fig.add_trace(self.make_time_domain_trace(np.abs(self.time_domain_modulated_unfiltered_analytical),
                                                  "Unfiltered analytical \n signal magnitude"))

        # Show filtered modulated time series and its analytical envelope
        fig.add_trace(self.make_time_domain_trace(self.time_domain_modulated_filtered, "Filtered modulated signal"))
        fig.add_trace(self.make_time_domain_trace(np.abs(self.time_domain_modulated_filtered_analytical),
                                                  "Filtered analytical \n signal magnitude"))

        fig.update_layout(title="Filtered and unfiltered signal envelopes")
        fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude")
        return fig

    def make_processed_frequency_domain_plot(self):
        fig = go.Figure()

        # Show the frequency domain of original modulating signal
        fig.add_trace(self.make_frequency_domain_magnitude_trace(self.frequency_domain_modulator, "Modulating signal"))

        # Show the frequency domain of the recovered modulating signal with and without filtering around the carrier
        fig.add_trace(
            self.make_frequency_domain_magnitude_trace(self.frequency_domain_modulated_filtered_analytical_magnitude,
                                                       "Filtered recovered \n modulating signal"))
        fig.add_trace(
            self.make_frequency_domain_magnitude_trace(self.frequency_domain_modulated_unfiltered_analytical_magnitude,
                                                       "Unfiltered recovered \n modulating signal"))

        fig.update_layout(
            title="Frequency domain of the true modulating signal and the recovered signals (DC gain not included)")
        fig.update_layout(xaxis_title="Frequency (Hz)",
                          yaxis_title="Amplitude",
                          xaxis_range=[1, 4 * self.parameter_grid["f_modulator"][
                              "max"]])  # set range up to 4 times modulating freq

        return fig
