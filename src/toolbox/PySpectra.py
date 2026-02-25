'''_____Standard imports_____'''
import numpy as np
import copy

'''_____Project imports_____'''
from src.toolbox.maths import unwrap_phase
from src.toolbox.loadings import load_data
from src.toolbox.filters import butter_highpass_filter
from src.toolbox.plottings import plots_signals


class Spectra(object):

    def __init__(self, data_dir, background_dir = None, ref_dir = None, sample_dir = None):
        self.data_dir = data_dir
        self.background_dir = background_dir
        self.ref_dir = ref_dir
        self.sample_dir = sample_dir

        # Initialize attributes to prevent plotting errors if left undefined
        self.background = None
        self.sample = None
        self.ref = None

    def load_data(self):
        """ This method serves to load the data """
        # Reshape securely to prevent dimensionality explosion
        self.raw = np.load(self.data_dir).reshape(1, 1, -1)

    def get_phase(self):
        """ This method computes the phase of the processed spectra. """
        self.phase = unwrap_phase(self.sub_raw)
        self.phase -= self.phase[0]

    def process_data(self):
        """ This method computes the processing of data """
        self.sub_raw = copy.copy(self.raw)

        if self.background_dir:
            self.background = load_data(self.background_dir).squeeze()
            self.sub_raw[0][0] += self.background

        if self.sample_dir:
            self.sample = load_data(self.sample_dir).squeeze()
            self.sub_raw[0][0] -= self.sample

        if self.ref_dir:
            self.ref = load_data(self.ref_dir).squeeze()
            self.sub_raw[0][0] -= self.ref

        self.sub_raw = butter_highpass_filter(self.sub_raw,
                                              cutoff=280,
                                              fs=40000,
                                              order=4)

    def plot(self):
        plots_signals(self.raw[0][0],
                      self.sub_raw[0][0],
                      self.ref,
                      self.sample,
                      self.background)