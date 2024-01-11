from __future__ import annotations
from typing import List, Optional

import os
import matplotlib

matplotlib.use('Qt5Agg')

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, resample

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PIL import Image


# width of image components after reconstruction
# see https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)
COMPONENT_SIZES = {
    "sync_a": (0, 39),
    "space_a": (39, 86),
    "image_a": (86, 995),
    "telemetry_a": (995, 1040),
    "sync_b": (1040, 1079),
    "space_b": (1079, 1126),
    "image_b": (1126, 2035),
    "telemetry_b": (2035, 2080),
}

# sequence for alignment
# https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)
SYNCHRONIZATION_SEQUENCE = np.array([0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 0, 0, 0,
                                     0, 0, 0]) - 128


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width: int = 3, height: int = 1, dpi: int = 100) -> None:
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(fig)


class NOAADecoder:
    def __init__(self, file_path: str, black_point: int = 5, white_point: int = 95) -> None:
        self.file_path = file_path

        # load the audio and convert to Hilbert transformed amplitude info
        self.audio_to_hilbert()

        # sampling from the Hilbert transformed signal for desired images
        self.subsample()

        # digitize signal to valid pixel intensities in the uint8 range
        self.quantize(black_point, white_point)

        # reshape the numpy array to a 2D image array
        self.reshape()

        # some empirically based filters for noisy rows
        self.filter_noisy_rows()

        # select the image components to include
        self.select_image_components()

        satellite_images_path = os.path.abspath(os.getcwd()) + '/satellite_images/'
        self.img_save_path = satellite_images_path + 'sat_img.png'

        if not os.path.exists(satellite_images_path):
            os.mkdir(satellite_images_path)
        
        
        img = Image.fromarray(self.image.astype(np.uint8))
        img.save(self.img_save_path)


    def audio_to_hilbert(self, resample_rate: int = 20800) -> None:
        '''
        Load the audio and convert to Hilbert transformed amplitude info

        :param in_path: string to path of audio file, relative or absolute
        :param resample_rate: rate to resample audio at
        :return: amplitude information corresponding to pixel intensity
        '''

        rate, self.audio = wavfile.read(self.file_path)

        # resample audio at appropriate rate (20800)
        coef = resample_rate / rate
        samples = int(coef * len(self.audio))
        self.audio = resample(self.audio, samples)

        # if two-channel audio, average across channels
        if self.audio.ndim > 1:
            self.audio = np.mean(self.audio, axis=1)

        # hilbert transform audio and extract envelope information
        self.hilbert_transformed = np.abs(hilbert(self.audio))


    def subsample(self, step: np.ndarray = 5) -> None:
        self.subsampled = resample(self.hilbert_transformed, len(self.hilbert_transformed) // step)

    
    def quantize(self, black_point: int, white_point: int) -> np.ndarray:
        '''
        Digitize signal to valid pixel intensities in the uint8 range

        :param arr: numpy array of continuous signal
        :param black_point: dynamic range lower bound, percent
        :param white_point: dynamic range upper bound, percent
        :return: quantized numpy array to uint8
        '''

        #  percent for upper and lower saturation
        low, high = np.percentile(self.subsampled, (black_point, white_point))

        # range adjustment and quantization and cast to 8-bit range
        self.quantized = np.round((255 * (self.subsampled - low)) / (high - low)).clip(0, 255).astype(np.uint8)
    

    def reshape(self,
                synchronization_sequence: np.ndarray = SYNCHRONIZATION_SEQUENCE,
                minimum_row_separation: int = 2000) -> None:
        '''
        Reshape the numpy array to a 2D image array

        :param arr: 1D numpy array to arrange into 2D image array
        :param synchronization_sequence: sequence to indicate row start
        :param minimum_row_separation: minimum columns of separation
            (a hair less than 2080)
        :return: a 2D reshaped image array
        '''

        # initialize
        rows, previous_corr, previous_ind = [None], -np.inf, 0

        for current_loc in range(len(self.quantized) - len(synchronization_sequence)):

            # proposed start of row, normalized to zero
            row = [x - 128 for x in self.quantized[current_loc : current_loc + len(synchronization_sequence)]]

            # correlation between the row and the synchronization sequence
            temp_corr = np.dot(synchronization_sequence, row)

            # if you're past the minimum separation, start hunting for the next synch
            if current_loc - previous_ind > minimum_row_separation:
                previous_corr, previous_ind = -np.inf, current_loc
                rows.append(self.quantized[current_loc : current_loc + 2080])

            # if the proposed region matches the sequence better, update
            elif temp_corr > previous_corr:
                previous_corr, previous_ind = temp_corr, current_loc
                rows[-1] = self.quantized[current_loc : current_loc + 2080]

        # stack the row to form the image, drop the incomplete rows at the end
        self.reshaped = np.vstack([row for row in rows if len(row) == 2080])

    
    @staticmethod
    def signal_to_noise(arr: np.ndarray, axis=0) -> np.ndarray:
        """
        Signal to noise (SNR) calculation (previously included in scipy)

        :param arr: array to calculate signal to noise against
        :param axis: axis to perform SNR calculation over
        :return: signal to noise ratio calculated along axis
        """

        # mean and standard deviation along axis
        mean, std = arr.mean(axis), arr.std(axis=axis)

        # SNR calculation along axis
        return np.where(std == 0, 0, mean / std)


    def filter_noisy_rows(self) -> None:
        '''
        Some empirically based filters for noisy rows

        :param arr: 2D image array
        :return: image with filtered noisy rows
        '''

        # calculate signal to noise and the row to row difference in SNR
        snr = self.signal_to_noise(self.reshaped, axis=1)
        snr_diff = np.diff(snr, prepend=0)

        # image filter for rows with high snr (pure noise) and minimal distance
        # in SNR between rows (no theoretical basis, just seems to work)
        self.denoised = self.reshaped[(snr > 0.6) & (snr_diff < 0.05) & (snr_diff > -0.05) & (snr_diff != 0), :]
        #self.denoised = self.reshaped[:, :]


    def select_image_components(self, 
                                components: Optional[List[str]] = ['image_a', 'image_b']) -> None:
        '''
        Select the image components to include

        :param arr: 2D image array
        :param components: portions of the image to preserve/filter
        :return: image array with just the appropriate image components
        '''

        # image array components
        image_regions = []

        # if there are no components, return the full image
        if components is None:
            return self.denoised

        # image components to include, based on column down selection
        for component in components:
            component_start, component_end = COMPONENT_SIZES[component]
            image_regions.append(self.denoised[:, component_start:component_end])

        self.image = np.hstack(image_regions)
        

    def get_signal_plot(self) -> MplCanvas:
        fig = MplCanvas()

        fig.axes.plot(self.audio)
        # fig.axes.xlabel("Samples")
        # fig.axes.ylabel("Amplitude")
        # fig.axes.title("Signal")

        return fig
