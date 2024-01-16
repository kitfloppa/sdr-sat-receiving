from __future__ import annotations
from typing import List, Optional

import os
import argparse
import numpy as np

from PIL import Image
from scipy.io import wavfile
from scipy.signal import hilbert, resample


# Width of image components after reconstruction
# See https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)
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

# Sequence for alignment
# https://www.sigidwiki.com/wiki/Automatic_Picture_Transmission_(APT)
SYNCHRONIZATION_SEQUENCE = np.array([0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 255, 255,
                                     0, 0, 255, 255, 0, 0, 0, 0, 0,
                                     0, 0, 0]) - 128

RESAMPLE_RATE = 20800


class NOAADecoder:
    def __init__(self,
                 black_point: int,
                 white_point: int,
                 components: Optional(List[str]),
                ) -> None:
    
        self.black_point = black_point
        self.white_point = white_point
        self.components = components


    @staticmethod
    def audio_to_hilbert(file_path) -> np.ndarray:
        '''
        Load the audio and convert to Hilbert transformed amplitude info

        :return: amplitude information corresponding to pixel intensity
        '''

        rate, audio = wavfile.read(file_path)

        # Resample audio at appropriate rate (20800)
        coef = RESAMPLE_RATE / rate
        samples = int(coef * len(audio))
        audio = resample(audio, samples)

        # If two-channel audio, average across channels
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Hilbert transform audio and extract envelope information
        return np.abs(hilbert(audio))


    @staticmethod
    def subsample(data: np.ndarray, step: np.ndarray = 5) -> None:
        return resample(data, len(data) // step)


    @staticmethod
    def quantize(data: np.ndarray, black_point: int, white_point: int) -> np.ndarray:
        '''
        Digitize signal to valid pixel intensities in the uint8 range

        :param black_point: dynamic range lower bound, percent
        :param white_point: dynamic range upper bound, percent
        :return: quantized numpy array to uint8
        '''

        # Percent for upper and lower saturation
        low, high = np.percentile(data, (black_point, white_point))

        # Range adjustment and quantization and cast to 8-bit range
        return np.round((255 * (data - low)) / (high - low)).clip(0, 255).astype(np.uint8)
    

    @staticmethod
    def reshape(data: np.ndarray,
                synchronization_sequence: np.ndarray = SYNCHRONIZATION_SEQUENCE,
                minimum_row_separation: int = 2000) -> np.ndarray:
        '''
        Reshape the numpy array to a 2D image array

        :param synchronization_sequence: sequence to indicate row start
        :param minimum_row_separation: minimum columns of separation
            (a hair less than 2080)
        :return: a 2D reshaped image array
        '''

        # Initialize
        rows, previous_corr, previous_ind = [None], -np.inf, 0

        for current_loc in range(len(data) - len(synchronization_sequence)):

            # Proposed start of row, normalized to zero
            row = [x - 128 for x in data[current_loc : current_loc + len(synchronization_sequence)]]

            # Correlation between the row and the synchronization sequence
            temp_corr = np.dot(synchronization_sequence, row)

            # If you're past the minimum separation, start hunting for the next synch
            if current_loc - previous_ind > minimum_row_separation:
                previous_corr, previous_ind = -np.inf, current_loc
                rows.append(data[current_loc : current_loc + 2080])

            # If the proposed region matches the sequence better, update
            elif temp_corr > previous_corr:
                previous_corr, previous_ind = temp_corr, current_loc
                rows[-1] = data[current_loc : current_loc + 2080]

        # Stack the row to form the image, drop the incomplete rows at the end
        return np.vstack([row for row in rows if len(row) == 2080])


    @staticmethod
    def select_image_components(data: np.ndarray, 
                                components: Optional[List[str]] = ['image_a', 'image_b']) -> np.ndarray:
        '''
        Select the image components to include

        :param components: portions of the image to preserve/filter
        :return: image array with just the appropriate image components
        '''

        # Image array components
        image_regions = []

        # If there are no components, return the full image
        if components is None:
            return data

        # Image components to include, based on column down selection
        for component in components:
            component_start, component_end = COMPONENT_SIZES[component]
            image_regions.append(data[:, component_start:component_end])

        return np.hstack(image_regions)
    

    def decode(self, file_path: str, out_file_path: str) -> None:
        # Load the audio and convert to Hilbert transformed amplitude info
        data = self.audio_to_hilbert(file_path)

        # Sampling from the Hilbert transformed signal for desired images
        data = self.subsample(data)

        # Digitize signal to valid pixel intensities in the uint8 range
        data = self.quantize(data, self.black_point, self.white_point)

        # Reshape the numpy array to a 2D image array
        data = self.reshape(data)

        # Select the image components to include
        image = self.select_image_components(data, self.components)

        img = Image.fromarray(image.astype(np.uint8))
        img.save(out_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='NOAA Decoder',
                        description='Decode WAV data to image.',
                        add_help=False)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='Show this help message and exit.')

    parser.add_argument('-i', '--input_file', required=True,
                        help='Input *.WAV APT file.')

    parser.add_argument('-bp', '--black_point', default=5,
                        help='Black point value.', type=int)

    parser.add_argument('-wp', '--white_point', default=95,
                        help='White point value.', type=int)
    
    parser.add_argument('-c', '--components', default=['image_a', 'image_b'],
                        action='append',
                        help='Names of pieces which would in output image.')

    parser.add_argument('-o', '--out_dir', required=True,
                        help='Output directory path.')
    
    args = parser.parse_args()
    
    standart_noaa_decoder = NOAADecoder(args.black_point, args.white_point, args.components)
    standart_noaa_decoder.decode(args.input_file, args.out_dir)
