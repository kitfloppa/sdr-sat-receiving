from __future__ import annotations

import os
import matplotlib

matplotlib.use('Qt5Agg')

import numpy as np
import scipy.signal
from scipy.io import wavfile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.image import imsave


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width: int = 3, height: int = 1, dpi: int = 100) -> None:
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        
        super(MplCanvas, self).__init__(fig)


class NOAADecoder:
    def __init__(self, file_path: str, norm: int) -> None:
        #extract the data from WAV file
        sampling_rate, self.signal_data = wavfile.read(file_path)

        #normalization value which will be later used to bring the amplitudes to be in 0 to 255 range
        am_envelope = self.hilbert(self.signal_data)

        self.image_path = self.signal_to_img(sampling_rate, am_envelope, norm)


    @staticmethod
    def hilbert(signal_data: np.array) -> np.array:
        #hilbert transfrom will give us an analytical signal
		#this will help us extract the envelopes instantaneously
		#for more info visit the following link
		#https://dsp.stackexchange.com/questions/25845/meaning-of-hilbert-transform

        #find the analytical signal
        analytic_signal = scipy.signal.hilbert(signal_data)

        #extract the amplitude envelope
        am_envelope = np.abs(analytic_signal)

        return am_envelope
    
    @staticmethod
    def signal_to_img(sampling_rate: int, am_envelope: np.array, norm: int) -> str:
        #calculate the width and height of the image
        width = int(sampling_rate * 0.5)
        height = am_envelope.shape[0] // width

        #create a numpy array with three channels for RGB and fill it up with zeroes
        img_data = np.zeros((height, width, 3), dtype=np.uint8)

        #keep track of pixel values
        x, y = 0, 0

        #traverse through the am_envelope and replace zeroes in numpy array with intensity values
        for i in range(am_envelope.shape[0]):
            #get the pixel intensity
            intensity = int(am_envelope[i] // norm)

            #make sure that the pixel intensity is between 0 and 255
            if intensity < 0: intensity = 0
            if intensity > 255: intensity = 255

            #put the pixel on to the image
            img_data[y][x] = intensity

            x += 1

            #if x is greater than width, sweep or jump to next line
            if x >= width:
                x, y = 0, y + 1

                if y >= height:
                    break
        
        satellite_images_path = os.path.abspath(os.getcwd()) + '/satellite_images/'
        img_path = satellite_images_path + 'sat_img.png'

        if not os.path.exists(satellite_images_path):
            os.mkdir(satellite_images_path)
        
        imsave(img_path, img_data)

        return img_path
        

    def get_signal_plot(self) -> MplCanvas:
        fig = MplCanvas()

        fig.axes.plot(self.signal_data)
        # fig.axes.xlabel("Samples")
        # fig.axes.ylabel("Amplitude")
        # fig.axes.title("Signal")

        return fig
