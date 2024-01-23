from __future__ import annotations, print_function

import os
import time
import json
import struct
import datetime
import requests
import SoapySDR
import numpy as np

from configparser import ConfigParser


CONFIG_PATH = 'config.ini'

# N2YO API -> https://www.n2yo.com/api/
def get_start_record_time(satellite_id: str, lat: str, lon: str, days: int) -> tuple[datetime.datetime, datetime.datetime]:
    configuration = ConfigParser()

    if not os.path.exists(CONFIG_PATH):
        raise FileExistsError('Config file is not exist.')

    configuration.read(CONFIG_PATH)
    
    apiurl = 'https://api.n2yo.com/rest/v1/satellite/visualpasses/'
    apikey = configuration['N2YO']['apikey']

    query = apiurl + satellite_id + '/' + lat + '/' + lon + '/0/' + str(days) + '/10/&apiKey=' + apikey
    n2yo_out = requests.get(query).text
    n2yo_out_dict = json.loads(n2yo_out)

    start_time = datetime.datetime.utcfromtimestamp(n2yo_out_dict['passes'][0]['startUTC'] + 36000)
    end_time = datetime.datetime.utcfromtimestamp(n2yo_out_dict['passes'][0]['endUTC'] + 36000)
    
    return start_time, end_time


def wait_for_start(start_receiving_time: datetime.datetime) -> None:
    # Wait for the start
    while True:
        time_now = datetime.datetime.now()
        diff_time = int((start_receiving_time - time_now).total_seconds())
        
        print('Time now {:02d}:{:02d}:{:02d}: Recording will be started after {}d {}h {}m {:02d}s...'.format(
            time_now.hour, time_now.minute, time_now.second, int((diff_time / 86400)), int((diff_time / 3600) % 24),
            int((diff_time / 60) % 60), diff_time % 60
            )
        )
        
        time.sleep(5)
        
        if diff_time <= 1:
            print()
            break


def sdr_init():
    soapy_device = "rtlsdr"
    device = SoapySDR.Device({"driver": soapy_device})

    return device


def sdr_record(device, frequency, sample_rate, gain, blocks_count):
    print("Frequency:", frequency)
    print("Sample rate:", sample_rate)
    print("Gain:", gain)
    print()

    channel = 0  # Always for RTL-SDR
    device.setFrequency(SoapySDR.SOAPY_SDR_RX, channel, "RF", frequency)
    device.setGain(SoapySDR.SOAPY_SDR_RX, channel, "TUNER", gain)
    device.setGainMode(SoapySDR.SOAPY_SDR_RX, channel, False)
    device.setSampleRate(SoapySDR.SOAPY_SDR_RX, channel, sample_rate)

    data_format = SoapySDR.SOAPY_SDR_CS8 # if 'rtlsdr' in soapy_device or 'hackrf' in soapy_device else SoapySDR.SOAPY_SDR_CS16
    stream = device.setupStream(SoapySDR.SOAPY_SDR_RX, data_format, [channel], {})
    device.activateStream(stream)

    block_size = device.getStreamMTU(stream)

    # IQ: 2 digits ver variable
    buffer_format = np.int8
    buffer_size = 2 * block_size # I + Q samples
    buffer = np.empty(buffer_size, buffer_format)

    # Number of blocks to save
    block, max_blocks = 0, blocks_count

    # Save to file
    file_name = 'records/HDSDR_%s_%dkHz_RF.wav' % (datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ"), frequency / 1000)
    print("Saving file:", file_name)
    
    with open(file_name, "wb") as wav:
        # Wav data info
        bits_per_sample, subchunk_size = 16, 16
        channels_num, samples_num = 2, int(max_blocks * block_size)
        subchunk2_size = int(samples_num * channels_num * bits_per_sample / 8)
        block_alignment = int(channels_num * bits_per_sample / 8)

        # Write RIFF header
        wav.write('RIFF'.encode('utf-8'))
        wav.write(struct.pack('<i', 4 + (8 + subchunk_size) + (8 + subchunk2_size)))  # Size of the overall file
        wav.write('WAVE'.encode('utf-8'))
        
        # Write fmt subchunk
        wav.write('fmt '.encode('utf-8'))  # chunk type
        wav.write(struct.pack('<i', subchunk_size))  # subchunk data size (16 for PCM)
        wav.write(struct.pack('<h', 1))  # compression type 1 - PCM
        wav.write(struct.pack('<h', channels_num))  # channels
        wav.write(struct.pack('<i', int(sample_rate)))  # sample rate
        wav.write(struct.pack('<i', int(sample_rate * bits_per_sample * channels_num/ 8)))  # byte rate
        wav.write(struct.pack('<h', block_alignment))  # block alignment
        wav.write(struct.pack('<h', bits_per_sample))  # sample depth
        
        # Write data subchunk
        wav.write('data'.encode('utf-8'))
        wav.write(struct.pack('<i', subchunk2_size))
        
        while True:
            d_info = device.readStream(stream, [buffer], buffer_size)
            
            if d_info.ret > 0:
                data = buffer[0:2 * d_info.ret]
                fileData = data
                
                if data_format == SoapySDR.SOAPY_SDR_CS8:
                   fileData = data.astype('int16')
                wav.write(fileData)
                block += 1
                
                if block > max_blocks:
                    break

    device.deactivateStream(stream)
    device.closeStream(stream)

if __name__ == "__main__":
    print("App started")

    # NOAA 15: 137.620  - 137620000
    # NOAA 18: 137.9125 - 137912500
    # NOAA 19: 137.100  - 137100000

    noaa_19_frequency = 137100000
    noaa_19_id = '33591'

    start_record_time, end_record_time = get_start_record_time(noaa_19_id, '43.105647', '131.873504', 1)
    wait_for_start(start_record_time)

    device = sdr_init()

    t_start = time.time()

    sdr_record(device, frequency=noaa_19_frequency, sample_rate=250000, gain=35, blocks_count=10)

    print("Recording complete, time = %ds" % int(time.time() - t_start))
