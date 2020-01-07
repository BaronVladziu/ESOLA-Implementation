# Imports
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import scipy.io.wavfile

from ESOLA import ESOLA


# Parameters
min_voice_frequency = 60
wave_path = 'wavs/SpeechTest3.wav'
frame_length_multiplier = 3  # Must be greater than 1 (value of about 3 is proposed in the paper)
prefered_number_of_overlapping_frames = 2  # It's called 'prefered', because algorithm may raise this value to prevent crash

time_change_factor = 0.8
pitch_shift_factor = 1.1

# Cut wave
wave, sample_frequency = soundfile.read(wave_path)

# ESOLA
modified_wave = ESOLA(
    wave, time_change_factor,
    pitch_shift_factor,
    frame_length_multiplier,
    prefered_number_of_overlapping_frames,
    min_voice_frequency,
    sample_frequency
)

# Print wave
plt.figure()
plt.plot(modified_wave)
plt.show()

# Save wave
scipy.io.wavfile.write('wavs/modified-test-wav.wav', sample_frequency, modified_wave)
