# Imports
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import scipy.io.wavfile

from ESOLA import ESOLA


# Parameters
min_voice_frequency = 60
wave_path = 'wavs/SpeechTest3.wav'
number_of_epochs_in_frame = 2  # Must be greater than 1 (value of about 3 is proposed in the paper)

time_change_factor = 2.0
pitch_shift_factor = 1.0

# Cut wave
wave, sample_frequency = soundfile.read(wave_path)

# ESOLA
modified_wave = ESOLA(
    wave, time_change_factor,
    pitch_shift_factor,
    number_of_epochs_in_frame,
    min_voice_frequency,
    sample_frequency
)

# Print wave
plt.figure()
plt.plot(modified_wave)
plt.show()

# Save wave
scipy.io.wavfile.write('wavs/modified-test-wav.wav', sample_frequency, modified_wave)
