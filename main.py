# Imports
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import scipy.io.wavfile

from ESOLA import ESOLA


# Parameters
min_voice_frequency = 60
# wave_paths = ['wavs/Female_1.wav', 'wavs/Female_2.wav', 'wavs/Male_1.wav', 'wavs/Male_2.wav', 'wavs/Saxophone.wav']
wave_paths = ['wavs/Female_2.wav']
number_of_epochs_in_frame = 2  # Must be greater than 1 (value of about 3 is proposed in the paper)

time_changes = [0.75, 1.0, 1.25]
pitch_shift_factor = 1.0

for wave_path in wave_paths:
    # Cut wave
    wave, sample_frequency = soundfile.read(wave_path)

    for time_change in time_changes:
        # ESOLA
        modified_wave = ESOLA(
            wave, 1/time_change,
            pitch_shift_factor,
            number_of_epochs_in_frame,
            min_voice_frequency,
            sample_frequency
        )

        # # Print wave
        # plt.figure()
        # plt.plot(modified_wave)
        # plt.show()

        # Save wave
        scipy.io.wavfile.write(wave_path[:-4] + '_' + str(time_change) + '.wav', sample_frequency, modified_wave)
