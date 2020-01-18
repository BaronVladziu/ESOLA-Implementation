import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def extract_epoch_indices(wave, min_voice_frequency, sample_frequency):
    """Function for extracting epoch indices from wave

    ARGUMENTS:
        wave (np.array): input wave
        min_voice_frequency (float): minimal possible fundamental frequency in the wave
        sample_frequency (int): sampling frequency of the wave

    RETURNS:
        epoch_indices (list(int)): epoch indices

    """
    max_distance_between_epochs = int(1/float(min_voice_frequency)*sample_frequency)

    # Preprocession
    print('Preprocessing...')
    x = np.zeros(len(wave), dtype=np.longdouble)
    x[0] = wave[0]
    for i in range(1, len(wave)):
        x[i] = wave[i] - wave[i-1]
    x /= np.max(np.abs(x))
    # Print
    plt.figure(figsize=(30, 3))
    plt.plot(x)

    # First go through zero-frequency resonator
    print('First go through zero-frequency resonator...')
    y1 = np.zeros(len(x), dtype=np.longdouble)
    y1[0] = x[0]
    y1[1] = x[1] + 2*y1[0]
    for i in range(2, len(x)):
        y1[i] = x[i] + 2*y1[i-1] - y1[i-2]
    y1 /= np.max(np.abs(y1))
    # Print
    plt.figure(figsize=(30, 3))
    plt.plot(y1)

    # Second go through zero-frequency resonator
    print('Second go through zero-frequency resonator...')
    y2 = np.zeros(len(y1), dtype=np.longdouble)
    y2[0] = y1[0]
    y2[1] = y1[1] + 2*y2[0]
    for i in range(2, len(y1)):
        y2[i] = y1[i] + 2*y2[i-1] - y2[i-2]
    y2 /= np.max(np.abs(y2))
    # Print
    plt.figure(figsize=(30, 3))
    plt.plot(y2)

    # Remove trend 1
    print('First go through trend remover...')
    window_length = int(0.005 * sample_frequency)
    # window_length = 1
    y3 = np.zeros(len(y2), dtype=np.longdouble)
    for i in range(len(y2)):
        if i-window_length < 0:
            mean = y2[i]
        elif i+window_length >= len(y2):
            mean = y2[i]
        else:
            mean = np.mean(y2[i - window_length : i + window_length + 1])
        y3[i] = y2[i] - mean
    assert y3[-1] == 0, str(y3[-1])
    y3 /= np.max(np.abs(y3))

    # Remove trend 2
    print('Second go through trend remover...')
    window_length = int(0.005 * sample_frequency)
    # window_length = 1
    y = np.zeros(len(y3), dtype=np.longdouble)
    for i in range(len(y3)):
        if i - window_length < 0:
            mean = y3[i]
        elif i + window_length >= len(y3):
            mean = y3[i]
        else:
            mean = np.mean(y3[i - window_length: i + window_length + 1])
        y[i] = y3[i] - mean
    assert y[-1] == 0, str(y[-1])
    y /= np.max(np.abs(y))

    # Plot
    plt.figure(figsize=(30, 3))
    plt.grid(axis='both')
    plt.plot(y)

    # Extract epoch indices
    print('Extracting epoch indices...')
    epoch_indices = list()
    last = y[0]
    for i in range(len(y)):
        act = y[i]
        if last < 0 and act > 0:
            epoch_indices.append(i)
        last = act

    # # Add missing epoch indices
    # print('Adding missing indices...')
    # if len(epoch_indices) > 0:
    #     i = 0
    #     while epoch_indices[0] > max_distance_between_epochs:
    #         epoch_indices.insert(0, epoch_indices[0]/2)
    #     while True:
    #         while i < len(epoch_indices) - 1:
    #             act_distance_between_epochs = epoch_indices[i+1] - epoch_indices[i]
    #             while act_distance_between_epochs > max_distance_between_epochs:
    #                 epoch_indices.insert(i+1, epoch_indices[i] + act_distance_between_epochs/2)  # This method sometimes misses epochs in unvoiced phones,
    #                                                                                              # so it shouldn't be noticible if epoch is perfectly in it's place,
    #                                                                                              # so simple mean should work just fine
    #                 act_distance_between_epochs = epoch_indices[i+1] - epoch_indices[i]
    #             i += 1
    #         if len(y) - epoch_indices[-1] > max_distance_between_epochs:
    #             epoch_indices.append(epoch_indices[-1] + (len(y) - epoch_indices[-1])/2)
    #         else:
    #             break

    # Print
    fig = np.zeros(len(y))
    for i in epoch_indices:
        fig[int(i)] = 1
    lin = np.arange(len(y))

    plt.figure(figsize=(30, 6))
    plt.grid(axis='both')
    plt.ylim((-1, 1))
    plt.plot(lin, y, 'r', lin, wave, 'g', lin, fig, 'b')

    plt.show()

    return epoch_indices


def time_stretch(wave, wav_epoch_indices, time_change_factor, number_of_epochs_in_frame, is_plotting_enabled=False):
    """Function for time-stretching using ESOLA algorithm

    ARGUMENTS:
        wave (np.array): wave to stretch
        wav_epoch_indices (list(int)): indices of epochs in input wave
        time_change_factor (float): time stretch factor. 1 - no change, 0.5 - twice shorter wave, 2 - twice longer wave
        number_of_epochs_in_frame (int): how many epochs will be contained in one frame
        is_plotting_enabled (bool): will function print figures

    RETURNS:
        synthesized_wav (np.array): stretched wave
        window_wav (np.array): window of the stretched wave

    """
    # Analysis
    analysis_frame_indices = [0]
    for epoch_index in wav_epoch_indices:
        analysis_frame_indices.append(int(epoch_index))
    analysis_frame_indices.append(len(wave))

    wav_frames = list()
    window_frames = list()
    for i in range(len(analysis_frame_indices) - number_of_epochs_in_frame):
        frame_length = int(analysis_frame_indices[i+number_of_epochs_in_frame] - analysis_frame_indices[i] - 1)
        window = np.blackman(frame_length)
        wav_frames.append(wave[analysis_frame_indices[i]:analysis_frame_indices[i]+frame_length]*window)
        window_frames.append(window)

    # Synthesis
    target_length = 0
    last_epoch_index = 0
    synthesized_wav = np.zeros(0)
    window_wav = np.zeros(0)
    for i in range(len(analysis_frame_indices) - number_of_epochs_in_frame):
        assert len(wav_frames[i]) == len(window_frames[i]), "ERROR: Wave and window frames have different length!"
        hop = analysis_frame_indices[i+1] - analysis_frame_indices[i]
        while target_length >= len(synthesized_wav):
            # Increase buffers
            buffer_increase = len(wav_frames[i]) - len(synthesized_wav) + last_epoch_index
            if buffer_increase > 0:
                synthesized_wav = np.concatenate([synthesized_wav, np.zeros(buffer_increase)])
                window_wav = np.concatenate([window_wav, np.zeros(buffer_increase)])
            # Add new frame
            synthesized_wav[last_epoch_index:last_epoch_index + len(wav_frames[i])] += wav_frames[i]
            window_wav[last_epoch_index:last_epoch_index + len(wav_frames[i])] += window_frames[i]

            # Update markers
            last_epoch_index += hop
        target_length += int(hop * time_change_factor)

    # Normalize
    for i in range(len(window_wav)):
        if window_wav[i] < 0.0001:
            window_wav[i] = 0.0001  # Avoiding potential dividing by 0
    synthesized_wav /= window_wav

    # # Plot
    # if is_plotting_enabled:
    #     b = 1500
    #     e = 2500
    #
    #     # Plot
    #     fig = np.zeros(len(wave))
    #     for i in wav_epoch_indices:
    #         fig[int(i)] = 1
    #     lin = np.arange(len(wave))
    #
    #     plt.figure(figsize=(30, 3))
    #     plt.plot(lin[b:e], wave[b:e], 'b', lin[b:e], fig[b:e], 'r')
    #     plt.show()

    return synthesized_wav


def ESOLA(wave, time_change_factor, pitch_shift_factor, number_of_epochs_in_frame,
          min_voice_frequency, sample_frequency):

    # Extract epochs
    print('1) FINDING EPOCHS...')
    epoch_indices = extract_epoch_indices(wave, min_voice_frequency, sample_frequency)

    # Stretch wave
    print('2) STRETCHING WAVE...')
    stretched_wave = time_stretch(
        wave,
        epoch_indices,
        time_change_factor * pitch_shift_factor,
        number_of_epochs_in_frame,
    )

    # Resample wave
    print('3) RESAMPLING WAVE...')
    resampled_wave = scipy.signal.resample(stretched_wave,  int(len(stretched_wave) / pitch_shift_factor))

    return resampled_wave
