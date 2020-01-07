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
    x = np.zeros(len(wave))
    x[0] = wave[0]
    for i in range(1, len(wave)):
        x[i] = wave[i] - wave[i-1]
    # Print
    plt.figure(figsize=(30, 3))
    plt.plot(x)

    # First go through zero-frequency resonator
    print('First go through zero-frequency resonator...')
    y1 = np.zeros(len(x))
    y1[0] = x[0]
    y1[1] = x[1] + 2*y1[0]
    for i in range(2, len(x)):
        y1[i] = x[i] + 2*y1[i-1] - y1[i-2]
    # Print
    plt.figure(figsize=(30, 3))
    plt.plot(y1)

    # Second go through zero-frequency resonator
    print('Second go through zero-frequency resonator...')
    y2 = np.zeros(len(y1))
    y2[0] = y1[0]
    y2[1] = y1[1] + 2*y2[0]
    for i in range(2, len(y1)):
        y2[i] = y1[i] + 2*y2[i-1] - y2[i-2]
    # Print
    plt.figure(figsize=(30, 3))
    plt.plot(y2)

    # Remove trend 1
    print('First go through trend remover...')
    y3 = np.zeros(len(y2))
    for i in range(len(y2)):
        mean = 0
        considered = 0
        for j in range(-max_distance_between_epochs, max_distance_between_epochs+1):
            if i+j >= 0 and i+j < len(y2):
                mean += y2[i+j]
                considered += 1
        mean /= considered
        y3[i] = y2[i] - mean

    # Remove trend 2
    print('Second go through trend remover...')
    y = np.zeros(len(y3))
    for i in range(len(y3)):
        mean = 0
        considered = 0
        for j in range(-max_distance_between_epochs, max_distance_between_epochs+1):
            if i+j >= 0 and i+j < len(y3):
                mean += y3[i+j]
                considered += 1
        mean /= considered
        y[i] = y3[i] - mean
    # Print
    plt.figure(figsize=(30, 3))
    plt.grid(axis='both')
    plt.ylim((-10000, 10000))
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

    # Add missing epoch indices
    print('Adding missing indices...')
    if len(epoch_indices) > 0:
        i = 0
        while epoch_indices[0] > max_distance_between_epochs:
            epoch_indices.insert(0, epoch_indices[0]/2)
        while True:
            while i < len(epoch_indices) - 1:
                act_distance_between_epochs = epoch_indices[i+1] - epoch_indices[i]
                while act_distance_between_epochs > max_distance_between_epochs:
                    epoch_indices.insert(i+1, epoch_indices[i] + act_distance_between_epochs/2)  # This method sometimes misses epochs in unvoiced phones,
                                                                                                 # so it shouldn't be noticible if epoch is perfectly in it's place,
                                                                                                 # so simple mean should work just fine
                    act_distance_between_epochs = epoch_indices[i+1] - epoch_indices[i]
                i += 1
            if len(y) - epoch_indices[-1] > max_distance_between_epochs:
                epoch_indices.append(epoch_indices[-1] + (len(y) - epoch_indices[-1])/2)
            else:
                break

    # Print
    fig = np.zeros(len(y))
    for i in epoch_indices:
        fig[int(i)] = 1
    lin = np.arange(len(y))

    plt.figure(figsize=(30, 6))
    plt.grid(axis='both')
    plt.ylim((-1, 1))
    plt.plot(lin, y / 10000, 'r', lin, wave, 'g', lin, fig, 'b')

    plt.show()

    return epoch_indices


def time_stretch(wave, wav_epoch_indices, time_change_factor, frame_length_multiplier, prefered_number_of_overlapping_frames,
          max_distance_between_epochs, is_plotting_enabled=False):
    """Function for time-stretching using ESOLA algorithm

    FUNCTION SOMETIMES CRASHES WHEN time_change_function IS NOT CONSTANT
    (I couldn't find the reason :c)

    ARGUMENTS:
        wave (np.array): wave to stretch
        wav_epoch_indices (list(int)): indices of epochs in input wave
        time_change_factor (float): time stretch factor. 1 - no change, 0.5 - twice shorter wave, 2 - twice longer wave
        frame_length_multiplier (float): how many times bigger will the frame be
        prefered_number_of_overlapping_frames (int): 1 - 0% overlap, 2 - 50% overlap, 4 - 75% overlap
        max_distance_between_epochs (int): max number of samples between epochs
        is_plotting_enabled (bool): will function print figures

    RETURNS:
        synthesized_wav (np.array): stretched wave
        window_wav (np.array): window of the stretched wave

    """
    # Calculate index change function
    index_change_function = np.matrix([[0, 0], [len(wave), int(time_change_factor * len(wave))]])

    # Reserve memory
    synthesized_wav = np.zeros(int(index_change_function.item(-1, 1)), dtype=float)
    window_wav = np.zeros(int(index_change_function.item(-1, 1)), dtype=float)

    # Find real max distance between epochs
    real_max_distance_between_epochs = 0
    for i in range(1, len(wav_epoch_indices)):
        if wav_epoch_indices[i] - wav_epoch_indices[i - 1] > real_max_distance_between_epochs:
            real_max_distance_between_epochs = wav_epoch_indices[i] - wav_epoch_indices[i - 1]

    # Ensure that ESOLA will not crash (it does not work, and I don't know why ;_; )
    # For this algorithm there must be at least one epoch between the start of each frame and the end of the previous one
    frame_length = int(max_distance_between_epochs * frame_length_multiplier)

    exact_number_of_overlaping_frames_required_for_the_algorithm_not_to_crash = (
        frame_length * time_change_factor
    ) / (
        frame_length - max_distance_between_epochs
    )
    number_of_overlaping_frames = int(exact_number_of_overlaping_frames_required_for_the_algorithm_not_to_crash) + 1
    if prefered_number_of_overlapping_frames >= number_of_overlaping_frames:
        number_of_overlaping_frames = prefered_number_of_overlapping_frames

    # Calculate frame steps and indices
    number_of_frame_indices = number_of_overlaping_frames * int(
        (len(synthesized_wav) - frame_length) / frame_length) + 1
    # Analysis
    analysis_frame_step = (len(wave) - frame_length - 1) / number_of_frame_indices
    analysis_frame_indices = np.arange(0, (len(wave) - frame_length), analysis_frame_step)
    for i in range(len(analysis_frame_indices)):
        analysis_frame_indices[i] = int(analysis_frame_indices[i])

    # Find max_distance_between_frames
    max_distance_between_frames = 0
    for i in range(1, len(analysis_frame_indices)):
        if max_distance_between_frames < analysis_frame_indices[i] - analysis_frame_indices[i - 1]:
            max_distance_between_frames = analysis_frame_indices[i] - analysis_frame_indices[i - 1]

    # Synthesis
    synthesis_frame_indices = np.zeros(0)
    previous_range_index = 0
    for analysis_index in analysis_frame_indices:
        while analysis_index >= index_change_function.item(previous_range_index + 1, 0):
            previous_range_index += 1
        i = index_change_function.item(previous_range_index, 0)
        j = index_change_function.item(previous_range_index, 1)
        a = analysis_index - i
        x = index_change_function.item(previous_range_index + 1, 0) - i
        y = index_change_function.item(previous_range_index + 1, 1) - j
        c = a * y / x + j
        synthesis_frame_indices = np.concatenate([synthesis_frame_indices, [c]])

    # Find max_possible_distance_between_frames
    max_possible_distance_between_frames = 0
    for i in range(1, len(synthesis_frame_indices)):
        if max_possible_distance_between_frames < synthesis_frame_indices[i] - synthesis_frame_indices[i - 1]:
            max_possible_distance_between_frames = synthesis_frame_indices[i] - synthesis_frame_indices[i - 1]

    if frame_length - max_possible_distance_between_frames < max_distance_between_epochs:
        print("ERROR: EIOLA WILL CRASH!!!")
        print("frame_length - max_possible_distance_between_frames:",
              frame_length - max_possible_distance_between_frames)
        print("max_distance_between_epochs:", max_distance_between_epochs)
        raise ValueError("ERROR: EIOLA WILL CRASH!!!")

    # Create window
    frame_window = np.blackman(frame_length)
    # Create epoch indices by analysis frame
    epoch_indices_by_analysis_frame = list()
    act_analysis_frame_index_index = 0
    for analysis_frame_index in analysis_frame_indices:
        epoch_indices_by_analysis_frame.append(np.zeros(0))
        for wav_epoch_index in wav_epoch_indices:
            if wav_epoch_index >= analysis_frame_index and wav_epoch_index < analysis_frame_index + frame_length:
                epoch_indices_by_analysis_frame[act_analysis_frame_index_index] = np.concatenate(
                    [epoch_indices_by_analysis_frame[act_analysis_frame_index_index], np.array([wav_epoch_index])])
        act_analysis_frame_index_index += 1

    if is_plotting_enabled:
        b = 1500
        e = 2500

        # Plot
        fig = np.zeros(len(wave))
        for i in wav_epoch_indices:
            fig[int(i)] = 1
        fig_ab = np.zeros(len(wave))
        for i in analysis_frame_indices:
            fig_ab[int(i)] = 0.7
        fig_ae = np.zeros(len(wave))
        for i in analysis_frame_indices:
            fig_ae[int(i) + frame_length] = 0.8
        lin = np.arange(len(wave))

        plt.figure(figsize=(30, 3))
        plt.plot(lin[b:e], wave[b:e], 'b', lin[b:e], fig[b:e], 'r', lin[b:e], fig_ab[b:e], 'g', lin[b:e], fig_ae[b:e],
                 'c')
        plt.show()

    # Synthesize first frame
    analysis_frame = wave[:frame_length]
    windowed_analysis_frame = analysis_frame * frame_window
    synthesized_wav[:frame_length] += windowed_analysis_frame
    window_wav[:frame_length] += frame_window
    # Update first epoch of next synthesis frame index
    first_epoch_of_next_synthesis_frame_index = wav_epoch_indices[0]
    i = 1
    while first_epoch_of_next_synthesis_frame_index < synthesis_frame_indices[1]:
        first_epoch_of_next_synthesis_frame_index = wav_epoch_indices[i]
        i += 1
    # Synthesis loop
    for synthesis_frame_indices_index in range(1, len(synthesis_frame_indices)):
        # Align analysis frame with synthesis frame
        alignment = np.min(epoch_indices_by_analysis_frame[synthesis_frame_indices_index] - analysis_frame_indices[
            synthesis_frame_indices_index] + synthesis_frame_indices[
                               synthesis_frame_indices_index] - first_epoch_of_next_synthesis_frame_index)
        # Create aligned analysis frame epoch indices
        aligned_analysis_frame_epoch_indices = np.zeros(0)
        for epoch in wav_epoch_indices:
            if int(analysis_frame_indices[synthesis_frame_indices_index]) + int(alignment) <= epoch < int(
                    analysis_frame_indices[synthesis_frame_indices_index]) + int(alignment) + frame_length:
                aligned_analysis_frame_epoch_indices = np.concatenate([aligned_analysis_frame_epoch_indices, [epoch]])
        # Synthesize frame
        analysis_frame = wave[int(analysis_frame_indices[synthesis_frame_indices_index]) + int(alignment): int(
            analysis_frame_indices[synthesis_frame_indices_index]) + int(alignment) + frame_length]
        max_window_wav_frame_length = len(window_wav) - int(synthesis_frame_indices[synthesis_frame_indices_index])
        if len(analysis_frame) < frame_length or max_window_wav_frame_length < frame_length:
            new_frame_length = min(len(analysis_frame), max_window_wav_frame_length)
            new_frame_window = np.blackman(new_frame_length)
            windowed_analysis_frame = analysis_frame[:new_frame_length] * new_frame_window
            window_wav[int(synthesis_frame_indices[synthesis_frame_indices_index]): int(
                synthesis_frame_indices[synthesis_frame_indices_index]) + len(
                windowed_analysis_frame)] += new_frame_window
        else:
            windowed_analysis_frame = analysis_frame * frame_window
            window_wav[int(synthesis_frame_indices[synthesis_frame_indices_index]): int(
                synthesis_frame_indices[synthesis_frame_indices_index]) + len(windowed_analysis_frame)] += frame_window
        synthesized_wav[int(synthesis_frame_indices[synthesis_frame_indices_index]): int(
            synthesis_frame_indices[synthesis_frame_indices_index]) + len(
            windowed_analysis_frame)] += windowed_analysis_frame
        # Update first epoch of next synthesis frame index
        if synthesis_frame_indices_index < len(synthesis_frame_indices) - 1:
            i = 0
            while first_epoch_of_next_synthesis_frame_index < synthesis_frame_indices[
                synthesis_frame_indices_index + 1]:
                # If i is out of range here then frame_length is too short or some epochs were not found (code at the begining of the algorithm should have prevented it)
                first_epoch_of_next_synthesis_frame_index = aligned_analysis_frame_epoch_indices[i] - \
                                                            analysis_frame_indices[
                                                                synthesis_frame_indices_index] - alignment + \
                                                            synthesis_frame_indices[synthesis_frame_indices_index]
                i += 1
    # Normalize
    for i in range(len(window_wav)):
        if window_wav[i] < 0.0001:
            window_wav[i] = 0.0001  # Avoiding potential dividing by 0
    synthesized_wav /= window_wav
    return synthesized_wav, window_wav


def ESOLA(wave, time_change_factor, pitch_shift_factor, frame_length_multiplier, prefered_number_of_overlapping_frames,
          min_voice_frequency, sample_frequency):

    # Extract epochs
    print('1) FINDING EPOCHS...')
    epoch_indices = extract_epoch_indices(wave, min_voice_frequency, sample_frequency)

    # Stretch wave
    print('2) STRETCHING WAVE...')
    max_distance_between_epochs = int(1 / float(min_voice_frequency) * sample_frequency) + 1
    stretched_wave, window_wave = time_stretch(
        wave,
        epoch_indices,
        time_change_factor * pitch_shift_factor,
        frame_length_multiplier,
        prefered_number_of_overlapping_frames,
        max_distance_between_epochs
    )

    # Resample wave
    print('3) RESAMPLING WAVE...')
    resampled_wave = scipy.signal.resample(stretched_wave,  int(len(stretched_wave) / pitch_shift_factor))

    return resampled_wave
