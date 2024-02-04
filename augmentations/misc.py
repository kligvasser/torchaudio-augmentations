import os
import numpy as np
import librosa
import pickle


def load_dict_from_pickle(file_path):
    with open(file_path, "rb") as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


def get_files_by_extension(folder_path, extension="wav"):
    file_list = list()
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            file_path = os.path.join(folder_path, file_name)
            file_list.append(file_path)
    return file_list


def load_audio(file_path, sr=None):
    try:
        audio_data, sampling_rate = librosa.load(file_path, sr=sr)
        return audio_data, sampling_rate
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None, None


def downsample_audio(audio, original_sr, target_sr):
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)


def pad_zeros_last_axis(arr, k):
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, k)]
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def cut_random_segment_repeat(audio, segment_size):
    original_size = audio.shape[-1]

    if segment_size <= original_size:
        start_index = np.random.randint(0, original_size - segment_size + 1)
        cut_segment = audio[..., start_index : start_index + segment_size]
    else:
        repeat_factor = segment_size // original_size
        repeated_audio = np.tile(audio, repeat_factor)
        cut_segment = repeated_audio[..., :segment_size]

    return cut_segment


def cut_random_segment_zeros(audio, segment_size):
    original_size = audio.shape[-1]

    if segment_size <= original_size:
        start_index = np.random.randint(0, original_size - segment_size + 1)
        cut_segment = audio[..., start_index : start_index + segment_size]
    else:
        pad_size = segment_size - original_size
        repeated_audio = pad_zeros_last_axis(audio, pad_size)
        cut_segment = repeated_audio[..., :segment_size]

    return cut_segment
